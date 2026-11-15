"""UTKFace 年龄回归：论文 1（控制头）与论文 2（EMA 时机）的主实验。

两个开关决定所有实验：--head / --ema-mode。其余超参对所有组别必须一致。

数据集轴：--dataset（默认 utkface，records/split/零基线/目录名与改造前逐字节相同；
koniq 的 records 走 --index 的索引 csv，raw 默认换成 KonIQ 的 memmap 缓存）。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import dump_fingerprint as fp
from data import cifar as cifar10
from data import koniq
from data.base import ImageScalarDataset
from data.utkface import scan, split_random
from models.net import DrivingNet
from models.resnet_reg import ResNetReg
from quant.ema_snap import SnapEMA
from quant.qat import snap_weights_
from quant.ste import plain_keys, register_ste_

KONIQ_INDEX = "data/koniq_index.csv"
KONIQ_RAW = "data/koniq_raw224.npy"


def run_tag(arch: str, quant_mode: str) -> str:
    """run 目录名的前缀：ste- 标前向伪量化，{arch}- 标非 DrivingNet 的架构轴。

    默认组合（arch=net + overwrite）返回空串，目录名与改造前逐字节相同。
    """
    return ("ste-" if quant_mode == "ste" else "") + ("" if arch == "net" else f"{arch}-")


def read_index_csv(path: str) -> list[dict]:
    """KonIQ records：直接读索引 csv（path/group/idx/mos/target/row），row 是 memmap 行号。

    不走 data.koniq.scan 的 glob 路径 —— 那要对 10,373 个文件名逐个查 MOS 表，本机已有
    索引就不必重算；target 与 scan 的 mos/100 逐位一致（见自测）。
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"path": r["path"], "group": r["group"], "idx": int(r["idx"]),
                         "mos": float(r["mos"]), "target": float(r["target"]),
                         "row": int(r["row"])})
    return rows


QMAX = 127.0        # quant.qat.snap_weights_ 的默认 8 bit 网格（qmax = 2^(8-1) - 1）


def _wmods(model) -> list:
    """参与投影的全部权重模块（与 quant.qat.snap_weights_ 同一判据）。"""
    return [m for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]


def _dist(vals: list) -> dict:
    """一组数的分布摘要（p10/p50/p90 用最近秩，不引入 numpy）。"""
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"n": n, "mean": sum(s) / n, "p10": q(0.10), "p50": q(0.50),
            "p90": q(0.90), "min": s[0], "max": s[-1]}


@torch.no_grad()
def _mech_pre(wmods, prev) -> dict:
    """opt.step() 之后、投影之前：每层 ||u_t|| 与格距 delta（按输出通道 |w|max/qmax，取通道均值）。

    每层先在 GPU 上算成 0 维张量再 stack，最后一次 .cpu() 搬回——避免每层一次同步。
    """
    us, ds, uv = [], [], []
    for m, w0 in zip(wmods, prev):
        w = m.weight.detach()
        du = w - w0
        us.append(du.norm())
        uv.append(du.reshape(-1))
        ds.append((w.reshape(w.shape[0], -1).abs().amax(dim=1).clamp_min(1e-8) / QMAX).mean())
    return {"u": torch.stack(us).cpu().tolist(), "delta": torch.stack(ds).cpu().tolist(),
            "uvec": uv}


@torch.no_grad()
def _mech_write(f, wmods, prev, pre, opt, lr: float, ep: int, step_i: int) -> None:
    """投影之后：被网格吃掉的更新比例 + Adam 的 m/v 摘要，一行 jsonl。"""
    g = opt.param_groups[0]
    b1, b2, eps = g["betas"][0], g["betas"][1], g["eps"]
    sts = [opt.state.get(m.weight) for m in wmods]
    t_all = [s["step"] for s in sts if s and "exp_avg" in s]
    t = (float(t_all[0].item()) if torch.is_tensor(t_all[0]) else float(t_all[0])) if t_all else 0.0
    sn, mm, vv, uu = [], [], [], []
    for i, (m, w0) in enumerate(zip(wmods, prev)):
        sn.append((m.weight.detach() - w0).norm())
        st = sts[i]
        if st and "exp_avg" in st and t > 0:
            avg, sq = st["exp_avg"], st["exp_avg_sq"]
            mm.append(avg.norm())
            vv.append(sq.sqrt().norm())
            uu.append((lr * (avg / (1 - b1 ** t)) / ((sq / (1 - b2 ** t)).sqrt() + eps)).norm())
    s, mm, vv, uu = (torch.stack(z).cpu().tolist() if z else [] for z in (sn, mm, vv, uu))
    eaten, su, ss, dnum, dden = [], 0.0, 0.0, [], []
    for i, u in enumerate(pre["u"]):
        uvi = pre["uvec"][i]
        d = wmods[i].weight.detach().reshape(-1) - prev[i].reshape(-1)
        un = (uvi * uvi).sum()
        dnum.append((d * uvi).sum())
        dden.append(un)
        su += u * u
        ss += s[i] * s[i]
        eaten.append(1.0 - s[i] / u if u > 0 else float("nan"))
    proj = (torch.stack(dnum) / torch.stack(dden).clamp_min(1e-30)).cpu().tolist()
    pg = (torch.stack(dnum).sum() / torch.stack(dden).sum().clamp_min(1e-30)).item()
    detail = [{"u_over_delta": (u / pre["delta"][i]) if pre["delta"][i] > 0 else float("nan"),
               "eaten": eaten[i], "proj": proj[i]} for i, u in enumerate(pre["u"])]
    rec = {"ep": ep, "step": step_i, "lr": lr,
           "u_over_delta": _dist([x["u_over_delta"] for x in detail]),
           "eaten": _dist(eaten),
           "eaten_global": (1.0 - (ss ** 0.5) / (su ** 0.5)) if su > 0 else float("nan"),
           "proj": _dist(proj), "eaten_proj": _dist([1.0 - x for x in proj]), "proj_global": pg,
           "adam_m": _dist(mm), "adam_sqrtv": _dist(vv), "adam_step": _dist(uu)}
    if step_i % 500 == 0:            # 每 500 步落一次每层明细，控制文件体积
        rec["layers"] = detail
    f.write(json.dumps(rec) + "\n")
    f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/data/utkface/UTKFace")
    p.add_argument("--raw", default="", help="uint8 memmap 缓存（koniq 下默认换成 KonIQ 缓存）")
    p.add_argument("--dataset", choices=["utkface", "koniq", "cifar10"], default="utkface",
                   help="utkface=现状；koniq=KonIQ-10k（records 从 --index 的 csv 读）；"
                        "cifar10=CIFAR-10 十类分类（官方 10k test 即 held-out，只配 --arch resnet）")
    p.add_argument("--index", default=KONIQ_INDEX, help="koniq 索引 csv（仅 --dataset koniq）")
    p.add_argument("--arch", choices=["net", "resnet"], default="net",
                   help="net=DrivingNet（现状）；resnet=ResNetReg(torchvision resnet18)")
    p.add_argument("--head", choices=["gelu", "relu_qat", "bilinear_glu"], default="bilinear_glu")
    p.add_argument("--ema-mode", choices=["none", "before_snap", "after_snap"], default="after_snap")
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--ema-source", choices=["param", "forward"], default="param",
                   help="影子平均的来源：param=浮点原始参数（现状，逐字节不变）；"
                        "forward=模块当前前向权重（ste 下即伪量化后的 wq）")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=-1, help="数据划分种子（-1 = 与 --seed 相同）")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--snap-every", type=int, default=1, help="每 N 步做一次权重吸附（1=每步）")
    p.add_argument("--quant-mode", choices=["overwrite", "ste"], default="overwrite",
                   help="overwrite=每 N 步把权重复写到 int8 网格（现状）；"
                        "ste=只在前向伪量化，参数保持浮点（直通梯度，绝不覆写）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--val-windows", type=int, default=2000)
    p.add_argument("--cifar-val", type=int, default=0,
                   help=">0: 从 50k 训练集分层留出该数量做选轮，官方 test 只在最后评一次")
    p.add_argument("--log-mech", action="store_true",
                   help="每个权重投影事件把机制观测量写入 mech.jsonl（默认关闭，数值路径逐位不变）")
    args = p.parse_args()
    if args.dataset == "koniq" and not args.raw:
        args.raw = KONIQ_RAW

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    torch.manual_seed(args.seed)

    is_cls = args.dataset == "cifar10"
    if is_cls and args.arch != "resnet":
        raise SystemExit("--dataset cifar10 只支持 --arch resnet")
    test_ds = None
    if is_cls:
        if args.cifar_val > 0:
            tr_idx, va_idx = cifar10.split_train(args.cifar_val, args.seed)
            tr_ds = cifar10.CifarDataset("train", augment=True, indices=tr_idx)
            va_ds = cifar10.CifarDataset("train", augment=False, indices=va_idx)
            test_ds = cifar10.CifarDataset("test", augment=False)
            print(f"train {len(tr_ds)} | val {len(va_ds)} held out of training | "
                  f"test {len(test_ds)}", flush=True)
        else:
            tr_ds = cifar10.CifarDataset("train", augment=True)
            va_ds = cifar10.CifarDataset("test", augment=False)
        train_r, val_r = [], []            # 官方划分，没有自己的 split 可记
        counts = [va_ds.labels.count(c) for c in range(10)]
        zero_mae = 1.0 - max(counts) / len(va_ds.labels)
        print(f"zero-baseline top-1 error {zero_mae:.4f}", flush=True)
    else:
        if args.dataset == "koniq":
            recs, split_fn = read_index_csv(args.index), koniq.split_random
        else:
            recs = scan(args.data)
            split_fn = split_random
        split_seed = args.seed if args.split_seed < 0 else args.split_seed
        train_r, val_r = split_fn(recs, 0.15, split_seed)
        tr_ds = ImageScalarDataset(train_r, raw=args.raw or None)
        va_ds = ImageScalarDataset(val_r, raw=args.raw or None)
        if args.max_windows and len(tr_ds) > args.max_windows:
            import random as _r
            rng = _r.Random(args.seed)
            tr_ds.records = rng.sample(tr_ds.records, args.max_windows)
        if len(va_ds) > args.val_windows:
            import random as _r2
            va_ds.records = _r2.Random(split_seed).sample(va_ds.records, args.val_windows)

        targets = [float(r["target"]) for r in val_r]
        mu = sum(targets) / len(targets)
        zero_mae = sum(abs(t - mu) for t in targets) / len(targets)
        print(f"train {len(tr_ds)} | val {len(va_ds)} | "
              f"zero-baseline MAE {zero_mae:.4f}", flush=True)

    nw = args.workers if device.type == "cuda" else 0
    tl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=nw,
                    pin_memory=device.type == "cuda", drop_last=True,
                    persistent_workers=nw > 0, prefetch_factor=4 if nw else None)
    vl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=nw,
                    pin_memory=device.type == "cuda", persistent_workers=nw > 0)

    if args.arch == "resnet":
        model = ResNetReg(num_classes=10 if is_cls else 1,
                          pretrained=not is_cls, cifar_stem=is_cls).to(device)
    else:
        model = DrivingNet(head=args.head, grid=(14, 14), seq=1).to(device)
    print(f"arch={args.arch} head={args.head} ema-source={args.ema_source} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    if args.quant_mode == "ste":
        n_ste = register_ste_(model, bits=8)
        print(f"quant-mode: ste — {n_ste} 个 Linear/Conv2d 前向伪量化（参数保持浮点，不覆写）",
              flush=True)
    ema = SnapEMA(model, args.ema_decay, args.ema_mode, args.ema_source)
    opt = torch.optim.AdamW(model.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    if is_cls:
        lossf = torch.nn.CrossEntropyLoss()

        def score_fn(pred, y):
            return (pred.argmax(-1) != y).sum().item()
    else:
        lossf = torch.nn.MSELoss()

        def score_fn(pred, y):
            return (pred - y).abs().sum().item()
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_amp:
        print(f"amp dtype: {amp_dtype} (scaler={use_scaler})", flush=True)

    tag = run_tag(args.arch, args.quant_mode)
    prefix = "cifar" if is_cls else ("koniq" if args.dataset == "koniq" else "utk")
    # koniq-* 前缀保证这份 run 永远不会被 deploy_table.py 的 results/utk-relu_qat-* glob 命中
    run = Path(args.out) / (f"{prefix}-{args.head}-{args.ema_mode}-{tag}"
                            f"s{args.seed}-{int(time.time())}")
    print(f"run dir: {run}", flush=True)
    fp.dump(run, vars(args), train_r, val_r)
    hist_path = run / "history.csv"
    hist_f = open(hist_path, "w", newline="")
    hist_w = csv.writer(hist_f)
    unit_cols = (["val_err_live", "val_err_ema", "zero_err"] if is_cls
                 else ["val_mae_live", "val_mae_ema", "zero_mae"])
    hist_w.writerow(["epoch", "train_loss", *unit_cols[:2], "lr", "sec",
                     unit_cols[2], "snap_err", "frozen_frac"])
    hist_f.flush()

    # 机制观测：--log-mech 关闭时下面所有分支都不执行，训练数值路径逐位不变
    mech_f = open(run / "mech.jsonl", "w") if args.log_mech else None
    wmods = _wmods(model)

    best_live, best_live_state = float("inf"), None
    best_ema, best_ema_state = float("inf"), None

    def val_score() -> float:
        """分类返回 top-1 错误率，回归返回 MAE：主结局永远是这一个数，越小越好。"""
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            tot = 0.0
            for b in vl:
                xb = b["images"].to(device, non_blocking=True)
                yb = (b["label"] if is_cls else b["target"]).to(device, non_blocking=True)
                o = model(xb)
                tot += score_fn(o if is_cls else o.squeeze(-1), yb)
        return tot / max(1, len(va_ds))

    for ep in range(args.epochs):
        model.train()
        t_ep = time.time()
        s_sum, n, step_i = 0.0, 0, 0
        snap_errs: list[float] = []
        for b in tl:
            opt.zero_grad(set_to_none=True)
            x = b["images"].to(device, non_blocking=True)
            y = (b["label"] if is_cls else b["target"]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(x)
                loss = lossf(out if is_cls else out.squeeze(-1), y)
            s_sum += float(loss.item())
            n += 1
            prev = None
            if (mech_f is not None and args.quant_mode == "overwrite"
                    and args.snap_every > 0 and step_i % args.snap_every == 0):
                prev = [m.weight.detach().clone() for m in wmods]
            if use_scaler:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            pre = _mech_pre(wmods, prev) if prev is not None else None
            if (args.quant_mode == "overwrite" and args.snap_every > 0
                    and step_i % args.snap_every == 0):
                _e = ema.update_(model, snap_weights_)
                if _e is not None:
                    snap_errs.append(_e)
            else:
                ema.update_(model, None)
            if pre is not None:
                _mech_write(mech_f, wmods, prev, pre, opt,
                            float(opt.param_groups[0]["lr"]), ep, step_i)
            step_i += 1

        model.eval()
        mae = val_score()

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict_ema(), strict=False)
        mae_ema = val_score()
        model.load_state_dict(backup, strict=False)

        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        snap_err = sum(snap_errs) / len(snap_errs) if snap_errs else float("nan")
        frozen = sum(1 for e in snap_errs if e < 1e-9) / len(snap_errs) if snap_errs else float("nan")
        hist_w.writerow([ep, f"{s_sum / max(1, n):.6f}", f"{mae:.6f}", f"{mae_ema:.6f}",
                         f"{cur_lr:.3e}", f"{time.time() - t_ep:.1f}", f"{zero_mae:.6f}",
                         f"{snap_err:.3e}", f"{frozen:.4f}"])
        hist_f.flush()
        if mae < best_live:
            best_live, best_live_state = mae, {k: v.detach().clone()
                                               for k, v in model.state_dict().items()}
        if mae_ema < best_ema:
            best_ema, best_ema_state = mae_ema, {k: v.clone()
                                                 for k, v in ema.state_dict_ema().items()}
        print(f"epoch {ep:03d} live {mae:.5f} (best {best_live:.5f}) | "
              f"ema {mae_ema:.5f} (best {best_ema:.5f}) | snap {snap_err:.2e} "
              f"frozen {frozen:.2f}", flush=True)

    hist_f.close()
    try:
        from plot_history import plot_history
        plot_history(hist_path, run / "curves.png",
                     title=f"{args.dataset} / {args.head} / {args.ema_mode}")
    except Exception as exc:
        print(f"plot skipped: {exc}", flush=True)

    sel_live, sel_ema = best_live, best_ema
    if test_ds is not None:       # 干净口径：选轮只看留出的 val，官方 test 最后评一次
        def eval_state(state, ds):
            model.load_state_dict(state, strict=False)
            model.eval()
            dl = DataLoader(ds, 256, shuffle=False, num_workers=0)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                tot = 0.0
                for b in dl:
                    xb = b["images"].to(device)
                    yb = b["label"].to(device)
                    tot += score_fn(model(xb), yb)
            return tot / max(1, len(ds))

        best_live = eval_state(best_live_state, test_ds)
        best_ema = eval_state(best_ema_state, test_ds)
        print(f"clean split: val {sel_live:.4f}/{sel_ema:.4f} -> test "
              f"{best_live:.4f}/{best_ema:.4f}", flush=True)

    if args.quant_mode == "ste":   # parametrize 键还原成标准名，值仍是浮点原始参数
        best_live_state, best_ema_state = plain_keys(best_live_state), plain_keys(best_ema_state)
    torch.save({"ema_state_dict": best_ema_state, "model_state_dict": best_live_state,
                "val_mae_live": best_live, "val_mae_ema": best_ema, "args": vars(args)},
               run / "best.pth")
    v_live = "BEATS" if best_live < zero_mae else "DOES NOT beat"
    v_ema = "BEATS" if best_ema < zero_mae else "DOES NOT beat"
    print(f"final: live {best_live:.4f} ({v_live}) | ema {best_ema:.4f} ({v_ema}) "
          f"| zero {zero_mae:.4f}", flush=True)
    res = {"val_mae_live": best_live, "val_mae_ema": best_ema, "zero_baseline_mae": zero_mae,
           "verdict_live": v_live, "verdict_ema": v_ema, "args": vars(args)}
    if is_cls:                   # 同两个字段装的是 top-1 错误率，另给一份明名，回归侧不动
        res.update({"val_err_live": sel_live, "val_err_ema": sel_ema,
                    "test_err_live": best_live, "test_err_ema": best_ema, "zero_err": zero_mae,
                    "cifar_val": args.cifar_val, "units": "top-1 error"})
    (run / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
