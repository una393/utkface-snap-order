"""vendor8: resolve 16 (seed, endpoint) checkpoints, export ONNX, write hb_compile yamls.

Checkpoint selection follows scripts/deploy_table.py:373-401 (freq = args.get("snap_every", 1),
split_seed_of, decay_of). Weights follow deploy_table.py:451:
    key = "ema_state_dict" if mode != "none" else "model_state_dict"
-> the none arm must be exported from the LIVE weights; its ema_state_dict is the untrained
initialisation (val_mae_ema 0.26-1.16 vs live 0.053-0.076), so exporting it would be nonsense.

Endpoints: N=1  -> --snap-every 1 (projection every step)
           N=inf -> --snap-every 0 (projection off)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch

from models.net import DrivingNet

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "npu_quant" / "onnx"
CFG_DIR = ROOT / "npu_quant" / "hb_config"
OUT_DIR = ROOT / "npu_quant" / "out"
SIZE = 224

YAML = """model_parameters:
  onnx_model: '{root}/npu_quant/onnx/{tag}.onnx'
  output_model_file_prefix: '{tag}'
  working_dir: '{root}/npu_quant/out/{tag}'
  march: 'nash-m'

input_parameters:
  input_name: 'sequence_images'
  input_type_rt: 'featuremap'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_shape: '1x1x3x224x224'

calibration_parameters:
  cal_data_dir: '{root}/npu_quant/calib'
  cal_data_type: 'float32'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
  debug: False
  core_num: 1
"""


def resolve() -> list[dict]:
    """One run per (seed, endpoint); filters mirror deploy_table.py."""
    best: dict[tuple[int, int], dict] = {}
    for d in sorted((ROOT / "results").glob("utk-relu_qat-none-s*")):
        if "-ste-" in d.name or "-resnet-" in d.name:
            continue                     # deploy_table.py:381-384 excludes both arms
        rj = d / "result.json"
        if not rj.exists():
            continue
        meta = json.loads(rj.read_text(encoding="utf-8"))
        a = meta["args"]
        if a.get("head") != "relu_qat" or a.get("ema_mode") != "none":
            continue
        seed = int(a.get("seed", 0))
        snap = a.get("snap_every", 1)
        snap = 1 if snap is None else int(snap)
        decay = a.get("ema_decay")
        if decay is None or abs(float(decay) - 0.99) > 1e-9:
            continue
        sp = a.get("split_seed", None)
        sp = seed if sp is None or int(sp) < 0 else int(sp)
        if sp != 0 or snap not in (1, 0):
            continue
        ep = "inf" if snap == 0 else "1"
        tag = f"vendor8_s{seed}_N{ep}"
        rec = {
            "tag": tag, "seed": seed, "endpoint": ep, "snap_every": snap,
            "run": d.name, "ckpt": str(d / "best.pth"),
            "ema_decay": float(decay), "split_seed": sp,
            "ema_mode": "none", "weights": "live",
            "val_mae_live": meta.get("val_mae_live"),
            "val_mae_ema": meta.get("val_mae_ema"),
        }
        prev = best.get((seed, snap))
        if prev is not None and prev["run"] != rec["run"]:
            # never silently pick one of two candidates
            raise SystemExit(f"ambiguous checkpoints for seed={seed} snap={snap}: "
                             f"{prev['run']} vs {rec['run']}")
        best[(seed, snap)] = rec
    rows = [best[k] for k in sorted(best)]
    for ep in ("1", "inf"):
        n = sum(1 for r in rows if r["endpoint"] == ep)
        if n != 8:
            raise SystemExit(f"endpoint N={ep}: expected 8 seeds, found {n}")
    if len(rows) != 16:
        raise SystemExit(f"expected 16 (seed, endpoint) checkpoints, found {len(rows)}")
    return rows


def build(ckpt: str, head: str = "relu_qat") -> DrivingNet:
    net = DrivingNet(head=head, grid=(14, 14), seq=1, pretrained=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]          # none arm -> live weights (see module docstring)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    net.eval()
    return net


def export_one(rec: dict) -> None:
    tag = rec["tag"]
    onnx_path = ONNX_DIR / f"{tag}.onnx"
    net = build(rec["ckpt"])
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    dummy = torch.zeros(1, 1, 3, SIZE, SIZE)
    with torch.no_grad():
        torch.onnx.export(net, dummy, str(onnx_path), opset_version=18,
                          input_names=["sequence_images"], output_names=["age"], dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    (CFG_DIR / f"{tag}.yaml").write_text(YAML.format(root="/work", tag=tag))
    print(f"[export] {tag} <- {Path(rec['ckpt']).parent.name}  "
          f"val_live={rec['val_mae_live']:.4f}  bytes={onnx_path.stat().st_size}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-export", action="store_true")
    a = p.parse_args()
    rows = resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"k_images": 200, "targets_from": "npu_quant/testbin/meta.json", "runs": rows}
    (OUT_DIR / "vendor8_manifest.json").write_text(json.dumps(manifest, indent=1))
    if not a.skip_export:
        for r in rows:
            export_one(r)
    for r in rows:
        print(f"{r['tag']:<16} {r['run']:<34} snap={r['snap_every']} "
              f"split={r['split_seed']} decay={r['ema_decay']}", flush=True)
    print(f"[manifest] {len(rows)} checkpoints -> npu_quant/out/vendor8_manifest.json")


if __name__ == "__main__":
    main()
