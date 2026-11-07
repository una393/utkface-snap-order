"""期刊双栏稿的标记性验收：交叉引用、页脚页码、题注、公式编号、越界。

用法（仓库根）：D:/tool/Python/python.exe scripts/_verify_xref.py
"""
import re
import collections
import pymupdf

PDF = "drafts/paper2_snap_order_zh.pdf"
COL_RIGHT = (286.38, 544.46)   # 左右栏右缘（实测值）
HANG_MAX = 554.5               # 行尾标点悬挂上限


def main():
    doc = pymupdf.open(PDF)
    raw = "".join(doc[i].get_text() for i in range(doc.page_count))
    # 标记必须在去空白后的文本上数：括号内半角空格是合法换行点
    norm = re.sub(r"\s+", "", raw)
    figs = {n: norm.count(f"（图{n}）") for n in range(1, 10)}
    tbls = {n: norm.count(f"（表{n}）") for n in range(1, 11)}
    print("标记 图:", figs, "合计", sum(figs.values()))
    print("标记 表:", tbls, "合计", sum(tbls.values()))

    bad_footer = [i + 1 for i in range(doc.page_count)
                  if str(i + 1) not in [l.strip() for l in doc[i].get_text().split("\n")]]
    print("页脚缺页码:", bad_footer or "无")

    cap = collections.Counter()
    for i in range(doc.page_count):
        for l in doc[i].get_text().split("\n"):
            s = l.strip()
            if re.match(r"^图 \d+\.", s):
                cap["图"] += 1
            if re.match(r"^表 \d+\.", s):
                cap["表"] += 1
    print("题注:", dict(cap))

    eq = {}
    over = []
    for i in range(doc.page_count):
        for b in doc[i].get_text("rawdict")["blocks"]:
            if b["type"] != 0:
                continue
            for l in b["lines"]:
                chars = [c for s in l["spans"] for c in s["chars"]]
                t = "".join(c["c"] for c in chars)
                if "式 (" in t:   # 表 9 的『式 (1)、(5)』单元格会被误判成公式编号
                    continue
                m = re.search(r"\((\d{1,2})\)\s*$", t)
                if m:
                    eq[int(m.group(1))] = round(max(c["bbox"][2] for c in chars if c["c"] == ")"), 2)
                x1 = max(s["bbox"][2] for s in l["spans"])
                if x1 > HANG_MAX:
                    over.append((i + 1, round(x1, 1), t.strip()[-12:]))
    print("式编号:", len(eq), "缺失:", [n for n in range(1, 17) if n not in eq],
          "右端值:", sorted(set(eq.values())))
    dev = max(abs(v - min(COL_RIGHT, key=lambda c: abs(c - v))) for v in eq.values())
    print("式编号右缘最大偏差: %.2f pt" % dev)
    print("超悬挂上限的行:", over or "无")


if __name__ == "__main__":
    main()
