"""用打过补丁的常量折叠跑 hb_compile（见 npu_quant/hb_patch_fold.py 的说明）。

用法（容器内）：
  python3 /work/npu_quant/hb_compile_patched.py -c /work/npu_quant/hb_config/xxx.yaml
"""

from __future__ import annotations

import faulthandler

import hb_patch_fold  # noqa: F401  只为安装补丁
from horizon_tc_ui.hb_compile import main

if __name__ == "__main__":
    faulthandler.enable()
    main()
