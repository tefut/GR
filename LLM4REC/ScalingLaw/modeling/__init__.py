import os
import sysconfig
import torch

from utils.logging_utils import logging

PY_PACKAGE_PATH = sysconfig.get_path("purelib")

libso_path = os.path.join(PY_PACKAGE_PATH, "libfbgemm_npu_api.so")
try:
    torch.ops.load_library(libso_path)
    logging.info("Loadding %s to torch.ops succeed.", libso_path)
except Exception as e:
    logging.info("Loadding %s to torch.ops failed: %s", libso_path, e)

HAS_FBGEMM_OPS = False
try:
    import fbgemm_gpu

    torch.ops.load_library(os.path.join(os.path.dirname(fbgemm_gpu.__file__), "fbgemm_gpu_py.so"))
    HAS_FBGEMM_OPS = True
    logging.info("MXRec fbgemm ops installed, ready to be called.")
except Exception as e:
    logging.info("MXRec fbgemm ops not installed: %s", e)

TORCHREC_VERSION_V11 = "torchrec1.1"
TORCHREC_VERSION_V05 = "torchrec0.5"
HAS_TORCHREC, TORCHREC_VERSION = False, TORCHREC_VERSION_V11
try:
    import torchrec

    HAS_TORCHREC = HAS_FBGEMM_OPS
    if HAS_TORCHREC:
        if "0.5.0" in torchrec.__version__:
            TORCHREC_VERSION = TORCHREC_VERSION_V05
        logging.info("%s and required ops installed, set 'model_cfg.name'='LongerEp' to enable.",
                     TORCHREC_VERSION)
    else:
        logging.info("Required ops of torchrec not installed.")
except Exception as e:
    logging.info("torchrec not installed: %s", e)

HAS_ATTN_FUSION_OPS = False
try:
    HAS_ATTN_FUSION_OPS = hasattr(torch.ops, "mxrec") and hasattr(torch.ops.mxrec, "hstu_dense")
    # 兼容老版本算子
    if not HAS_ATTN_FUSION_OPS:
        fuops_so_path = os.path.join(os.path.dirname(torch.__file__), "libhstu_dense_ops.so")
        if os.path.exists(fuops_so_path):
            torch.ops.load_library(fuops_so_path)

    HAS_ATTN_FUSION_OPS = hasattr(torch.ops, "mxrec") and hasattr(torch.ops.mxrec, "hstu_dense")
    if HAS_ATTN_FUSION_OPS:
        logging.info("MXRec attention jagged fusion ops installed.")
    else:
        logging.info("MXRec attention jagged fusion ops not installed.")
except Exception as e:
    logging.info("Loading MXRec attention jagged fusion ops failed: %s", e)


JAGGED_OPS_VERSION_V1 = "V1@torch2.1"
JAGGED_OPS_VERSION_V2 = "V2@torch2.6"
JAGGED_OPS_VERSION_UNK = "Unknown"
if (hasattr(torch.ops, "mxrec")
    and hasattr(torch.ops.mxrec, "dense_to_jagged")
    and hasattr(torch.ops.mxrec, "jagged_to_padded_dense")):
    JAGGED_OPS_VERSION = JAGGED_OPS_VERSION_V2
elif HAS_FBGEMM_OPS:
    JAGGED_OPS_VERSION = JAGGED_OPS_VERSION_V1
else:
    JAGGED_OPS_VERSION = JAGGED_OPS_VERSION_UNK
logging.info("The version of dense_to_jagged and jagged_to_padded_dense ops is %s.", JAGGED_OPS_VERSION)
HAS_JAGGED_OPS = JAGGED_OPS_VERSION != JAGGED_OPS_VERSION_UNK
if HAS_JAGGED_OPS:
    logging.info("dense_to_jagged and jagged_to_padded_dense ops are installed.")
else:
    logging.info("dense_to_jagged and jagged_to_padded_dense ops are not installed.")
