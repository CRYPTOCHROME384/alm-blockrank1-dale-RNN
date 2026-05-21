# current_rnn/utils_celltype.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence


EXC_KEYWORDS = (

)

INH_KEYWORDS = (
    "Vip",
    "Sst",
    "Pvalb",
    "Lamp5",
    "Sncg",
    "Serpinf1",
    "Meis2"        
)


def subclass_to_is_excitatory(subclass: str) -> bool:

    if subclass is None:
        return True

    s = str(subclass).lower().strip()
    if s in ("", "nan", "none"):
        return True

    has_exc = any(k in s for k in EXC_KEYWORDS)
    has_inh = any(k in s for k in INH_KEYWORDS)

    if has_exc and not has_inh:
        return True
    if has_inh and not has_exc:
        return False

    # 模糊情况默认 excitatory（通常是 IT-like）
    return True


def load_is_excitatory_from_npz(npz_path: str) -> np.ndarray:

    data = np.load(npz_path, allow_pickle=True)

    if "cell_types" not in data.files:
        raise KeyError(f"{npz_path} 中没有字段 'cell_types'")

    subclasses = data["cell_types"]  # dtype=object, len=N
    subclasses = np.asarray(subclasses).astype(str)

    is_exc = np.array(
        [subclass_to_is_excitatory(s) for s in subclasses],
        dtype=bool
    )
    return is_exc