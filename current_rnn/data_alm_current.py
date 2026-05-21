# current_rnn/data_alm_current.py

import os
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch


def build_tone_waveform(
    T: int,
    fps: float,
    S_frame: int,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
) -> np.ndarray:
    
    x = np.zeros(T, dtype=np.float32)

    on_frames = int(round(on_sec * fps))
    off_frames = int(round(off_sec * fps))
    sample_frames = int(round(sample_len_sec * fps))

    start = int(S_frame)
    end = min(T, start + sample_frames)

    t = start
    for k in range(n_bursts):
        if t >= end:
            break
        # ON
        t_on_end = min(end, t + on_frames)
        x[t:t_on_end] = 1.0
        t = t_on_end
        # OFF (last burst can omit off)
        if k < n_bursts - 1:
            t_off_end = min(end, t + off_frames)
            # already zeros
            t = t_off_end

    return x

def _normalize_cond_names(arr) -> List[str]:
    """
    Ensure cond_names loaded from npz is a list of Python strings.
    """
    if isinstance(arr, np.ndarray):
        # Could be array of bytes/str/object
        return [str(x) for x in arr.tolist()]
    # Fallback
    return [str(x) for x in list(arr)]


def normalize_cell_sign(x: Any, default: str = "inhibitory") -> str:
    """Normalize registry-provided cell sign labels to excitatory/inhibitory."""
    aliases = {
        "exc": "excitatory",
        "e": "excitatory",
        "excit": "excitatory",
        "excitatory": "excitatory",
        "inh": "inhibitory",
        "i": "inhibitory",
        "inhib": "inhibitory",
        "inhibitory": "inhibitory",
    }
    s = "" if x is None else str(x).strip().lower()
    if s in {"", "nan", "none", "null", "na"}:
        s = str(default).strip().lower()
    if s not in aliases:
        raise ValueError(
            f"Unsupported cell_sign={x!r}. Expected one of {sorted(list(set(aliases.keys())))} "
            "or an empty/null value for fallback."
        )
    return aliases[s]


def cell_sign_to_is_excitatory(x: Any, default: str = "inhibitory") -> bool:
    return normalize_cell_sign(x, default=default) == "excitatory"


def build_dale_mask_from_types(is_excitatory: np.ndarray) -> torch.Tensor:
    """
    return dale_mask: [N, N]：
        excitatory → J[:, j] >= 0  (mask = +1)
        inhibitory → J[:, j] <= 0  (mask = -1)
    """
    N = is_excitatory.shape[0]
    dale_mask = np.zeros((N, N), dtype=np.int8)

    exc_idx = np.where(is_excitatory)[0]
    inh_idx = np.where(~is_excitatory)[0]

    # excitatory columns
    dale_mask[:, exc_idx] = 1
    # inhibitory columns
    dale_mask[:, inh_idx] = -1

    # do not constrain self-connection
    np.fill_diagonal(dale_mask, 0)

    return torch.from_numpy(dale_mask)

def load_alm_psth_npz(
    npz_path: str,
    cond_filter: Optional[List[str]] = None,
    max_time: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load trial-averaged PSTH and metadata from a Stage 1 .npz file
    produced by alm_data/0.average.py.

    The .npz is expected to contain (see 0.average.py):
        - session_id, plane, animal
        - cond_names: array of condition names (keys of cell_psth)
        - cell_psth: dict[name] -> ndarray (cells, time)
        - cell_clusters, cell_subclasses
        - fps, t0_frame, event_frames, pre_frames, post_frames
        - keep_idx, n_cells_before
        - cond_counts, source_used, pkl_key_used, ...

    This function:
        1) Selects conditions to use (e.g. ['left_correct', 'right_correct']).
        2) Builds a tensor psth of shape [C, T, N], where:
           - C = number of selected conditions
           - T = number of time points (optionally truncated by max_time)
           - N = number of neurons after cell-type filtering
        3) Returns (psth, meta), where psth is a torch.Tensor and meta is a dict.

    Args:
        npz_path:   path to the Stage 1 npz file.
        cond_filter:
            - If None: try to use ['left_correct', 'right_correct'] if present,
              otherwise use all available conditions found in cell_psth.
            - If list: use the intersection of this list with available keys.
              If nothing matches, raise ValueError.
        max_time:
            - If not None: truncate time dimension to [0:max_time] frames.
        device:
            - Optional torch device to move psth tensor onto.
        dtype:
            - torch dtype for the returned psth.

    Returns:
        psth: torch.Tensor of shape [C, T, N]
        meta: dict containing metadata and numpy arrays (celltype info, etc.)
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # -------------------------------------------------------------------------
    # Condition names & cell_psth
    # -------------------------------------------------------------------------
    cond_names_all = _normalize_cond_names(data["cond_names"])
    cell_psth_obj = data["cell_psth"]
    # cell_psth was saved as a dict (object array), need .item() to recover
    if isinstance(cell_psth_obj, np.ndarray):
        cell_psth: Dict[str, np.ndarray] = cell_psth_obj.item()
    else:
        cell_psth = cell_psth_obj

    # --- decide which conditions to use ---
    if cond_filter is None:
        preferred = ["left_correct", "right_correct"]
        cond_names = [c for c in preferred if c in cell_psth]
        if not cond_names:
            # fallback: use all conditions that actually exist in cell_psth
            cond_names = [c for c in cond_names_all if c in cell_psth]
    else:
        cond_names = [c for c in cond_filter if c in cell_psth]
        if not cond_names:
            raise ValueError(
                f"No requested conditions {cond_filter} found in cell_psth keys "
                f"({list(cell_psth.keys())})."
            )

    if not cond_names:
        raise ValueError(
            f"No valid conditions found in npz file {npz_path}. "
            "Check that 0.average.py produced non-empty cell_psth."
        )

    # -------------------------------------------------------------------------
    # Build PSTH array: [C, T, N]
    # Each cell_psth[name] has shape (cells, time)
    # We transpose to (T, N) and then stack -> (C, T, N)
    # -------------------------------------------------------------------------
    psth_list = []
    T_min = None
    N = None

    # First pass: ensure all conditions have consistent (cells, time)
    for name in cond_names:
        M = cell_psth[name]  # (cells, time)
        if M is None:
            raise ValueError(f"cell_psth['{name}'] is None in {npz_path}")

        if N is None:
            N = M.shape[0]
        elif M.shape[0] != N:
            raise ValueError(
                f"Inconsistent neuron count among conditions in {npz_path}: "
                f"condition '{name}' has {M.shape[0]} cells, expected {N}."
            )

        if T_min is None:
            T_min = M.shape[1]
        else:
            T_min = min(T_min, M.shape[1])

    # If max_time is specified, clip by max_time; otherwise use min length across conditions
    if max_time is not None:
        T = min(T_min, max_time)
    else:
        T = T_min

    for name in cond_names:
        M = cell_psth[name]  # (cells, time)
        M = M[:, :T]         # truncate time if needed -> (N, T)
        M = M.T              # (T, N)
        psth_list.append(M)

    # stack to get (C, T, N)
    psth_np = np.stack(psth_list, axis=0)
    psth = torch.as_tensor(psth_np, dtype=dtype)
    if device is not None:
        psth = psth.to(device)

    # -------------------------------------------------------------------------
    # Collect metadata for later analyses
    # -------------------------------------------------------------------------
    # Cell-type info
    cell_clusters = np.asarray(data["cell_clusters"])
    cell_subclasses = np.asarray(data["cell_subclasses"])
    # some npz also stores 'cell_types'; keep if present
    cell_types = np.asarray(data["cell_types"]) if "cell_types" in data else cell_clusters

    # Time / alignment info
    fps = float(data["fps"])
    t0_frame = int(data["t0_frame"])
    pre_frames = int(data["pre_frames"])
    post_frames = int(data["post_frames"])

    # event_frames saved as a dict -> unwrap
    event_frames_obj = data["event_frames"]
    if isinstance(event_frames_obj, np.ndarray):
        # usually a 0-d object array containing a dict
        event_frames = event_frames_obj.item()
    else:
        event_frames = event_frames_obj

    # Index mapping and counts
    keep_idx = np.asarray(data["keep_idx"], dtype=int)
    n_cells_before = int(data["n_cells_before"])
    cond_counts_obj = data["cond_counts"]
    if isinstance(cond_counts_obj, np.ndarray):
        cond_counts = cond_counts_obj.item()
    else:
        cond_counts = cond_counts_obj

    # Session-level info
    session_id = str(data["session_id"])
    plane = str(data["plane"])
    animal = str(data["animal"])
    align_to = str(data["align_to"])
    source_used = str(data["source_used"])
    pkl_key_used = str(data["pkl_key_used"])

    meta: Dict[str, Any] = {
        # Core dimensions
        "cond_names": cond_names,
        "all_cond_names": cond_names_all,
        "C": len(cond_names),
        "T": T,
        "N": N,

        # Time / alignment
        "fps": fps,
        "t0_frame": t0_frame,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "event_frames": event_frames,  # dict like {'S': ss, 'D': ld, 'R': go}
        "align_to": align_to,

        # Cell-type / indexing
        "cell_clusters": cell_clusters,
        "cell_subclasses": cell_subclasses,
        "cell_types": cell_types,
        "keep_idx": keep_idx,
        "n_cells_before": n_cells_before,

        # Condition counts (per condition)
        "cond_counts": cond_counts,

        # Session identifiers
        "session_id": session_id,
        "plane": plane,
        "animal": animal,

        # Provenance
        "source_used": source_used,
        "pkl_key_used": pkl_key_used,
        "npz_path": os.path.abspath(npz_path),
    }

    return psth, meta


def _unwrap_dictlike(obj: Any, key_name: str, path: str) -> Dict[str, Any]:
    """Unwrap dict-like objects saved into npz (often as 0-d object arrays)."""
    x = obj
    if isinstance(x, np.ndarray):
        # Most common: 0-d object array containing a dict
        if x.ndim == 0:
            x = x.item()
        elif x.size == 1:
            x = x.reshape(-1)[0]
            if isinstance(x, np.ndarray) and x.ndim == 0:
                x = x.item()
    if isinstance(x, dict):
        return x
    raise TypeError(f"{path}: expected '{key_name}' to be dict-like, got {type(x)}")


def _resolve_trials_npz_path(
    *,
    stage1_npz_path: str,
    unit_key: str,
    animal: Optional[str] = None,
    session_id: Optional[str] = None,
    plane: Optional[str] = None,
    trials_root: Optional[str] = None,
    mode: str = "auto_from_stage1",
) -> str:
    """Resolve trials npz path without hard-coding machine-specific roots."""
    stage1_abs = os.path.abspath(str(stage1_npz_path))
    uk = str(unit_key)

    candidates: List[str] = []

    # (A) user-provided trials_root override
    if trials_root is not None and str(trials_root).strip() != "":
        tr = os.path.abspath(str(trials_root))
        if animal is not None and str(animal).strip() != "":
            candidates.append(os.path.join(tr, str(animal), f"trials_{uk}.npz"))
        candidates.append(os.path.join(tr, f"trials_{uk}.npz"))

    # (B) auto mapping based on stage1 path
    if str(mode) == "auto_from_stage1":
        d = os.path.dirname(stage1_abs)
        bn = os.path.basename(stage1_abs)
        if bn.startswith("psth_"):
            bn2 = "trials_" + bn[len("psth_") :]
        else:
            bn2 = bn.replace("psth_", "trials_", 1)
            if bn2 == bn:
                bn2 = "trials_" + bn

        candidates.append(os.path.join(d, bn2))

        needle = os.sep + "stage1" + os.sep
        if needle in stage1_abs:
            alt = stage1_abs.replace(needle, os.sep + "trials" + os.sep)
            candidates.append(os.path.join(os.path.dirname(alt), bn2))
    else:
        raise ValueError(f"Unknown trials_path_mode={mode!r} (supported: 'auto_from_stage1')")

    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    # Not found: raise with rich context for debugging.
    msg = (
        "[TRPATH] Failed to resolve trials npz.\n"
        f"  unit_key={uk}\n"
        f"  animal={str(animal)} session_id={str(session_id)} plane={str(plane)}\n"
        f"  stage1={stage1_abs}\n"
        "  candidates:\n    - "
        + "\n    - ".join([os.path.abspath(c) for c in candidates])
    )
    raise FileNotFoundError(msg)


def load_alm_trials_npz(trials_npz_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load trials npz (0224-style) with minimal validation."""
    if not os.path.isfile(trials_npz_path):
        raise FileNotFoundError(f"trials npz file not found: {trials_npz_path}")

    with np.load(trials_npz_path, allow_pickle=True) as z:
        files = set(z.files)
        if "cell_trials" not in files:
            raise KeyError(f"{trials_npz_path}: missing key 'cell_trials' (available keys: {sorted(list(files))})")
        if "keep_idx" not in files:
            raise KeyError(f"{trials_npz_path}: missing key 'keep_idx' (available keys: {sorted(list(files))})")

        cell_trials_raw = _unwrap_dictlike(z["cell_trials"], "cell_trials", trials_npz_path)
        cell_trials: Dict[str, np.ndarray] = {str(k): np.asarray(v) for k, v in cell_trials_raw.items()}

        keep_idx = np.asarray(z["keep_idx"], dtype=int)
        if keep_idx.ndim != 1:
            keep_idx = keep_idx.reshape(-1)

        meta: Dict[str, Any] = {"keep_idx": keep_idx}
        if "cond_names" in files:
            meta["cond_names"] = _normalize_cond_names(z["cond_names"])
        if "fps" in files:
            meta["fps"] = float(np.asarray(z["fps"]).reshape(-1)[0])
        if "event_frames" in files:
            ef = z["event_frames"]
            if isinstance(ef, np.ndarray) and ef.ndim == 0:
                ef = ef.item()
            meta["event_frames"] = ef

    return cell_trials, meta


def load_trials_sub_from_stage1(
    *,
    stage1_npz_path: str,
    stage1_meta: Dict[str, Any],
    unit_key: str,
    pos_list: List[int],
    T_shared: int,
    cond_names: List[str],
    strict_trials_align: bool = True,
    trials_root: Optional[str] = None,
    trials_path_mode: str = "auto_from_stage1",
    debug_trials_align: bool = False,
) -> Dict[str, Any]:
    """Resolve/load trials and build per-unit subset aligned to Stage1 keep_idx/cond/time."""
    if stage1_meta is None:
        raise ValueError("stage1_meta is None")

    stage1_keep = np.asarray(stage1_meta.get("keep_idx", None), dtype=int)
    if stage1_keep is None or stage1_keep.ndim != 1:
        raise KeyError(f"{stage1_npz_path}: stage1_meta['keep_idx'] missing/invalid")

    animal = stage1_meta.get("animal", None)
    session_id = stage1_meta.get("session_id", None)
    plane = stage1_meta.get("plane", None)

    trials_npz_path = _resolve_trials_npz_path(
        stage1_npz_path=str(stage1_npz_path),
        unit_key=str(unit_key),
        animal=None if animal is None else str(animal),
        session_id=None if session_id is None else str(session_id),
        plane=None if plane is None else str(plane),
        trials_root=trials_root,
        mode=str(trials_path_mode),
    )

    print(
        f"[TRPATH] unit={str(unit_key)} stage1={os.path.abspath(str(stage1_npz_path))} trials={os.path.abspath(str(trials_npz_path))}",
        flush=True,
    )

    cell_trials, trials_meta = load_alm_trials_npz(trials_npz_path)
    trials_keep = np.asarray(trials_meta.get("keep_idx", None), dtype=int)
    if trials_keep is None or trials_keep.ndim != 1:
        raise KeyError(f"{trials_npz_path}: trials_meta['keep_idx'] missing/invalid")

    if bool(strict_trials_align):
        if int(stage1_keep.shape[0]) != int(trials_keep.shape[0]):
            raise ValueError(
                "[TRALIGN] keep_idx length mismatch:\n"
                f"  unit={str(unit_key)} animal={str(animal)} session_id={str(session_id)} plane={str(plane)}\n"
                f"  stage1_keep_len={int(stage1_keep.shape[0])} trials_keep_len={int(trials_keep.shape[0])}\n"
                f"  stage1_keep_head={stage1_keep[:5].tolist()} trials_keep_head={trials_keep[:5].tolist()}\n"
                f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
                f"  trials={os.path.abspath(str(trials_npz_path))}"
            )
        if not np.array_equal(stage1_keep, trials_keep):
            raise ValueError(
                "[TRALIGN] keep_idx values mismatch:\n"
                f"  unit={str(unit_key)} animal={str(animal)} session_id={str(session_id)} plane={str(plane)}\n"
                f"  stage1_keep_head={stage1_keep[:5].tolist()} trials_keep_head={trials_keep[:5].tolist()}\n"
                f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
                f"  trials={os.path.abspath(str(trials_npz_path))}"
            )

    n_keep = int(stage1_keep.shape[0])
    pos = [int(p) for p in (pos_list or [])]
    bad = [p for p in pos if (p < 0 or p >= n_keep)]
    if len(bad) > 0:
        raise ValueError(
            "[TRALIGN] pos_list out of range:\n"
            f"  unit={str(unit_key)} n_keep={n_keep} bad_pos={bad[:10]} (total_bad={len(bad)})\n"
            f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
            f"  trials={os.path.abspath(str(trials_npz_path))}"
        )

    # Ensure required conditions exist
    missing_conds = [c for c in (cond_names or []) if str(c) not in cell_trials]
    if len(missing_conds) > 0:
        raise KeyError(
            "[TRALIGN] trials missing required condition(s):\n"
            f"  unit={str(unit_key)} missing={missing_conds}\n"
            f"  available={sorted(list(cell_trials.keys()))}\n"
            f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
            f"  trials={os.path.abspath(str(trials_npz_path))}"
        )

    trials_sub: Dict[str, np.ndarray] = {}
    trials_mean_sub: Dict[str, np.ndarray] = {}

    for cname in cond_names:
        c = str(cname)
        arr = np.asarray(cell_trials[c])
        if arr.ndim != 3:
            raise ValueError(
                "[TRALIGN] cell_trials array must be 3D [C_keep,T_keep,nTr]:\n"
                f"  unit={str(unit_key)} cond={c} shape={tuple(arr.shape)}\n"
                f"  trials={os.path.abspath(str(trials_npz_path))}"
            )
        if int(arr.shape[0]) != n_keep:
            raise ValueError(
                "[TRALIGN] C_keep mismatch:\n"
                f"  unit={str(unit_key)} cond={c} trials_C_keep={int(arr.shape[0])} stage1_n_keep={n_keep}\n"
                f"  trials_shape={tuple(arr.shape)}\n"
                f"  stage1_keep_head={stage1_keep[:5].tolist()} trials_keep_head={trials_keep[:5].tolist()}\n"
                f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
                f"  trials={os.path.abspath(str(trials_npz_path))}"
            )
        if int(arr.shape[1]) < int(T_shared):
            raise ValueError(
                "[TRALIGN] T_keep too short for T_shared:\n"
                f"  unit={str(unit_key)} cond={c} trials_T_keep={int(arr.shape[1])} T_shared={int(T_shared)}\n"
                f"  trials_shape={tuple(arr.shape)}\n"
                f"  stage1={os.path.abspath(str(stage1_npz_path))}\n"
                f"  trials={os.path.abspath(str(trials_npz_path))}"
            )

        # arr: [C_keep, T_keep, nTr] -> subset to [K, T_shared, nTr]
        arr_k = arr[pos, : int(T_shared), :].astype(np.float32, copy=False)  # [K,T,R]
        # -> [R,T,K]
        sub = np.transpose(arr_k, (2, 1, 0)).astype(np.float32, copy=False)
        trials_sub[c] = sub
        trials_mean_sub[c] = sub.mean(axis=0).astype(np.float32, copy=False)

        if bool(debug_trials_align):
            print(
                f"[TRALIGN] unit={str(unit_key)} cond={c} raw_shape={list(arr.shape)} trials_sub={list(sub.shape)}",
                flush=True,
            )

    if bool(debug_trials_align):
        keep_equal = np.array_equal(stage1_keep, trials_keep)
        print(
            f"[TRALIGN] unit={str(unit_key)} keep_idx_equal={bool(keep_equal)} n_keep={n_keep}",
            flush=True,
        )

    return {
        "trials_npz_path": os.path.abspath(str(trials_npz_path)),
        "trials_sub": trials_sub,
        "trials_mean_sub": trials_mean_sub,
    }
