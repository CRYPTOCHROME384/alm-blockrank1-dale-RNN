#!/usr/bin/env python
import argparse
import os
import sys
from typing import Optional

import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
LEGACY_ALM_DATA_DIR = "/home/jingyi.xu/code_rnn/alm_data"

if LEGACY_ALM_DATA_DIR not in sys.path:
    sys.path.append(LEGACY_ALM_DATA_DIR)

from trials_patch_lib import export_trials_from_stage1  # type: ignore


def _load_registry_unique_sessions(registry_dir: str, animal: str) -> pd.DataFrame:
    cand1 = os.path.join(str(registry_dir), f"{animal}_registry.csv")
    cand2 = os.path.join(str(registry_dir), "registry.csv")
    path = cand1 if os.path.isfile(cand1) else cand2
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find registry csv in {registry_dir} (tried {cand1} and {cand2})")

    df = pd.read_csv(path, low_memory=False)
    required = ["animal", "session_id", "plane", "unit_key", "npz_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Registry missing required columns {missing}; available columns={list(df.columns)}")

    sessions = (
        df[required]
        .drop_duplicates()
        .sort_values(["animal", "session_id", "plane", "unit_key"])
        .reset_index(drop=True)
    )
    return sessions


def _format_paths(animal: str, session_id: str, plane: int, trial_root: str):
    trial_dir = os.path.join(str(trial_root), str(animal))
    sid_plane = f"{session_id}.{int(plane)}"
    trial_pkl = os.path.join(trial_dir, f"{animal}_twNew_{sid_plane}.trial_2p.pkl")
    bpod_npy = os.path.join(trial_dir, f"{animal}_twNew_{session_id}.bpod.npy")
    licks_npy = os.path.join(trial_dir, f"{animal}_twNew_{session_id}.licks.npy")
    return sid_plane, trial_pkl, bpod_npy, licks_npy


def _export_one(
    *,
    stage1_npz: str,
    animal: str,
    session_id: str,
    plane: int,
    trial_root: str,
    out_root: str,
    force: bool,
    allow_missing_bpod_licks: bool,
) -> str:
    sid_plane, trial_pkl, bpod_npy, licks_npy = _format_paths(
        animal=str(animal),
        session_id=str(session_id),
        plane=int(plane),
        trial_root=str(trial_root),
    )
    if not os.path.isfile(trial_pkl):
        raise FileNotFoundError(f"Missing trial_2p.pkl for {sid_plane}: {trial_pkl}")

    bpod_arg: Optional[str] = bpod_npy if os.path.isfile(bpod_npy) else None
    licks_arg: Optional[str] = licks_npy if os.path.isfile(licks_npy) else None
    if (bpod_arg is None or licks_arg is None) and (not bool(allow_missing_bpod_licks)):
        raise FileNotFoundError(
            f"Missing bpod/licks for {sid_plane}: "
            f"bpod_exists={os.path.isfile(bpod_npy)} licks_exists={os.path.isfile(licks_npy)}"
        )

    out_dir = os.path.join(str(out_root), str(animal))
    out_path = os.path.join(out_dir, f"trials_{sid_plane}.npz")
    export_trials_from_stage1(
        stage1_npz=str(stage1_npz),
        trial_pkl=str(trial_pkl),
        out_path=str(out_path),
        force=bool(force),
        max_time=None,
        smooth_ms=0.0,
        bpod_npy=bpod_arg,
        licks_npy=licks_arg,
        swap_p1_p2=False,
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export registry sessions into an independent trials_*.npz cache.")
    ap.add_argument("--registry-dir", required=True)
    ap.add_argument("--animal", required=True, help="Registry bundle name, e.g. ei_observed_kd53_kd91_kd95_e800_i200")
    ap.add_argument("--trial-root", default="/allen/aind/scratch/jingyi/2p")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-sessions", type=int, default=0, help="Optional positive limit over unique unit_key sessions.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-missing-bpod-licks", action="store_true")
    args = ap.parse_args()

    sessions = _load_registry_unique_sessions(args.registry_dir, args.animal)
    if int(args.max_sessions) > 0:
        sessions = sessions.iloc[: int(args.max_sessions)].reset_index(drop=True)

    ok = 0
    fail = 0
    print(f"[INFO] registry_sessions={len(sessions)} out_root={args.out_root}")
    for row in sessions.to_dict("records"):
        sid_plane = f"{row['session_id']}.{int(row['plane'])}"
        try:
            out_path = _export_one(
                stage1_npz=str(row["npz_path"]),
                animal=str(row["animal"]),
                session_id=str(row["session_id"]),
                plane=int(row["plane"]),
                trial_root=str(args.trial_root),
                out_root=str(args.out_root),
                force=bool(args.force),
                allow_missing_bpod_licks=bool(args.allow_missing_bpod_licks),
            )
            ok += 1
            print(f"[OK] {sid_plane} -> {out_path}")
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {sid_plane}: {exc}")

    print(f"[SUMMARY] ok={ok} fail={fail} out_root={args.out_root}")
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
