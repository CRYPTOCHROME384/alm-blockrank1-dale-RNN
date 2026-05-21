# current_rnn/training_current.py

import os
import sys
import json
import time
from typing import Optional, List, Dict, Any

import numpy as np
import torch as tch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Make project root importable so we can reuse losses.py and plotting.py
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from losses import LossAverageTrials          # reuse existing loss
import plotting                               # reuse existing plotting helpers

from model_current import ALMCurrentRNN       # current_rnn/model_current.py
from data_alm_current import (
    load_alm_psth_npz,
    build_dale_mask_from_types,
    load_trials_sub_from_stage1,
    normalize_cell_sign,
    cell_sign_to_is_excitatory,
)  # current_rnn/data_alm_current.py
from utils_celltype import load_is_excitatory_from_npz  # current_rnn/utils_celltype.py

def _load_default_parameters(params_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load parameter dictionary.
    Priority when params_path is None:
      1) <ROOT_DIR>/parameters.json
      2) <ROOT_DIR>/parameters_list.json  (legacy)
    """
    cand = []
    if params_path is not None:
        cand = [params_path]
    else:
        cand = [os.path.join(ROOT_DIR, "parameters.json"), os.path.join(ROOT_DIR, "parameters_list.json")]
    for p in cand:
        if os.path.isfile(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No parameter json found. Tried: {cand}")



def _maybe_attach_lick_reward_to_meta(
    meta: Dict[str, Any],
    npz_path: str,
    T: int,
) -> Dict[str, Any]:
    """Attach lick/reward traces (if present) from the stage1 npz into meta.

    This keeps training/eval code simple even if load_alm_psth_npz does not yet
    return these fields.

    Expected shapes in the stage1 npz:
      reward_trace:      [C_all, T_all]
      lick_rate_left:    [C_all, T_all]
      lick_rate_right:   [C_all, T_all]
      lick_rate_total:   [C_all, T_all]
      t_rel_go_sec:      [T_all]
      idx_2p_in_bpod:    [n_trials_2p]  (informational)

    We slice to the *currently loaded* conditions in meta['cond_names'] and the
    currently loaded time length T.
    """
    if meta is None:
        return meta

    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return meta

    if "cond_names" not in z:
        return meta

    cond_all = [str(x) for x in z["cond_names"].tolist()]
    cond_cur = [str(x) for x in meta.get("cond_names", [])]

    name2i = {n: i for i, n in enumerate(cond_all)}
    try:
        idx = [name2i[n] for n in cond_cur]
    except KeyError:
        # If names do not match, fall back to prefix matching (rare but useful)
        idx = []
        for n in cond_cur:
            hit = None
            for i, na in enumerate(cond_all):
                if na == n or na.startswith(n) or n.startswith(na):
                    hit = i
                    break
            if hit is None:
                raise KeyError(
                    f"Cannot map cond '{n}' to stage1 npz cond_names.\n"
                    f"meta.cond_names={cond_cur}\n"
                    f"npz.cond_names={cond_all}"
                )
            idx.append(hit)

    def _slice_CT(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"Expected [C,T] array, got shape {arr.shape}")
        arr = arr[idx, :]
        if arr.shape[1] >= T:
            arr = arr[:, :T]
        else:
            # If T is longer (shouldn't happen), pad with zeros
            pad = T - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        return arr.astype(np.float32, copy=False)

    for k in ["reward_trace", "lick_rate_left", "lick_rate_right", "lick_rate_total"]:
        if k in z:
            meta[k] = _slice_CT(np.asarray(z[k]))

    if "t_rel_go_sec" in z:
        t = np.asarray(z["t_rel_go_sec"]).astype(np.float32, copy=False)
        meta["t_rel_go_sec"] = t[:T] if t.shape[0] >= T else np.pad(t, (0, T - t.shape[0]))

    if "idx_2p_in_bpod" in z:
        meta["idx_2p_in_bpod"] = np.asarray(z["idx_2p_in_bpod"])

    return meta



def _build_sample_waveforms(
    T: int,
    fps: float,
    S_frame: int,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
) -> np.ndarray:
    """Return tone_wave[T], noise_wave[T] as float32."""
    tone = np.zeros(T, dtype=np.float32)
    noise = np.zeros(T, dtype=np.float32)

    sample_frames = int(round(sample_len_sec * fps))
    start = int(S_frame)
    end = min(T, start + sample_frames)

    # noise: constant during sample
    noise[start:end] = 1.0

    # tone bursts: ON/OFF pattern
    on_f = int(round(on_sec * fps))
    off_f = int(round(off_sec * fps))

    t = start
    for k in range(n_bursts):
        if t >= end:
            break
        t_on_end = min(end, t + on_f)
        tone[t:t_on_end] = 1.0
        t = t_on_end
        if k < n_bursts - 1:
            t = min(end, t + off_f)

    return tone, noise


def _build_input_tensor(
    C: int,
    T: int,
    cond_names: List[str],
    device: tch.device,
    amp_input: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
    amp_stim: Optional[float] = None,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    amp_go_cue: Optional[float] = None,
    include_reward: bool = True,
    amp_reward: Optional[float] = None,
) -> tch.Tensor:
    """Build external input u for each condition.

    Output u has shape [C, T, D_in] where:

      u(t) = [cond_onehot] + [stim_left, stim_right] + [go_cue] + [reward]

    Notes on trainability:
      - The temporal traces (go_cue, reward_trace) are fixed functions of time
        (trial averages, aligned to the go cue).
      - The *mapping* from these traces into neuron currents is learned via W_in,
        i.e., each input channel corresponds to a trainable N-vector (one weight
        per neuron).
    """
    if amp_stim is None:
        amp_stim = amp_input
    if amp_go_cue is None:
        amp_go_cue = amp_input
    if amp_reward is None:
        amp_reward = 1.0

    # -----------------------------
    # (1) condition one-hot: [C,T,C]
    # -----------------------------
    u = tch.zeros((C, T, C), device=device, dtype=tch.float32)
    for i in range(C):
        u[i, :, i] = float(amp_input)

    # -----------------------------
    # (2) sample stimulus waveforms: [C,T,2]
    # -----------------------------
    stim = tch.zeros((C, T, 2), device=device, dtype=tch.float32)

    if meta is None:
        raise ValueError("meta is required to build stimulus/go/reward inputs (need event_frames, fps, etc.)")

    fps = float(meta.get("fps", 1.0))
    S_frame = int(meta.get("event_frames", {}).get("S", 0))

    tone, noise = _build_sample_waveforms(
    T=T,
    fps=fps,
    S_frame=S_frame,
    sample_len_sec=sample_len_sec,
    on_sec=on_sec,
    off_sec=off_sec,
    n_bursts=n_bursts,
    )

    # scale in numpy (or torch都行)，然后转 torch 到同一 device
    tone = (tone * float(amp_stim)).astype(np.float32, copy=False)
    noise = (noise * float(amp_stim)).astype(np.float32, copy=False)

    tone_t = tch.from_numpy(tone).to(device=device, dtype=tch.float32).view(T, 1)   # [T,1]
    noise_t = tch.from_numpy(noise).to(device=device, dtype=tch.float32).view(T, 1) # [T,1]
    # assign per condition
    for i, name in enumerate(cond_names):
        s = str(name).lower()

        is_lc = ("left_correct" in s) or (s.startswith("lc")) or ("_lc" in s)
        is_rc = ("right_correct" in s) or (s.startswith("rc")) or ("_rc" in s)
        if is_lc and not is_rc:
            stim[i, :, 0:1] = tone_t
            stim[i, :, 1:2] = noise_t
        elif is_rc and not is_lc:
            stim[i, :, 0:1] = noise_t
            stim[i, :, 1:2] = tone_t
        else:
            pass

    u = tch.cat([u, stim], dim=-1)  # [C,T,C+2]

    # -----------------------------
    # (3) go cue: [C,T,1] (0.1s pulse starting at R frame)
    # -----------------------------
    if include_go_cue:
        R = _get_event_frame(meta, ["R", "go", "go_cue", "go_cue_onset", "G"])
        if R is None:
            raise KeyError("Go cue frame not found in meta['event_frames'] (expected key 'R').")
        go_frames = int(round(float(go_cue_sec) * fps))
        go_frames = max(1, go_frames)
        go = tch.zeros((T, 1), device=device, dtype=tch.float32)
        a0 = max(0, int(R))
        a1 = min(T, int(R) + go_frames)
        if a1 > a0:
            go[a0:a1, 0] = float(amp_go_cue)
        go = go.unsqueeze(0).repeat(C, 1, 1)  # [C,T,1]
        u = tch.cat([u, go], dim=-1)  # +1

    # -----------------------------
    # (4) reward: [C,T,1] (per-cond trace, aligned to go cue)
    # -----------------------------
    if include_reward:
        if "reward_trace" not in meta:
            raise KeyError(
                "meta['reward_trace'] missing. Run build_lick_reward_trace.py --inplace "
                "to write reward_trace into the stage1 npz, then re-load, or call "
                "_maybe_attach_lick_reward_to_meta(meta, npz_path, T) before building u."
            )
        rew = np.asarray(meta["reward_trace"], dtype=np.float32)
        if rew.shape[0] != C:
            raise ValueError(f"reward_trace first dim must match C={C}, got {rew.shape}")
        if rew.shape[1] != T:
            # allow mild mismatch
            rew = rew[:, :T] if rew.shape[1] >= T else np.pad(rew, ((0, 0), (0, T - rew.shape[1])), mode="constant")

        reward = tch.from_numpy(rew).to(device=device, dtype=tch.float32).unsqueeze(-1)  # [C,T,1]
        reward = reward * float(amp_reward)
        u = tch.cat([u, reward], dim=-1)  # +1

    return u


def _kernel_size_from_bin_ms(fps: float, bin_ms: float) -> int:
    """Convert bin_ms to an odd kernel size in frames (>=1), so output length stays T."""
    if bin_ms is None or bin_ms <= 0:
        return 1
    k = int(round(float(bin_ms) / 1000.0 * float(fps)))
    k = max(1, k)
    if (k % 2) == 0:
        k += 1
    return k


def _time_bin_smooth_ctn(x: tch.Tensor, fps: float, bin_ms: float) -> tch.Tensor:
    """Boxcar smooth along time for x shaped [C, T, N] (or [C, T, D]). Length-preserving."""
    k = _kernel_size_from_bin_ms(fps=fps, bin_ms=bin_ms)
    if k <= 1:
        return x

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C,T,*], got {tuple(x.shape)}")

    C, T, D = x.shape
    y = x.permute(0, 2, 1).contiguous().view(C * D, 1, T)   # [C*D,1,T]
    pad = k // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=k, stride=1)            # [C*D,1,T]
    y = y.view(C, D, T).permute(0, 2, 1).contiguous()       # [C,T,D]
    return y

def _regularization_l2(
    net: ALMCurrentRNN, lam_J: float = 0.0, lam_W: float = 0.0
) -> tch.Tensor:
    """
    Simple L2 regularization on J and W_in.
    """
    reg = tch.zeros((), device=net.J.device)
    if lam_J > 0.0:
        reg = reg + lam_J * net.J.pow(2).mean()
    if lam_W > 0.0:
        reg = reg + lam_W * net.W_in.pow(2).mean()
    return reg

import numpy as np

def _get_event_frame(meta: Dict[str, Any], keys: List[str]) -> Optional[int]:
    """Try multiple keys in meta['event_frames'] and return the first found."""
    ev = meta.get("event_frames", None)
    if not isinstance(ev, dict):
        return None
    for k in keys:
        if k in ev:
            return int(ev[k])
    return None


def _build_time_mask_sample_delay_resp(
    T: int,
    fps: float,
    meta: dict,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
) -> np.ndarray:
    """
    mask = [S+ignore, D) + [D, R) + [R, R+resp_sec]
      S = sample onset
      D = delay onset
      R = go cue
    """
    # 事件帧（按你现有 meta.event_frames 的习惯，优先用 'S','D','G'）
    S = _get_event_frame(meta, ["S", "sample", "sample_on", "sample_onset"])
    D = _get_event_frame(meta, ["D", "delay", "delay_on", "delay_onset"])
    G = _get_event_frame(meta, ["R", "go", "go_cue", "response", "response_onset"])

    if S is None:
        raise KeyError("Cannot find sample onset frame in meta['event_frames'] (expected key like 'S').")
    if G is None:
        raise KeyError("Cannot find go cue frame in meta['event_frames'] (expected key like 'G').")

    ignore_frames = int(round(sample_ignore_ms * fps / 1000.0))
    start = S + ignore_frames

    if D is None:
        D = start

    m = np.zeros(T, dtype=bool)

    # sample: [start, D)
    a0 = max(0, start)
    a1 = min(T, D)
    if a1 > a0:
        m[a0:a1] = True

    # delay: [D, G)
    b0 = max(0, D)
    b1 = min(T, G)
    if b1 > b0:
        m[b0:b1] = True

    # response: [G, G + resp_sec]
    resp_frames = int(round(resp_sec * fps))
    c0 = max(0, G)
    c1 = min(T, G + resp_frames)
    if c1 > c0:
        m[c0:c1] = True

    return m


def _build_time_mask_sample_phase(
    T: int,
    meta: Dict[str, Any],
    device: tch.device,
    ignore_ms: float = 0.0,
    sample_window_ms: Optional[float] = None,
) -> tch.Tensor:
    """
    """
    fps = float(meta["fps"])
    event_frames = meta["event_frames"]  # dict like {'S': ss, 'D': ld, 'R': go}

    if "S" not in event_frames:
        raise KeyError("event_frames does not contain key 'S' (sample onset).")

    S_frame = int(event_frames["S"])

    # Number of frames to ignore after sample onset
    ignore_frames = int(round(ignore_ms / 1000.0 * fps))

    start_frame = S_frame + ignore_frames

    # Decide end_frame
    if sample_window_ms is not None:
        window_frames = int(round(sample_window_ms / 1000.0 * fps))
        end_frame = start_frame + window_frames
    else:
        # If delay onset 'D' is available, use it as the end of sample phase; otherwise use T.
        if "D" in event_frames:
            end_frame = int(event_frames["D"])
        else:
            end_frame = T

    # Clip to valid range
    start_frame = max(0, min(start_frame, T))
    end_frame = max(0, min(end_frame, T))

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time window for sample phase: "
            f"start_frame={start_frame}, end_frame={end_frame}, T={T}"
        )

    time_mask = tch.zeros(T, dtype=tch.bool, device=device)
    time_mask[start_frame:end_frame] = True

    return time_mask


def train_current_alm(
    npz_path: str,
    cond_filter=None,
    max_time=None,
    lr: float = 1e-3,
    max_epochs: int = 50000,
    seed: int = 42,
    noise_std: float = 0.0,
    psth_bin_ms: float = 200.0,
    lam_J: float = 0.0,
    lam_W: float = 0.0,
    params_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    tag: str = "",
    # time masking options:
    use_time_mask: bool = True,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
    # input channels:
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    include_reward: bool = True,
    amp_reward: Optional[float] = None,
    amp_go_cue: Optional[float] = None,
) -> None:
    """
    Train a single-session N-dimensional current-based RNN on trial-averaged ALM data.

    Args:
        npz_path:
            Path to the Stage 1 .npz file produced by 0.average.py.
        cond_filter:
            Optional list of condition names to use, e.g. ['left_correct', 'right_correct'].
            If None, load_alm_psth_npz will choose a reasonable default.
        max_time:
            Optional truncation of the time axis to max_time frames.
        lr:
            Learning rate for Adam.
        max_epochs:
            Number of training epochs.
        seed:
            Random seed for reproducibility.
        noise_std:
            Standard deviation of additive Gaussian noise on h in the RNN.
            For the first deterministic model, this can be kept at 0.0.
        lam_J:
            L2 regularization weight on J.
        lam_W:
            L2 regularization weight on W_in.
        params_path:
            Path to parameters_list.json. If None, uses <ROOT_DIR>/parameters_list.json.
        out_dir:
            Directory to save models and loss plots. If None, uses <ROOT_DIR>/results_current.
        tag:
            String tag appended to filenames for this training run.

        use_time_mask:
            If True, restrict the loss computation to a subset of time points
            specified by mask_mode and related parameters.
        mask_mode:
            Currently only "sample" is implemented: select the sample phase.
        sample_ignore_ms:
            When mask_mode == "sample": number of milliseconds after sample onset
            to exclude from the training window (e.g., 50.0 ms).
        sample_window_ms:
            When mask_mode == "sample": duration of the training window (in ms)
            after the ignored period. If None, the window extends until delay onset
            (event_frames['D']) if present, otherwise until the end of the trial.

    Returns:
        A dictionary with:
            - 'net':         trained model (ALMCurrentRNN)
            - 'psth':        ground-truth psth tensor [C, T, N]
            - 'meta':        metadata dict from load_alm_psth_npz
            - 'loss_history': numpy array of training loss values
            - 'time_mask':   torch.BoolTensor of shape [T] (or None if not used)
    """
    # -------------------------------------------------------------------------
    # Setup and configuration
    # -------------------------------------------------------------------------
    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, "results_current")
    os.makedirs(out_dir, exist_ok=True)

    default_parameters = _load_default_parameters(params_path)

    # Device selection
    device_str = default_parameters.get("device", "cpu")
    device = tch.device(device_str if tch.cuda.is_available() or device_str == "cpu" else "cpu")

    # Set random seeds
    np.random.seed(seed)
    tch.manual_seed(seed)
    if device.type == "cuda":
        tch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # Load trial-averaged PSTH and metadata
    # -------------------------------------------------------------------------
    psth, meta = load_alm_psth_npz(
        npz_path=npz_path,
        cond_filter=cond_filter,
        max_time=max_time,
        device=device,
        dtype=tch.float32,
    )
    # psth: [C, T, N]
    C, T, N = psth.shape
    cond_names = meta["cond_names"]

    # Attach lick/reward traces (if present) so _build_input_tensor can add reward channel
    meta = _maybe_attach_lick_reward_to_meta(meta=meta, npz_path=npz_path, T=T)

    # -------------------------------------------------------------------------
    # Optional: boxcar time-binning (length-preserving smoothing) on PSTH
    # -------------------------------------------------------------------------
    fps = float(meta.get("fps", 1.0))
    if psth_bin_ms is not None and psth_bin_ms > 0:
        psth = _time_bin_smooth_ctn(psth, fps=fps, bin_ms=float(psth_bin_ms))
        print(f"[INFO] Applied PSTH boxcar smoothing: bin_ms={psth_bin_ms} (fps={fps})")
    else:
        print("[INFO] PSTH boxcar smoothing disabled (psth_bin_ms<=0).")

    # -------------------------------------------------------------------------
    # Build time mask (optional)
    # -------------------------------------------------------------------------
    time_mask = None
    if use_time_mask:
        time_mask = _build_time_mask_sample_delay_resp(
        T=T,
        fps=float(meta["fps"]),
        meta=meta,
        sample_ignore_ms=sample_ignore_ms,
        resp_sec=2.0,
        )
    else:
        time_mask = np.ones(T, dtype=bool)


    # -------------------------------------------------------------------------
    # Build external input tensor u: [C, T, D_in]
    # For now, D_in = C and u is condition one-hot.
    # -------------------------------------------------------------------------
    amp_input = float(default_parameters.get("amp_input", 1.0))
    u = _build_input_tensor(
    C=C,
    T=T,
    cond_names=cond_names,
    device=device,
    amp_input=amp_input,
    meta=meta,
    amp_stim=amp_input,
    sample_len_sec=1.15,
    on_sec=0.15,
    off_sec=0.10,
    n_bursts=5,
    include_go_cue=include_go_cue,
    go_cue_sec=go_cue_sec,
    amp_go_cue=amp_go_cue,
    include_reward=include_reward,
    amp_reward=amp_reward,
    )

    D_in = u.shape[-1]

    # -------------------------------------------------------------------------
    # Instantiate the RNN
    # -------------------------------------------------------------------------
    dt = float(default_parameters.get("dt", 1.0))
    tau = float(default_parameters.get("tau", 1.0))
    substeps = int(default_parameters.get("substeps", 1))

    is_exc = load_is_excitatory_from_npz(npz_path) 

    dale_mask = build_dale_mask_from_types(is_exc)

    net = ALMCurrentRNN(
        N=N,
        D_in=D_in,
        dt=dt,
        tau=tau,
        substeps=substeps,
        nonlinearity="tanh",
        device=device,
        dale_mask=dale_mask.to(device),
    )
    # -------------------------------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------------------------------
    loss_trials = LossAverageTrials()
    optimizer = tch.optim.Adam(net.parameters(), lr=lr)

    loss_history = np.zeros(max_epochs, dtype=np.float32)

    # Use session id and tag to build filenames
    session_id = meta.get("session_id", "session")
    plane = meta.get("plane", "plane")
    animal = meta.get("animal", "animal")

    if tag:
        run_tag = f"{animal}_{session_id}_{plane}_{tag}"
    else:
        run_tag = f"{animal}_{session_id}_{plane}"

    model_path = os.path.join(out_dir, f"rnn_current_{run_tag}.pt")
    loss_plot_path = os.path.join(out_dir, f"loss_{run_tag}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"[INFO] Training ALMCurrentRNN on {npz_path}")
    print(f"[INFO] Conditions: {cond_names}")
    print(f"[INFO] psth shape: C={C}, T={T}, N={N}, device={device}")
    print(f"[INFO] dt={dt}, tau={tau}, amp_input={amp_input}, lr={lr}")
    print(f"[INFO] Saving model to: {model_path}")
    print(f"[INFO] Training for {max_epochs} epochs...")

    start_time = time.time()
    best_loss = float("inf")
    best_state_dict = None

    for epoch in range(max_epochs):
        net.train()
        optimizer.zero_grad()

        # Forward: u has shape [C, T, D_in]
        out = net(u, h0=None, noise_std=noise_std, return_rate=True)
        rates_pred = out["rate"]  # shape [C, T, N]

        # Optionally apply time mask on the time dimension
        if time_mask is not None:
            psth_used = psth[:, time_mask, :]         # [C, T_mask, N]
            rates_used = rates_pred[:, time_mask, :]  # [C, T_mask, N]
        else:
            psth_used = psth
            rates_used = rates_pred

        # Loss: trial-averaged reconstruction + L2 regularization
        loss_fit = loss_trials(psth_used, rates_used)
        loss_reg = _regularization_l2(net, lam_J=lam_J, lam_W=lam_W)
        loss = loss_fit + loss_reg

        loss.backward()
        optimizer.step()

        # Optionally enforce Dale's law by projecting J if a mask is set
        net.apply_dale_mask()

        # Record loss
        loss_val = float(loss.item())
        loss_history[epoch] = loss_val

        # Track best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        # Logging
        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Loss = {loss_val:.6f} | "
                f"Fit = {float(loss_fit.item()):.6f} | "
                f"Reg = {float(loss_reg.item()):.6f} | "
                f"Elapsed = {elapsed/60:.1f} min"
            )

        # Optionally plot loss curve periodically
        if (epoch + 1) % 1000 == 0:
            plotting.plot_loss(epoch + 1, loss_history, title="Total loss", tag=loss_plot_path)

    total_time = time.time() - start_time
    print(f"[INFO] Training finished in {total_time/60:.1f} minutes.")
    print(f"[INFO] Best loss = {best_loss:.6f}")

    # -------------------------------------------------------------------------
    # Save best model and final loss curve
    # -------------------------------------------------------------------------
    if best_state_dict is not None:
        tch.save(best_state_dict, model_path)
        print(f"[OK] Best model saved to {model_path}")

    # Final loss plot
    plotting.plot_loss(max_epochs, loss_history, title="Total loss", tag=loss_plot_path)
    print(f"[OK] Loss curve saved to {loss_plot_path}.png")

    result = {
        "net": net,
        "psth": psth,
        "meta": meta,
        "loss_history": loss_history,
        "time_mask": time_mask,
    }
    return result

# ===========================
# Global-registry training
# ===========================
import csv
from collections import defaultdict

def _time_bin_smooth_psth(x: tch.Tensor, fps: float, bin_ms: float) -> tch.Tensor:
    return _time_bin_smooth_ctn(x, fps=fps, bin_ms=bin_ms)

def _read_registry_csv(registry_csv_path):
    rows = []
    with open(registry_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) == 0:
        raise ValueError("Empty registry csv: %s" % registry_csv_path)
    return rows

def _group_registry_rows(rows):
    """
    Group rows by unit_key. Minimal required keys:
      - unit_key, npz_path, global_idx, array_idx
    """
    by_unit = defaultdict(list)
    max_g = -1
    for r in rows:
        if "unit_key" not in r or "npz_path" not in r or "global_idx" not in r or "array_idx" not in r:
            raise KeyError("Registry row missing required columns. Need unit_key/npz_path/global_idx/array_idx.")
        g = int(r["global_idx"])
        aidx = int(r["array_idx"])
        max_g = max(max_g, g)
        by_unit[str(r["unit_key"])].append((g, aidx, r))
    n_obs = max_g + 1
    return by_unit, n_obs

def _build_keepidx_pos_map(keep_idx_arr):
    # keep_idx_arr: np.ndarray[int], length N_kept
    d = {}
    for i in range(int(keep_idx_arr.shape[0])):
        d[int(keep_idx_arr[i])] = i
    return d


def _infer_observed_cell_signs_from_registry_df(df):
    """Build observed E/I sign vector from registry rows.

    Backward compatibility:
      - if cell_sign is missing, all observed rows default to inhibitory.
    """
    if "global_idx" not in df.columns:
        raise KeyError("Registry must contain global_idx column.")
    if int(df.shape[0]) == 0:
        raise ValueError("Registry dataframe is empty.")

    try:
        gvals = np.asarray(df["global_idx"].to_numpy(), dtype=np.int64)
    except Exception as e:
        raise ValueError("Registry global_idx column must be numeric.") from e
    if np.any(gvals < 0):
        raise ValueError("Registry global_idx must be >= 0.")
    n_obs = int(gvals.max()) + 1
    seen = np.zeros(n_obs, dtype=np.int8)
    is_exc = np.zeros(n_obs, dtype=np.bool_)
    explicit = "cell_sign" in df.columns

    if explicit:
        signs = [normalize_cell_sign(v, default="inhibitory") for v in df["cell_sign"].tolist()]
    else:
        signs = ["inhibitory"] * int(df.shape[0])

    for g, sign in zip(gvals.tolist(), signs):
        exc = bool(cell_sign_to_is_excitatory(sign, default="inhibitory"))
        if seen[g]:
            if bool(is_exc[g]) != exc:
                raise ValueError(f"Registry global_idx={g} has inconsistent cell_sign assignments.")
        else:
            is_exc[g] = exc
            seen[g] = 1

    missing = np.where(seen == 0)[0]
    if missing.size > 0:
        raise ValueError(
            f"Registry global_idx is not dense over [0, {n_obs - 1}]; "
            f"missing examples={missing[:10].tolist()} total_missing={int(missing.size)}"
        )

    stats = {
        "n_obs_total": int(n_obs),
        "n_obs_exc": int(np.sum(is_exc)),
        "n_obs_inh": int(n_obs - np.sum(is_exc)),
        "registry_has_explicit_cell_sign": bool(explicit),
    }
    return is_exc, stats

def _preload_units_from_registry(
    by_unit,
    n_exc_virtual,
    device,
    cond_filter,
    max_time,
    psth_bin_ms,
    sample_ignore_ms,
    # ---- optional knobs (won't break existing call sites) ----
    amp_input: float = 1.0,
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    include_reward: bool = True,
    reward_mode: str = "correctport",
    resp_sec: float = 2.0,
    # ---- trials (Phase 2 minimal) ----
    use_trials: bool = False,
    strict_trials_align: bool = True,
    trials_root: Optional[str] = None,
    trials_path_mode: str = "auto_from_stage1",
    trial_keys: Optional[List[str]] = None,
    debug_trials_align: bool = False,
    # ---- Phase 3A (minimal) ----
    phase3_precompute_var_real: bool = False,
    var_unbiased: bool = False,
    min_trials_for_var_real: int = 3,
):
    """
    Returns:
      units: list[dict] with keys:
        - unit_key, npz_path
        - psth_sub:  [C,T,K]  (target; registry-selected subset for this unit)
        - u:         [C,T,D]  (inputs; tone/go/reward)
        - idx_net:   [K] long (indices into full net of size N_total)
        - time_mask: [T] bool (torch)
        - meta: dict
      shared: dict with keys:
        - C, T, fps
        - N_obs_total, N_total
    """
    import numpy as np
    import torch as tch

    def _get_smoother():
        # Prefer psth smoother if present; otherwise fallback to continuous smoother.
        if "_time_bin_smooth_psth" in globals() and callable(globals()["_time_bin_smooth_psth"]):
            return globals()["_time_bin_smooth_psth"]
        if "_time_bin_smooth_ctn" in globals() and callable(globals()["_time_bin_smooth_ctn"]):
            return globals()["_time_bin_smooth_ctn"]
        return None

    smoother = _get_smoother()

    # -----------------------------
    # Pass 1: load each unit, build psth_sub (pre-clip), record (C,T,fps)
    # -----------------------------
    pre = []
    C_ref = None
    fps_ref = None
    g_max = -1
    T_min = None

    for unit_key, items in by_unit.items():
        if len(items) == 0:
            continue

        # all rows in this unit share npz_path
        npz_path = str(items[0][2]["npz_path"])
        for (_, _, rr) in items[1:]:
            if str(rr["npz_path"]) != npz_path:
                raise ValueError(f"unit_key {unit_key} has inconsistent npz_path in registry.")

        psth, meta = load_alm_psth_npz(
            npz_path=npz_path,
            cond_filter=cond_filter,
            max_time=None,          # we handle max_time and cross-unit alignment ourselves
            device=device,
        )

        # optional hard clip (user provided)
        if max_time is not None:
            psth = psth[:, : int(max_time), :]

        keep_idx = meta.get("keep_idx", None)
        if keep_idx is None:
            raise KeyError(
                f"meta['keep_idx'] missing for {npz_path}. "
                "Stage1 npz must contain keep_idx for registry mapping."
            )
        keep_map = _build_keepidx_pos_map(np.asarray(keep_idx, dtype=int))

        # positions in psth dim=2 for this unit
        g_list = []
        pos_list = []
        for (g, array_idx, _) in items:
            ai = int(array_idx)
            if ai not in keep_map:
                raise ValueError(
                    f"array_idx={ai} not found in keep_idx of {npz_path} (unit_key={unit_key}). "
                    "Check registry builder mapping."
                )
            pos_list.append(int(keep_map[ai]))
            g_list.append(int(g))

        if len(g_list) == 0:
            continue

        pos_t = tch.as_tensor(pos_list, dtype=tch.long, device=device)
        psth_sub = psth.index_select(dim=2, index=pos_t)  # [C,T,K]

        # bin/smooth target psth (if requested)
        if psth_bin_ms is not None and float(psth_bin_ms) > 0:
            if smoother is None:
                raise NameError("No smoothing function found. Define _time_bin_smooth_psth or _time_bin_smooth_ctn.")
            fps = float(meta["fps"])
            psth_sub = smoother(psth_sub, fps=fps, bin_ms=float(psth_bin_ms))

        C = int(psth_sub.shape[0])
        T = int(psth_sub.shape[1])
        fps = float(meta["fps"])

        if C_ref is None:
            C_ref = C
        elif C != C_ref:
            raise ValueError(f"Inconsistent C across units: unit {unit_key} has C={C}, ref C={C_ref}")

        if fps_ref is None:
            fps_ref = fps
        else:
            # allow tiny numeric jitter
            if abs(fps - fps_ref) > 1e-3:
                raise ValueError(f"Inconsistent fps across units: unit {unit_key} fps={fps}, ref fps={fps_ref}")

        T_min = T if T_min is None else min(T_min, T)

        g_max = max(g_max, max(g_list))
        pre.append(
            dict(
                unit_key=unit_key,
                npz_path=npz_path,
                psth_sub=psth_sub,
                meta=meta,
                g_list=g_list,
                pos_list=pos_list,
            )
        )

    if len(pre) == 0:
        raise ValueError("No units loaded from registry (by_unit is empty after filtering).")

    if T_min is None or C_ref is None or fps_ref is None:
        raise RuntimeError("Failed to determine shared (C,T,fps).")

    # Final shared T
    T_shared = int(T_min)

    # network sizes
    N_obs_total = int(g_max + 1)  # observed block index is global_idx in the registry
    N_total = int(n_exc_virtual) + N_obs_total

    # -----------------------------
    # Pass 2: clip to T_shared, attach lick/reward, build u/time_mask/idx_net
    # -----------------------------
    units = []
    for item in pre:
        unit_key = item["unit_key"]
        npz_path = item["npz_path"]
        meta = item["meta"]
        g_list = item["g_list"]
        pos_list = item.get("pos_list", [])

        psth_sub = item["psth_sub"][:, :T_shared, :]  # [C,T,K] clip

        # attach lick/reward traces (MUST match the final T)
        meta = _maybe_attach_lick_reward_to_meta(meta, npz_path=npz_path, T=T_shared)

        cond_names = list(meta["cond_names"])
        if len(cond_names) != int(psth_sub.shape[0]):
            raise ValueError(
                f"cond_names length mismatch for {unit_key}: "
                f"len(cond_names)={len(cond_names)} but C={int(psth_sub.shape[0])}"
            )

        trials_npz_path = None
        trials_sub = None
        trials_mean_sub = None
        trials_mean_psth = None
        trials_n_real = None
        trials_var_sub = None
        trials_var_psth = None
        if bool(use_trials):
            tr = load_trials_sub_from_stage1(
                stage1_npz_path=str(npz_path),
                stage1_meta=meta,
                unit_key=str(unit_key),
                pos_list=[int(x) for x in (pos_list or [])],
                T_shared=int(T_shared),
                cond_names=[str(x) for x in cond_names],
                strict_trials_align=bool(strict_trials_align),
                trials_root=trials_root,
                trials_path_mode=str(trials_path_mode),
                debug_trials_align=bool(debug_trials_align),
            )
            trials_npz_path = str(tr.get("trials_npz_path", ""))
            trials_sub = tr.get("trials_sub", None)
            mean_np = tr.get("trials_mean_sub", None)
            if not isinstance(mean_np, dict):
                raise TypeError(f"[TRALIGN] expected trials_mean_sub dict from loader, got {type(mean_np)}")
            trials_mean_sub = {
                str(c): tch.as_tensor(mean_np[str(c)], dtype=tch.float32, device=device) for c in cond_names
            }
            trials_mean_psth = tch.stack([trials_mean_sub[str(c)] for c in cond_names], dim=0)  # [C,T,K]

            # Apply the same smoothing as stage1 PSTH (if enabled) to keep targets comparable.
            if psth_bin_ms is not None and float(psth_bin_ms) > 0:
                if smoother is None:
                    raise NameError("No smoothing function found. Define _time_bin_smooth_psth or _time_bin_smooth_ctn.")
                fps = float(meta["fps"])
                trials_mean_psth = smoother(trials_mean_psth, fps=fps, bin_ms=float(psth_bin_ms))

            if bool(debug_trials_align):
                with tch.no_grad():
                    d = trials_mean_psth - psth_sub
                    mse = float(d.pow(2).mean().detach().cpu().item())
                    maxabs = float(d.abs().max().detach().cpu().item())
                print(
                    f"[TRDBG] unit={str(unit_key)} mean_vs_psth mse={mse:.6g} maxabs={maxabs:.6g}",
                    flush=True,
                )

            # Phase3A: cache real trial counts and (optionally) real variance moments.
            if not isinstance(trials_sub, dict):
                raise TypeError(f"[TRALIGN] expected trials_sub dict, got {type(trials_sub)}")
            trials_n_real = {str(c): int(np.asarray(trials_sub[str(c)]).shape[0]) for c in cond_names}
            if bool(phase3_precompute_var_real):
                trials_var_sub = {}
                for c in cond_names:
                    arr = np.asarray(trials_sub[str(c)])
                    rr = int(arr.shape[0])
                    if rr < int(min_trials_for_var_real):
                        continue
                    xt = tch.as_tensor(arr, dtype=tch.float32, device=device)  # [R,T,K]
                    trials_var_sub[str(c)] = xt.var(dim=0, unbiased=bool(var_unbiased))  # [T,K]
                # Stack to [C,T,K] for conditions with enough real trials; missing -> zeros.
                vv = []
                for c in cond_names:
                    if trials_var_sub is not None and str(c) in trials_var_sub:
                        vv.append(trials_var_sub[str(c)])
                    else:
                        vv.append(tch.zeros((int(T_shared), int(psth_sub.shape[2])), device=device, dtype=tch.float32))
                trials_var_psth = tch.stack(vv, dim=0)  # [C,T,K]

        # build input tensor u: [C,T,D]
        u = _build_input_tensor(
            C=len(cond_names),
            T=T_shared,
            cond_names=cond_names,
            device=device,
            amp_input=float(amp_input),
            include_go_cue=bool(include_go_cue),
            go_cue_sec=float(go_cue_sec),
            include_reward=bool(include_reward),
            meta=meta,
        )

        # build time mask: [T]
        time_mask_np = _build_time_mask_sample_delay_resp(
            T=T_shared,
            fps=float(meta["fps"]),
            meta=meta,
            sample_ignore_ms=float(sample_ignore_ms),
            resp_sec=float(resp_sec),
        )
        time_mask = tch.as_tensor(time_mask_np.astype(np.bool_), device=device)

        # map observed global index -> net index by adding n_exc_virtual offset
        idx_net = tch.as_tensor(
            [int(g) + int(n_exc_virtual) for g in g_list],
            dtype=tch.long,
            device=device,
        )

        units.append(
            dict(
                unit_key=unit_key,
                npz_path=npz_path,
                psth_sub=psth_sub,
                u=u,
                idx_net=idx_net,
                time_mask=time_mask,
                meta=meta,
                **(
                    {}
                    if not bool(use_trials)
                    else dict(
                        trials_npz_path=trials_npz_path,
                        trials_sub=trials_sub,
                        trials_mean_sub=trials_mean_sub,
                        trials_mean_psth=trials_mean_psth,
                        trials_n_real=trials_n_real,
                        trials_var_sub=trials_var_sub,
                        trials_var_psth=trials_var_psth,
                    )
                ),
            )
        )

    shared = dict(C=int(C_ref), T=int(T_shared), fps=float(fps_ref), N_obs_total=N_obs_total, N_total=N_total)
    return units, shared






def _normalize_celltype_label(x: Any) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    if s == "":
        return "unknown"
    return s


def _extract_celltype_labels_local_from_registry_rows(
    *,
    unit_rows: List[Any],
    arr_idx_local: List[int],
    registry_label_cols: List[str],
):
    """Return (labels_local, used_col) from registry rows if available.

    unit_rows: list of tuples (global_idx, array_idx, row_dict) for one unit.
    arr_idx_local: local observed neuron order used in training for this unit.
    registry_label_cols: candidate column names in registry.csv, e.g.
      ["cell_subclass", "cell_cluster"].

    Raises KeyError/ValueError if registry labels cannot be built.
    """
    if unit_rows is None or len(unit_rows) == 0:
        raise KeyError("unit_rows is empty")

    # Map array_idx -> row dict (registry rows are already aligned to unit neurons).
    row_by_array = {}
    for _g, a, rowd in unit_rows:
        row_by_array[int(a)] = rowd

    # Find a usable registry column with at least one non-unknown label among this unit's rows.
    used_col = None
    for col in registry_label_cols:
        ok = False
        for ai in arr_idx_local:
            rowd = row_by_array.get(int(ai), None)
            if rowd is None or (str(col) not in rowd):
                continue
            s = _normalize_celltype_label(rowd.get(str(col), None))
            if s.strip().lower() not in {"", "unknown", "nan", "none"}:
                ok = True
                break
        if ok:
            used_col = str(col)
            break

    if used_col is None:
        cols_present = sorted({str(k) for (_g, _a, rowd) in unit_rows for k in rowd.keys()})
        raise KeyError(
            f"No usable registry celltype column found in candidates={list(registry_label_cols)}; "
            f"available cols example includes {cols_present[:20]}"
        )

    labels_local = []
    missing_ai = []
    for ai in arr_idx_local:
        rowd = row_by_array.get(int(ai), None)
        if rowd is None:
            missing_ai.append(int(ai))
            labels_local.append("unknown")
            continue
        labels_local.append(_normalize_celltype_label(rowd.get(used_col, None)))

    if len(missing_ai) > 0:
        raise ValueError(f"Registry rows missing for array_idx(s) {missing_ai[:10]} (total={len(missing_ai)})")

    return labels_local, used_col


def _load_celltype_labels_for_keep(npz_path: str, meta: Dict[str, Any], label_key_candidates: List[str]) -> np.ndarray:
    """
    Return labels_keep aligned to meta['keep_idx'] (length = N_keep).
    Tries meta first, then npz file. Candidate keys typically include
    ['cell_types', 'cell_subclasses'].
    """
    keep_idx = np.asarray(meta.get("keep_idx", None), dtype=int)
    if keep_idx is None or keep_idx.ndim != 1:
        raise KeyError(f"{npz_path}: meta['keep_idx'] missing/invalid; cannot build celltype groups.")

    labels_full = None
    # 1) meta
    for k in label_key_candidates:
        if k in meta and meta[k] is not None:
            labels_full = np.asarray(meta[k])
            break

    # 2) npz fallback
    if labels_full is None:
        with np.load(npz_path, allow_pickle=True) as z:
            for k in label_key_candidates:
                if k in z.files:
                    labels_full = np.asarray(z[k])
                    break

    if labels_full is None:
        raise KeyError(f"{npz_path}: cannot find any label key in {label_key_candidates}")

    labels_full = np.asarray(labels_full)
    if labels_full.ndim != 1:
        labels_full = labels_full.reshape(-1)
    if int(labels_full.shape[0]) <= int(keep_idx.max()):
        raise ValueError(
            f"{npz_path}: label array length={int(labels_full.shape[0])} <= keep_idx.max={int(keep_idx.max())}; cannot index keep labels."
        )

    labels_keep = labels_full[keep_idx]
    labels_keep = np.asarray([_normalize_celltype_label(x) for x in labels_keep], dtype=object)
    return labels_keep


def _build_celltype_groups_local(
    *,
    labels_local: List[str],
    device: tch.device,
    exclude_labels: List[str],
    min_group_size: int = 2,
):
    """Build local-K grouping tensors for scheme-A celltype template loss.

    Returns a dict with:
      groups: list[LongTensor] local neuron indices in each valid celltype group
      group_names: list[str]
      counts: dict[str,int]
      n_valid_types: int
    """
    exc = set(str(x).strip().lower() for x in exclude_labels)
    groups_map = {}
    for i, lab in enumerate(labels_local):
        s = _normalize_celltype_label(lab)
        if s.strip().lower() in exc:
            continue
        groups_map.setdefault(s, []).append(int(i))

    group_names = []
    groups = []
    counts = {}
    for name, idxs in groups_map.items():
        counts[name] = len(idxs)
        if len(idxs) < int(min_group_size):
            continue
        group_names.append(name)
        groups.append(tch.as_tensor(idxs, dtype=tch.long, device=device))

    return {
        "groups": groups,
        "group_names": group_names,
        "counts": counts,
        "n_valid_types": len(groups),
        "n_total_labeled": len(labels_local),
    }


def _celltype_template_loss_A(pred_use: tch.Tensor, groups: List[tch.Tensor]) -> tch.Tensor:
    """Scheme A: within-type mean template constraint.

    pred_use: [C, Tm, K]
    For each valid celltype group g (size>=2), compute group mean template over neuron dim,
    then penalize average squared deviation of each neuron from that template.
    Final loss = average over groups.
    """
    if len(groups) == 0:
        return pred_use.new_zeros(())

    acc = pred_use.new_zeros(())
    n = 0
    for gidx in groups:
        xg = pred_use.index_select(dim=2, index=gidx)   # [C,Tm,Kg]
        if int(xg.shape[2]) < 2:
            continue
        mu = xg.mean(dim=2, keepdim=True)               # [C,Tm,1]
        acc = acc + (xg - mu).pow(2).mean()
        n += 1
    if n == 0:
        return pred_use.new_zeros(())
    return acc / float(n)


def _lambda_linear_warmup_ramp(ep1: int, lam: float, warmup_epochs: int, ramp_epochs: int) -> float:
    """ep1 is 1-based epoch index."""
    lam = float(lam)
    if lam <= 0.0:
        return 0.0
    w = int(warmup_epochs)
    r = int(ramp_epochs)
    if ep1 <= w:
        return 0.0
    if r <= 0:
        return lam
    # linearly ramp from 0 -> lam over r epochs starting after warmup
    t = float(ep1 - w) / float(r)
    t = 1.0 if t >= 1.0 else (0.0 if t <= 0.0 else t)
    return lam * t


def _get_recurrent_J(net) -> tch.Tensor:
    model = getattr(net, "module", net)
    if hasattr(model, "get_recurrent_matrix"):
        return model.get_recurrent_matrix(apply_dale=True)
    if hasattr(model, "J"):
        return getattr(model, "J")
    raise AttributeError("Could not locate recurrent matrix J on model.")


def _compute_J_regularization_stats(net):
    J = _get_recurrent_J(net)
    return J.pow(2).mean(), J.norm(), J.abs().max()


def _count_trainable_parameters(net) -> int:
    model = getattr(net, "module", net)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _count_recurrent_trainable_parameters(net) -> int:
    model = getattr(net, "module", net)
    if hasattr(model, "recurrent_trainable_parameter_count"):
        return int(model.recurrent_trainable_parameter_count())
    if hasattr(model, "J"):
        return int(model.J.numel())
    total = 0
    for name in ("J_lr_u", "J_lr_v"):
        if hasattr(model, name):
            total += int(getattr(model, name).numel())
    return int(total)


def _var_transform(v: tch.Tensor, space: str, eps: float) -> tch.Tensor:
    s = str(space)
    if s == "var":
        return v
    if s == "logvar":
        return tch.log(v + float(eps))
    raise ValueError(f"var_loss_space must be 'var' or 'logvar', got {space!r}")


def _var_weight_from_ntr(R_real: int, mode: str) -> float:
    m = str(mode)
    rr = int(R_real)
    if m == "none":
        return 1.0
    if m == "sqrt_nminus1":
        return float(max(rr - 1, 1)) ** 0.5
    if m == "nminus1":
        return float(max(rr - 1, 1))
    raise ValueError(f"var_loss_weighting must be 'none'|'sqrt_nminus1'|'nminus1', got {mode!r}")


def _simulate_trials_x0(
    *,
    net,
    u: tch.Tensor,
    idx_net: tch.Tensor,
    n_sim: int,
    noise_mode: str,
    sigma_x0: float,
    noise_std_step: float,
    fixed_seed_bank: bool = False,
    eval_seed_base: int = 12345,
    unit_key: str = "",
    debug_noise: bool = False,
) -> tch.Tensor:
    """Return Y_sim shaped [R_sim, C, T, K] (simulated trials via x0 noise)."""
    C = int(u.shape[0])
    N = int(getattr(net, "N", 0))
    if N <= 0:
        raise RuntimeError("net.N is invalid; cannot build x0 noise.")
    dev = u.device
    n_sim = int(n_sim)
    if n_sim <= 0:
        raise ValueError(f"n_sim must be >=1, got {n_sim}")

    mode = str(noise_mode)
    sig = float(sigma_x0)

    # If x0 noise disabled, run exactly one deterministic rollout (no extra RNG).
    if mode == "none" or sig <= 0.0:
        if bool(debug_noise):
            print(
                f"[NOISEDBG] unit={unit_key} mode={mode} sigma_x0={sig} n_sim=1 noise_std_step={float(noise_std_step)}",
                flush=True,
            )
        out = net(u, h0=None, noise_std=float(noise_std_step), return_rate=True)
        pred_sub = out["rate"].index_select(dim=2, index=idx_net)  # [C,T,K]
        return pred_sub.unsqueeze(0)  # [1,C,T,K]

    if mode != "x0_gaussian":
        raise ValueError(f"noise_mode must be 'none' or 'x0_gaussian', got {noise_mode!r}")

    # Fixed eval seed bank: build per-(unit,cond,sim) deterministic eps.
    gen = None
    if bool(fixed_seed_bank):
        if dev.type == "cuda":
            gen = tch.Generator(device="cuda")
        else:
            gen = tch.Generator()

    if bool(debug_noise):
        print(
            f"[NOISEDBG] unit={unit_key} mode={mode} sigma_x0={sig} n_sim={n_sim} noise_std_step={float(noise_std_step)} fixed_seed_bank={bool(fixed_seed_bank)}",
            flush=True,
        )

    # Stable unit hash for deterministic seed bank (avoid Python's randomized hash()).
    import zlib
    unit_hash = int(zlib.adler32(str(unit_key).encode("utf-8"))) & 0xFFFFFFFF

    y_list = []
    for si in range(n_sim):
        if gen is None:
            eps = tch.randn((C, N), device=dev, dtype=tch.float32)
        else:
            # Deterministic per condition.
            eps = tch.empty((C, N), device=dev, dtype=tch.float32)
            for ci in range(C):
                seed = int(eval_seed_base) + (1000003 * unit_hash) + (1009 * int(ci)) + int(si)
                gen.manual_seed(int(seed) & 0x7FFFFFFF)
                eps[ci, :] = tch.randn((N,), generator=gen, device=dev, dtype=tch.float32)

        model = getattr(net, "module", net)
        h0_dtype = next(model.parameters()).dtype
        h0 = (sig * eps).to(device=dev, dtype=h0_dtype)
        out = net(u, h0=h0, noise_std=float(noise_std_step), return_rate=True)
        pred_sub = out["rate"].index_select(dim=2, index=idx_net)  # [C,T,K]
        y_list.append(pred_sub)

    return tch.stack(y_list, dim=0)  # [R,C,T,K]


def _global_eval_units(
    *,
    net,
    units: List[Dict[str, Any]],
    loss_fn,
    lambda_celltype: float = 0.0,
    lambda_J_eff: float = 0.0,
    noise_std_eval: float = 0.0,
    use_trials: bool = False,
):
    """No-grad global evaluation over all preloaded units.

    Returns per-unit uniform averages (matching legacy single-unit sampling distribution):
      {
        'total': float, 'psth': float, 'type': float, 'J': float, 'J_frob': float, 'J_maxabs': float,
        'n_units': int, 'n_units_with_type': int
      }
    """
    net.eval()
    sum_total = 0.0
    sum_psth = 0.0
    sum_type = 0.0
    n = 0
    n_type_units = 0

    with tch.no_grad():
        loss_J_t, J_frob_t, J_maxabs_t = _compute_J_regularization_stats(net)
        for b in units:
            u = tch.nan_to_num(b["u"], nan=0.0, posinf=0.0, neginf=0.0)
            psth_target = b["psth_sub"]
            if bool(use_trials):
                if "trials_mean_psth" not in b:
                    raise KeyError("[TRALIGN] unit missing trials_mean_psth in eval (use_trials=True)")
                psth_target = b["trials_mean_psth"]
            psth_target = tch.nan_to_num(psth_target, nan=0.0, posinf=0.0, neginf=0.0)
            idx_net = b["idx_net"]
            time_mask = b["time_mask"]
            ct_info = b.get("celltype_info", None)

            out = net(u, h0=None, noise_std=float(noise_std_eval), return_rate=True)
            rates = out["rate"]
            pred_sub = rates.index_select(dim=2, index=idx_net)

            psth_use = psth_target[:, time_mask, :]
            pred_use = pred_sub[:, time_mask, :]

            lp = loss_fn(psth_use, pred_use)
            lt = pred_use.new_zeros(())
            if float(lambda_celltype) > 0.0 and isinstance(ct_info, dict):
                groups = ct_info.get("groups", [])
                if len(groups) > 0:
                    lt = _celltype_template_loss_A(pred_use, groups)
                    n_type_units += 1
            ltot = lp + float(lambda_celltype) * lt

            sum_psth += float(lp.detach().cpu().item())
            sum_type += float(lt.detach().cpu().item())
            sum_total += float(ltot.detach().cpu().item())
            n += 1

    if n == 0:
        return {
            "total": float("nan"),
            "psth": float("nan"),
            "type": float("nan"),
            "J": float("nan"),
            "J_frob": float("nan"),
            "J_maxabs": float("nan"),
            "n_units": 0,
            "n_units_with_type": 0,
        }

    loss_J_val = float(loss_J_t.detach().cpu().item())
    J_frob_val = float(J_frob_t.detach().cpu().item())
    J_maxabs_val = float(J_maxabs_t.detach().cpu().item())
    total_val = (sum_total / float(n)) + float(lambda_J_eff) * loss_J_val

    return {
        "total": total_val,
        "psth": sum_psth / float(n),
        "type": sum_type / float(n),
        "J": loss_J_val,
        "J_frob": J_frob_val,
        "J_maxabs": J_maxabs_val,
        "n_units": int(n),
        "n_units_with_type": int(n_type_units),
    }


def _global_eval_units_sampled(
    *,
    net,
    units: List[Dict[str, Any]],
    loss_fn,
    lambda_celltype: float,
    use_trials: bool,
    noise_mode: str,
    sigma_x0: float,
    n_sim_trials_eval: int,
    var_loss_space: str,
    var_loss_eps: float,
    var_unbiased: bool,
    min_trials_for_var_real: int,
    var_loss_weighting: str,
    eval_seed_base: int,
    fixed_eval_seed_bank: bool,
    debug_var_loss: bool = False,
    debug_noise: bool = False,
) -> Dict[str, Any]:
    """Sampled eval: match mean + variance moments (no trial pairing)."""
    net.eval()
    sum_mean = 0.0
    sum_var = 0.0
    sum_type = 0.0
    n = 0
    n_type_units = 0

    with tch.no_grad():
        for b in units:
            if not bool(use_trials):
                continue
            if "trials_mean_psth" not in b or "trials_sub" not in b:
                raise KeyError("[TRALIGN] sampled eval requires trials_mean_psth and trials_sub in batch")

            u = tch.nan_to_num(b["u"], nan=0.0, posinf=0.0, neginf=0.0)
            idx_net = b["idx_net"]
            time_mask = b["time_mask"]
            unit_key = str(b.get("unit_key", ""))
            ct_info = b.get("celltype_info", None)

            # Simulate trials with x0 noise (process noise off).
            y = _simulate_trials_x0(
                net=net,
                u=u,
                idx_net=idx_net,
                n_sim=int(n_sim_trials_eval),
                noise_mode=str(noise_mode),
                sigma_x0=float(sigma_x0),
                noise_std_step=0.0,
                fixed_seed_bank=bool(fixed_eval_seed_bank),
                eval_seed_base=int(eval_seed_base),
                unit_key=unit_key,
                debug_noise=bool(debug_noise),
            )  # [R,C,T,K]

            mu_sim = y.mean(dim=0)  # [C,T,K]
            # var over sim trials
            if int(y.shape[0]) >= 2:
                var_sim = y.var(dim=0, unbiased=bool(var_unbiased))  # [C,T,K]
            else:
                var_sim = tch.zeros_like(mu_sim)

            mu_real = tch.nan_to_num(b["trials_mean_psth"], nan=0.0, posinf=0.0, neginf=0.0)
            mean_loss = loss_fn(mu_real[:, time_mask, :], mu_sim[:, time_mask, :])

            type_loss = mu_sim.new_zeros(())
            if float(lambda_celltype) > 0.0 and isinstance(ct_info, dict):
                groups = ct_info.get("groups", [])
                if len(groups) > 0:
                    type_loss = _celltype_template_loss_A(mu_sim[:, time_mask, :], groups)
                    n_type_units += 1

            # Real variance
            trials_sub = b["trials_sub"]
            cond_names = list(b["meta"]["cond_names"])
            n_real = b.get("trials_n_real", None)
            if not isinstance(n_real, dict):
                n_real = {str(c): int(np.asarray(trials_sub[str(c)]).shape[0]) for c in cond_names}

            w_sum = 0.0
            acc = 0.0
            for ci, c in enumerate(cond_names):
                rr = int(n_real.get(str(c), 0))
                if rr < int(min_trials_for_var_real):
                    continue
                if b.get("trials_var_psth", None) is not None:
                    var_real = tch.as_tensor(b["trials_var_psth"][ci, :, :], device=mu_sim.device, dtype=tch.float32)  # [T,K]
                else:
                    xt = tch.as_tensor(np.asarray(trials_sub[str(c)]), dtype=tch.float32, device=mu_sim.device)  # [R,T,K]
                    var_real = xt.var(dim=0, unbiased=bool(var_unbiased))  # [T,K]
                v_pred = _var_transform(var_sim[ci, :, :], space=str(var_loss_space), eps=float(var_loss_eps))
                v_tgt = _var_transform(var_real, space=str(var_loss_space), eps=float(var_loss_eps))
                lc = (v_pred[time_mask, :] - v_tgt[time_mask, :]).pow(2).mean()
                w = _var_weight_from_ntr(rr, mode=str(var_loss_weighting))
                acc += float(w) * float(lc.detach().cpu().item())
                w_sum += float(w)

                if bool(debug_var_loss) and n == 0 and ci == 0:
                    with tch.no_grad():
                        print(
                            f"[VARDBG] (eval-smp) unit={unit_key} cond={str(c)} R_real={rr} R_sim={int(y.shape[0])} loss_c={float(lc.detach().cpu().item()):.6g} w={w:.3g} space={str(var_loss_space)}",
                            flush=True,
                        )

            var_loss = mu_sim.new_zeros(())
            if w_sum > 0.0:
                var_loss = mu_sim.new_tensor(acc / w_sum)

            sum_mean += float(mean_loss.detach().cpu().item())
            sum_var += float(var_loss.detach().cpu().item())
            sum_type += float(type_loss.detach().cpu().item())
            n += 1

    if n == 0:
        return {"mean": float("nan"), "var": float("nan"), "type": float("nan"), "n_units": 0, "n_units_with_type": 0}

    return {
        "mean": sum_mean / float(n),
        "var": sum_var / float(n),
        "type": sum_type / float(n),
        "n_units": int(n),
        "n_units_with_type": int(n_type_units),
    }

def _atomic_torch_save(obj, path: str):
    tmp = path + ".tmp"
    tch.save(obj, tmp)
    os.replace(tmp, path)


def _save_training_history_artifacts(history_rows, out_dir: str, title: Optional[str] = None):
    import pandas as pd

    csv_path = os.path.join(out_dir, "train_history.csv")
    png_path = os.path.join(out_dir, "loss_curves.png")
    columns = [
        "epoch",
        "unit_key",
        "train_total",
        "train_psth",
        "train_psth_weighted",
        "train_psth_sample",
        "train_psth_delay",
        "train_psth_response",
        "train_type",
        "train_J",
        "lambda_J_eff",
        "J_frob",
        "J_maxabs",
        "eval_total",
        "eval_psth",
        "eval_psth_sample",
        "eval_psth_delay",
        "eval_psth_response",
        "eval_type",
        "eval_J",
    ]

    df = pd.DataFrame(history_rows)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df.loc[:, columns]
    df.to_csv(csv_path, index=False)
    plotting.plot_training_history_summary(df, png_path, title=title)
    return csv_path, png_path


def train_current_alm_global(
    registry_dir: str,
    animal: str,
    out_dir: str,
    *,
    # training
    max_epochs: int = 3000,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    seed: int = 0,
    noise_std: float = 0.0,
    grad_clip: float = 1.0,
    print_every: int = 50,
    # model dynamics
    dt: float = 0.03436,
    tau: float = 0.01,
    substeps: int = 4,
    nonlinearity: str = "tanh",
    dale: bool = False,
    # global sizing
    n_exc_virtual: int = 800,
    recurrent_mode: str = "full",
    recurrent_rank: int = 0,
    random_bg_scale: float = 0.0,
    # unit/session sampling
    unit_sampling: str = "random",   # "random" | "cycle"
    max_sessions: int = None,         # optional: only use top-K sessions from registry builder
    # data shaping
    cond_filter=None,
    max_time=None,
    psth_bin_ms: float = 0.0,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
    # explicit recurrent regularization
    lambda_J: float = 0.0,
    lambda_J_warmup_epochs: int = 0,
    lambda_J_ramp_epochs: int = 0,
    # celltype loss (scheme A)
    lambda_celltype: float = 0.0,
    celltype_label_keys: Optional[List[str]] = None,
    celltype_registry_cols: Optional[List[str]] = None,
    celltype_exclude: Optional[List[str]] = None,
    celltype_min_count: int = 2,
    # checkpointing / reproducibility
    save_best_every: int = 100,
    save_latest_every: int = 0,
    eval_every: int = 100,
    best_metric: str = "eval_psth",
    run_config: Optional[Dict[str, Any]] = None,
    # ---- trials (Phase 2 minimal) ----
    use_trials: bool = False,
    strict_trials_align: bool = True,
    trials_root: Optional[str] = None,
    trials_path_mode: str = "auto_from_stage1",
    trial_keys: Optional[List[str]] = None,
    debug_trials_align: bool = False,
    # ---- Phase 3A (minimal): x0 noise + mean/var moment loss ----
    phase3_enable_var_loss: bool = False,
    noise_mode: str = "none",  # "none" | "x0_gaussian"
    celltype_loss_on: str = "mean_only",
    sigma_x0: float = 0.0,
    n_sim_trials_train: int = 16,
    n_sim_trials_eval: int = 32,
    lambda_var: float = 0.0,
    lambda_var_warmup_epochs: int = 100,
    lambda_var_ramp_epochs: int = 200,
    var_loss_space: str = "logvar",
    var_loss_eps: float = 1e-6,
    var_unbiased: bool = False,
    min_trials_for_var_real: int = 3,
    var_loss_weighting: str = "sqrt_nminus1",
    eval_seed_base: int = 12345,
    fixed_eval_seed_bank: bool = True,
    debug_var_loss: bool = False,
    debug_noise: bool = False,
):
    import os, time, json
    import numpy as np
    import pandas as pd
    import torch as tch

    os.makedirs(out_dir, exist_ok=True)

    dev = tch.device("cuda" if tch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(int(seed))
    np.random.seed(int(seed))
    tch.manual_seed(int(seed))
    if dev.type == "cuda":
        tch.cuda.manual_seed_all(int(seed))

    if celltype_label_keys is None:
        celltype_label_keys = ["cell_types", "cell_subclasses"]
    if celltype_registry_cols is None:
        celltype_registry_cols = ["cell_subclass", "cell_cluster", "cell_type", "celltype"]
    if celltype_exclude is None:
        celltype_exclude = ["", "nan", "none", "unknown"]

    # ----------------------------
    # 1) Load registry
    # ----------------------------
    registry_csv = os.path.join(registry_dir, f"{animal}_registry.csv")
    if not os.path.isfile(registry_csv):
        registry_csv = os.path.join(registry_dir, "registry.csv")
    if not os.path.isfile(registry_csv):
        raise FileNotFoundError(f"Cannot find registry csv in {registry_dir} (tried {animal}_registry.csv and registry.csv)")

    df = pd.read_csv(registry_csv)
    required_cols = ["unit_key", "global_idx", "array_idx", "npz_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Registry missing columns: {missing}. Got columns={list(df.columns)}")

    observed_is_exc, observed_sign_stats = _infer_observed_cell_signs_from_registry_df(df)

    by_unit = {}
    for _, r in df.iterrows():
        uk = str(r["unit_key"])
        g = int(r["global_idx"])
        a = int(r["array_idx"])
        by_unit.setdefault(uk, []).append((g, a, dict(r)))

    unit_keys = sorted(by_unit.keys())
    if max_sessions is not None and int(max_sessions) > 0:
        unit_keys = unit_keys[: int(max_sessions)]
        by_unit = {k: by_unit[k] for k in unit_keys}

    # cache local array_idx order per unit (must match preload order)
    array_idx_by_unit = {uk: [int(a) for (_, a, _) in by_unit[uk]] for uk in by_unit.keys()}

    # ----------------------------
    # 2) Preload unit tensors
    # ----------------------------
    if str(celltype_loss_on) != "mean_only":
        raise ValueError(f"celltype_loss_on must be 'mean_only' for Phase3A, got {celltype_loss_on!r}")

    phase3_var_enabled = bool(use_trials) and bool(phase3_enable_var_loss) and (float(lambda_var) > 0.0)
    units, shared = _preload_units_from_registry(
        by_unit=by_unit,
        n_exc_virtual=int(n_exc_virtual),
        device=dev,
        cond_filter=cond_filter,
        max_time=max_time,
        psth_bin_ms=psth_bin_ms,
        sample_ignore_ms=sample_ignore_ms,
        resp_sec=resp_sec,
        use_trials=bool(use_trials),
        strict_trials_align=bool(strict_trials_align),
        trials_root=trials_root,
        trials_path_mode=str(trials_path_mode),
        trial_keys=trial_keys,
        debug_trials_align=bool(debug_trials_align),
        phase3_precompute_var_real=bool(phase3_var_enabled),
        var_unbiased=bool(var_unbiased),
        min_trials_for_var_real=int(min_trials_for_var_real),
    )
    if len(units) == 0:
        raise RuntimeError("No units loaded from registry. Check registry csv and filters.")

    # Determine observed inhibitory count from registry global_idx
    all_g = []
    for u0 in units:
        all_g.append((u0["idx_net"].detach().cpu().numpy() - int(n_exc_virtual)).astype(int))
    all_g = np.concatenate(all_g, axis=0)
    n_obs = int(all_g.max()) + 1
    n_total = int(n_exc_virtual) + int(n_obs)
    D_in = int(units[0]["u"].shape[-1])

    # ----------------------------
    # 2.5) Build per-unit celltype groups (scheme A)
    #     Priority: registry.csv columns -> meta/npz fallback
    # ----------------------------
    n_units_with_ct = 0
    n_units_ct_registry = 0
    n_units_ct_fallback = 0
    n_units_ct_error = 0
    debug_ct_examples = []
    for b in units:
        uk = str(b.get("unit_key", ""))
        arr_idx_local = array_idx_by_unit.get(uk, [])
        try:
            labels_local = None
            ct_source = None

            # (A) registry.csv priority (recommended for current kd95 registry)
            try:
                labels_local_reg, used_col = _extract_celltype_labels_local_from_registry_rows(
                    unit_rows=by_unit.get(uk, []),
                    arr_idx_local=[int(x) for x in arr_idx_local],
                    registry_label_cols=[str(x) for x in celltype_registry_cols],
                )
                labels_local = [str(x) for x in labels_local_reg]
                ct_source = f"registry:{used_col}"
            except Exception as e_reg:
                reg_err = str(e_reg)

            # (B) fallback to meta/npz labels aligned by keep_idx
            if labels_local is None:
                labels_keep = _load_celltype_labels_for_keep(
                    npz_path=str(b["npz_path"]),
                    meta=b["meta"],
                    label_key_candidates=[str(x) for x in celltype_label_keys],
                )
                keep_idx = np.asarray(b["meta"]["keep_idx"], dtype=int)
                keep_map = {int(keep_idx[i]): int(i) for i in range(int(keep_idx.shape[0]))}
                labels_local_fb = []
                for ai in arr_idx_local:
                    if int(ai) not in keep_map:
                        raise ValueError(f"array_idx {ai} not in keep_idx for unit {uk}")
                    p = int(keep_map[int(ai)])
                    labels_local_fb.append(str(labels_keep[p]))
                labels_local = labels_local_fb
                ct_source = "meta/npz"

            ct_info = _build_celltype_groups_local(
                labels_local=labels_local,
                device=dev,
                exclude_labels=[str(x) for x in celltype_exclude],
                min_group_size=int(celltype_min_count),
            )
            ct_info["labels_local"] = labels_local
            ct_info["array_idx_local"] = [int(x) for x in arr_idx_local]
            ct_info["source"] = ct_source
            b["celltype_info"] = ct_info

            if ct_source.startswith("registry:"):
                n_units_ct_registry += 1
            else:
                n_units_ct_fallback += 1

            if ct_info.get("n_valid_types", 0) > 0:
                n_units_with_ct += 1

            if len(debug_ct_examples) < 5:
                debug_ct_examples.append({
                    "unit_key": uk,
                    "source": ct_source,
                    "n_valid_types": int(ct_info.get("n_valid_types", 0)),
                    "counts": dict(ct_info.get("counts", {})),
                })

        except Exception as e:
            n_units_ct_error += 1
            b["celltype_info"] = {
                "groups": [],
                "group_names": [],
                "counts": {},
                "n_valid_types": 0,
                "source": "error",
                "error": str(e),
            }
            if len(debug_ct_examples) < 5:
                debug_ct_examples.append({
                    "unit_key": uk,
                    "source": "error",
                    "n_valid_types": 0,
                    "counts": {},
                    "error": str(e),
                })

    print(
        f"[INFO] preloaded units={len(units)} shared(C={shared['C']},T={shared['T']},fps={shared['fps']:.3f}) "
        f"n_obs={n_obs} (exc={int(np.sum(observed_is_exc))}, inh={int(n_obs - np.sum(observed_is_exc))}) "
        f"n_total={n_total} units_with_celltype_groups={n_units_with_ct} "
        f"(ct_source registry={n_units_ct_registry}, fallback={n_units_ct_fallback}, error={n_units_ct_error})",
        flush=True,
    )
    for ex in debug_ct_examples:
        msg = (
            f"[CTDBG] unit={ex.get('unit_key','')} source={ex.get('source','')} "
            f"n_valid_types={ex.get('n_valid_types',0)} counts={ex.get('counts',{})}"
        )
        if 'error' in ex:
            msg += f" err={ex.get('error')}"
        print(msg, flush=True)

    # ----------------------------
    # 3) Dale mask
    # ----------------------------
    dale_mask = None
    if bool(dale):
        full_is_exc = np.concatenate(
            [
                np.ones(int(n_exc_virtual), dtype=np.bool_),
                observed_is_exc.astype(np.bool_, copy=False),
            ],
            axis=0,
        )
        dale_mask = build_dale_mask_from_types(full_is_exc).to(device=dev)

    # ----------------------------
    # 4) Build model
    # ----------------------------
    net = ALMCurrentRNN(
        N=int(n_total),
        D_in=int(D_in),
        dt=float(dt),
        tau=float(tau),
        substeps=int(substeps),
        nonlinearity=str(nonlinearity),
        device=dev,
        dale_mask=dale_mask,
        recurrent_mode=str(recurrent_mode),
        recurrent_rank=int(recurrent_rank),
        random_bg_scale=float(random_bg_scale),
    ).to(dev)

    opt = tch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = LossAverageTrials()
    n_trainable_params = _count_trainable_parameters(net)
    n_recurrent_trainable_params = _count_recurrent_trainable_parameters(net)

    # filenames/tag (available before training so checkpoints can be written periodically)
    tag = f"{animal}_global_nobs{n_obs}_nexc{int(n_exc_virtual)}_ntotal{n_total}"
    model_path = os.path.join(out_dir, f"rnn_current_{tag}.pt")          # legacy alias (final best)
    best_path = os.path.join(out_dir, f"rnn_current_{tag}.best.pt")      # periodically updated best
    latest_path = os.path.join(out_dir, f"rnn_current_{tag}.latest.pt")  # optional latest
    meta_path = os.path.join(out_dir, f"rnn_current_{tag}.meta.json")
    params_path = os.path.join(out_dir, f"rnn_current_{tag}.train_config.json")

    run_cfg_to_save = {
        "registry_dir": str(registry_dir),
        "animal": str(animal),
        "out_dir": str(out_dir),
        "max_epochs": int(max_epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "noise_std": float(noise_std),
        "grad_clip": None if grad_clip is None else float(grad_clip),
        "print_every": int(print_every),
        "dt": float(dt),
        "tau": float(tau),
        "substeps": int(substeps),
        "nonlinearity": str(nonlinearity),
        "dale": bool(dale),
        "n_exc_virtual": int(n_exc_virtual),
        "recurrent_mode": str(recurrent_mode),
        "recurrent_rank": int(recurrent_rank),
        "random_bg_scale": float(random_bg_scale),
        "unit_sampling": str(unit_sampling),
        "max_sessions": None if max_sessions is None else int(max_sessions),
        "cond_filter": None if cond_filter is None else [str(x) for x in cond_filter],
        "max_time": None if max_time is None else int(max_time),
        "psth_bin_ms": float(psth_bin_ms),
        "sample_ignore_ms": float(sample_ignore_ms),
        "resp_sec": float(resp_sec),
        "lambda_J": float(lambda_J),
        "lambda_J_warmup_epochs": int(lambda_J_warmup_epochs),
        "lambda_J_ramp_epochs": int(lambda_J_ramp_epochs),
        "lambda_celltype": float(lambda_celltype),
        "celltype_label_keys": [str(x) for x in celltype_label_keys],
        "celltype_registry_cols": [str(x) for x in celltype_registry_cols],
        "celltype_exclude": [str(x) for x in celltype_exclude],
        "celltype_min_count": int(celltype_min_count),
        "save_best_every": int(save_best_every),
        "save_latest_every": int(save_latest_every),
        "eval_every": int(eval_every),
        "best_metric": str(best_metric),
        "use_trials": bool(use_trials),
        "strict_trials_align": bool(strict_trials_align),
        "trials_root": None if trials_root is None else str(trials_root),
        "trials_path_mode": str(trials_path_mode),
        "trial_keys": None if trial_keys is None else [str(x) for x in trial_keys],
        "debug_trials_align": bool(debug_trials_align),
        "phase3_enable_var_loss": bool(phase3_enable_var_loss),
        "noise_mode": str(noise_mode),
        "celltype_loss_on": str(celltype_loss_on),
        "sigma_x0": float(sigma_x0),
        "n_sim_trials_train": int(n_sim_trials_train),
        "n_sim_trials_eval": int(n_sim_trials_eval),
        "lambda_var": float(lambda_var),
        "lambda_var_warmup_epochs": int(lambda_var_warmup_epochs),
        "lambda_var_ramp_epochs": int(lambda_var_ramp_epochs),
        "var_loss_space": str(var_loss_space),
        "var_loss_eps": float(var_loss_eps),
        "var_unbiased": bool(var_unbiased),
        "min_trials_for_var_real": int(min_trials_for_var_real),
        "var_loss_weighting": str(var_loss_weighting),
        "eval_seed_base": int(eval_seed_base),
        "fixed_eval_seed_bank": bool(fixed_eval_seed_bank),
        "debug_var_loss": bool(debug_var_loss),
        "debug_noise": bool(debug_noise),
    }
    if isinstance(run_config, dict):
        # merge user config (source of truth) but keep resolved runtime keys explicit
        merged = dict(run_config)
        merged["_resolved_runtime"] = run_cfg_to_save
        run_cfg_to_save = merged
    with open(params_path, "w") as f:
        json.dump(run_cfg_to_save, f, indent=2)

    # ----------------------------
    # 5) Training loop
    # ----------------------------
    best_total = float("inf")          # legacy: best single-step train total
    best_train_ep = 0
    best_score = float("inf")          # checkpoint selection score (configurable)
    best_state = None
    best_ep = 0                        # epoch for best checkpoint selection score
    best_eval_stats = None             # latest best eval stats (if eval-based selection)
    unit_order = list(range(len(units)))
    t0 = time.time()
    history_rows = []
    history_title = f"{animal} training history"
    history_csv_path = os.path.join(out_dir, "train_history.csv")
    history_png_path = os.path.join(out_dir, "loss_curves.png")

    valid_best_metrics = {"train_step_total", "eval_total", "eval_psth"}
    if str(best_metric) not in valid_best_metrics:
        raise ValueError(f"best_metric must be one of {sorted(valid_best_metrics)}, got {best_metric!r}")

    def _flush_history_safe():
        if len(history_rows) == 0:
            return
        try:
            _save_training_history_artifacts(history_rows, out_dir=out_dir, title=history_title)
        except Exception as exc:
            print(f"[WARN] Failed to save training history artifacts: {exc}", flush=True)

    try:
        for ep in range(int(max_epochs)):
            net.train()

            if unit_sampling == "random":
                ui = int(rng.randint(0, len(units)))
            elif unit_sampling == "cycle":
                ui = unit_order[ep % len(unit_order)]
            else:
                raise ValueError(f"unit_sampling must be 'random' or 'cycle', got {unit_sampling}")

            batch = units[ui]
            u = batch["u"]
            psth_sub = batch["psth_sub"]
            idx_net = batch["idx_net"]
            time_mask = batch["time_mask"]
            ct_info = batch.get("celltype_info", None)

            u = tch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            psth_sub = tch.nan_to_num(psth_sub, nan=0.0, posinf=0.0, neginf=0.0)
            psth_target = psth_sub
            if bool(use_trials):
                if "trials_mean_psth" not in batch:
                    raise KeyError("[TRALIGN] unit missing trials_mean_psth in training (use_trials=True)")
                psth_target = batch["trials_mean_psth"]
                psth_target = tch.nan_to_num(psth_target, nan=0.0, posinf=0.0, neginf=0.0)
                if bool(debug_trials_align) and ep == 0:
                    print(
                        f"[TRDBG] psth_target_source=trials_mean unit={batch.get('unit_key','NA')}",
                        flush=True,
                    )

            ep1 = int(ep + 1)
            lam_var_eff = 0.0
            if bool(phase3_var_enabled):
                lam_var_eff = _lambda_linear_warmup_ramp(
                    ep1=ep1,
                    lam=float(lambda_var),
                    warmup_epochs=int(lambda_var_warmup_epochs),
                    ramp_epochs=int(lambda_var_ramp_epochs),
                )
            lam_J_eff = _lambda_linear_warmup_ramp(
                ep1=ep1,
                lam=float(lambda_J),
                warmup_epochs=int(lambda_J_warmup_epochs),
                ramp_epochs=int(lambda_J_ramp_epochs),
            )

            noise_std_step = float(noise_std)
            if bool(phase3_var_enabled):
                noise_std_step = 0.0

            opt.zero_grad(set_to_none=True)

            loss_var = u.new_zeros(())
            var_val = 0.0
            rates = None

            if (not bool(phase3_var_enabled)) or (float(lam_var_eff) <= 0.0):
                out = net(u, h0=None, noise_std=float(noise_std_step), return_rate=True)
                rates = out["rate"]
                pred_sub = rates.index_select(dim=2, index=idx_net)

                psth_use = psth_target[:, time_mask, :]
                pred_use = pred_sub[:, time_mask, :]

                loss_psth = loss_fn(psth_use, pred_use)
                loss_type = pred_use.new_zeros(())
                if float(lambda_celltype) > 0.0 and ct_info is not None:
                    groups = ct_info.get("groups", [])
                    if len(groups) > 0:
                        loss_type = _celltype_template_loss_A(pred_use, groups)

                loss_total_base = loss_psth + float(lambda_celltype) * loss_type
            else:
                if not bool(use_trials):
                    raise ValueError("Phase3A requires use_trials=true (real trials mean/var targets).")

                y = _simulate_trials_x0(
                    net=net,
                    u=u,
                    idx_net=idx_net,
                    n_sim=int(n_sim_trials_train),
                    noise_mode=str(noise_mode),
                    sigma_x0=float(sigma_x0),
                    noise_std_step=float(noise_std_step),
                    fixed_seed_bank=False,
                    eval_seed_base=int(eval_seed_base),
                    unit_key=str(batch.get("unit_key", "")),
                    debug_noise=bool(debug_noise) and (ep == 0),
                )  # [R,C,T,K]

                mu_sim = y.mean(dim=0)  # [C,T,K]
                if int(y.shape[0]) >= 2:
                    var_sim = y.var(dim=0, unbiased=bool(var_unbiased))  # [C,T,K]
                else:
                    var_sim = tch.zeros_like(mu_sim)

                mu_use = mu_sim[:, time_mask, :]
                mu_tgt = psth_target[:, time_mask, :]
                loss_psth = loss_fn(mu_tgt, mu_use)

                loss_type = mu_use.new_zeros(())
                if float(lambda_celltype) > 0.0 and ct_info is not None:
                    groups = ct_info.get("groups", [])
                    if len(groups) > 0:
                        loss_type = _celltype_template_loss_A(mu_use, groups)

                if "trials_var_psth" in batch and batch.get("trials_var_psth", None) is not None:
                    var_real = tch.nan_to_num(batch["trials_var_psth"], nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    var_real_list = []
                    for c in list(batch["meta"]["cond_names"]):
                        xt = tch.as_tensor(np.asarray(batch["trials_sub"][str(c)]), dtype=tch.float32, device=mu_sim.device)
                        var_real_list.append(xt.var(dim=0, unbiased=bool(var_unbiased)))
                    var_real = tch.stack(var_real_list, dim=0)

                n_real = batch.get("trials_n_real", None)
                if not isinstance(n_real, dict):
                    n_real = {str(c): int(np.asarray(batch["trials_sub"][str(c)]).shape[0]) for c in list(batch["meta"]["cond_names"])}

                w_sum = 0.0
                acc = mu_sim.new_zeros(())
                cond_names = list(batch["meta"]["cond_names"])
                for ci, c in enumerate(cond_names):
                    rr = int(n_real.get(str(c), 0))
                    if rr < int(min_trials_for_var_real):
                        continue
                    w = _var_weight_from_ntr(rr, mode=str(var_loss_weighting))
                    v_pred = _var_transform(var_sim[ci, :, :], space=str(var_loss_space), eps=float(var_loss_eps))
                    v_tgt = _var_transform(var_real[ci, :, :], space=str(var_loss_space), eps=float(var_loss_eps))
                    lc = (v_pred[time_mask, :] - v_tgt[time_mask, :]).pow(2).mean()
                    acc = acc + float(w) * lc
                    w_sum += float(w)
                    if bool(debug_var_loss) and ep == 0 and ci == 0:
                        with tch.no_grad():
                            print(
                                f"[VARDBG] unit={str(batch.get('unit_key',''))} cond={str(c)} R_real={rr} R_sim={int(y.shape[0])} "
                                f"var_real_mean={float(var_real[ci].mean().detach().cpu().item()):.6g} var_sim_mean={float(var_sim[ci].mean().detach().cpu().item()):.6g} "
                                f"loss_c={float(lc.detach().cpu().item()):.6g} w={float(w):.3g} space={str(var_loss_space)}",
                                flush=True,
                            )

                if w_sum > 0.0:
                    loss_var = acc / float(w_sum)
                else:
                    loss_var = mu_sim.new_zeros(())

                loss_total_base = loss_psth + float(lambda_celltype) * loss_type + float(lam_var_eff) * loss_var
                var_val = float(loss_var.detach().cpu().item())

            loss_J_t, J_frob_t, J_maxabs_t = _compute_J_regularization_stats(net)
            loss_total = loss_total_base + float(lam_J_eff) * loss_J_t
            loss_J_val = float(loss_J_t.detach().cpu().item())
            J_frob_val = float(J_frob_t.detach().cpu().item())
            J_maxabs_val = float(J_maxabs_t.detach().cpu().item())

            if not tch.isfinite(loss_total):
                with tch.no_grad():
                    msg = {
                        "ep": ep + 1,
                        "unit_key": batch.get("unit_key", "NA"),
                        "loss_total": str(loss_total.detach().cpu().item()),
                        "loss_psth": str(loss_psth.detach().cpu().item()),
                        "loss_type": str(loss_type.detach().cpu().item()),
                        "loss_var": str(loss_var.detach().cpu().item()) if loss_var is not None else "NA",
                        "loss_J": str(loss_J_t.detach().cpu().item()),
                        "lam_var_eff": float(lam_var_eff),
                        "lam_J_eff": float(lam_J_eff),
                        "u_minmax": (float(u.min().cpu()), float(u.max().cpu())),
                        "psth_minmax": (float(psth_sub.min().cpu()), float(psth_sub.max().cpu())),
                        "rate_minmax": ("NA" if rates is None else (float(rates.min().cpu()), float(rates.max().cpu()))),
                        "J_norm": float(J_frob_t.detach().cpu().item()),
                        "W_norm": float(net.W_in.norm().detach().cpu()),
                        "dt_over_tau": float(net.dt / net.tau),
                        "substeps": int(getattr(net, "substeps", 1)),
                    }
                history_rows.append(
                    {
                        "epoch": int(ep + 1),
                        "unit_key": str(batch.get("unit_key", "NA")),
                        "train_total": float("nan"),
                        "train_psth": float(loss_psth.detach().cpu().item()) if bool(tch.isfinite(loss_psth).detach().cpu().item()) else float("nan"),
                        "train_type": float(loss_type.detach().cpu().item()) if bool(tch.isfinite(loss_type).detach().cpu().item()) else float("nan"),
                        "train_J": float(loss_J_val),
                        "lambda_J_eff": float(lam_J_eff),
                        "J_frob": float(J_frob_val),
                        "J_maxabs": float(J_maxabs_val),
                        "eval_total": float("nan"),
                        "eval_psth": float("nan"),
                        "eval_type": float("nan"),
                        "eval_J": float("nan"),
                    }
                )
                _flush_history_safe()
                raise FloatingPointError("Non-finite loss detected:\n" + json.dumps(msg, indent=2))

            loss_total.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                tch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
            opt.step()
            if bool(dale):
                net.apply_dale_mask()

            total_val = float(loss_total.detach().cpu().item())
            psth_val = float(loss_psth.detach().cpu().item())
            type_val = float(loss_type.detach().cpu().item())
            eval_total = float("nan")
            eval_psth = float("nan")
            eval_type = float("nan")
            eval_J = float("nan")

            if total_val < best_total:
                best_total = total_val
                best_train_ep = int(ep + 1)

            if str(best_metric) == "train_step_total" and total_val < best_score:
                best_score = total_val
                best_ep = int(ep + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            if (ep == 0) or ((ep + 1) % max(1, int(print_every)) == 0):
                elapsed = time.time() - t0
                n_types = int(ct_info.get("n_valid_types", 0)) if isinstance(ct_info, dict) else 0
                if bool(phase3_var_enabled):
                    nm = str(noise_mode)
                    sig = float(sigma_x0)
                    noise_tag = "none" if (nm == "none" or sig <= 0.0) else (f"x0(s={sig:.3g})" if nm == "x0_gaussian" else nm)
                    print(
                        "[global] ep=%d/%d total=%.6f mean=%.6f type=%.6f var=%.6f J=%.6g (lam_ct=%.4g, lam_var=%.4g, lam_J=%.4g) noise=%s Jf=%.6g Jmax=%.6g best=%.6f unit=%s (K=%d,nType=%d) elapsed=%.1fs"
                        % (
                            ep + 1, int(max_epochs), total_val, psth_val, type_val, float(var_val), loss_J_val,
                            float(lambda_celltype), float(lam_var_eff), float(lam_J_eff), str(noise_tag),
                            J_frob_val, J_maxabs_val,
                            float(best_total), batch.get("unit_key", "NA"), int(idx_net.numel()), n_types, elapsed,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "[global] ep=%d/%d total=%.6f psth=%.6f type=%.6f J=%.6g (lam_ct=%.4g, lam_J=%.4g) Jf=%.6g Jmax=%.6g best=%.6f unit=%s (K=%d,nType=%d) elapsed=%.1fs"
                        % (
                            ep + 1, int(max_epochs), total_val, psth_val, type_val, loss_J_val,
                            float(lambda_celltype), float(lam_J_eff),
                            J_frob_val, J_maxabs_val,
                            float(best_total), batch.get("unit_key", "NA"), int(idx_net.numel()), n_types, elapsed,
                        ),
                        flush=True,
                    )

            do_eval = (int(eval_every) > 0) and ((ep == 0) or ((ep + 1) % int(eval_every) == 0))
            if do_eval:
                ev = _global_eval_units(
                    net=net,
                    units=units,
                    loss_fn=loss_fn,
                    lambda_celltype=float(lambda_celltype),
                    lambda_J_eff=float(lam_J_eff),
                    noise_std_eval=0.0,
                    use_trials=bool(use_trials),
                )
                eval_total = float(ev["total"])
                eval_psth = float(ev["psth"])
                eval_type = float(ev["type"])
                eval_J = float(ev["J"])
                eval_J_frob = float(ev["J_frob"])
                eval_J_maxabs = float(ev["J_maxabs"])
                sel_val = None
                if str(best_metric) == "eval_psth":
                    sel_val = eval_psth
                elif str(best_metric) == "eval_total":
                    sel_val = eval_total

                if sel_val is not None and sel_val < best_score:
                    best_score = float(sel_val)
                    best_ep = int(ep + 1)
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    best_eval_stats = {
                        "total": eval_total,
                        "psth": eval_psth,
                        "type": eval_type,
                        "J": eval_J,
                        "J_frob": eval_J_frob,
                        "J_maxabs": eval_J_maxabs,
                        "n_units": int(ev.get("n_units", len(units))),
                        "n_units_with_type": int(ev.get("n_units_with_type", 0)),
                    }

                if bool(phase3_var_enabled):
                    print(
                        "[eval-det] ep=%d/%d best_metric=%s score=%.6f | total=%.6f psth=%.6f type=%.6f J=%.6g (lam_ct=%.4g, lam_J=%.4g) Jf=%.6g Jmax=%.6g | bestSel=%.6f@ep%d"
                        % (
                            ep + 1, int(max_epochs), str(best_metric),
                            float("nan") if sel_val is None else float(sel_val),
                            eval_total, eval_psth, eval_type, eval_J,
                            float(lambda_celltype), float(lam_J_eff), eval_J_frob, eval_J_maxabs,
                            float(best_score), int(best_ep),
                        ),
                        flush=True,
                    )

                    evs = _global_eval_units_sampled(
                        net=net,
                        units=units,
                        loss_fn=loss_fn,
                        lambda_celltype=float(lambda_celltype),
                        use_trials=bool(use_trials),
                        noise_mode=str(noise_mode),
                        sigma_x0=float(sigma_x0),
                        n_sim_trials_eval=int(n_sim_trials_eval),
                        var_loss_space=str(var_loss_space),
                        var_loss_eps=float(var_loss_eps),
                        var_unbiased=bool(var_unbiased),
                        min_trials_for_var_real=int(min_trials_for_var_real),
                        var_loss_weighting=str(var_loss_weighting),
                        eval_seed_base=int(eval_seed_base),
                        fixed_eval_seed_bank=bool(fixed_eval_seed_bank),
                        debug_var_loss=bool(debug_var_loss),
                        debug_noise=bool(debug_noise),
                    )
                    print(
                        "[eval-smp] ep=%d/%d mean=%.6f var=%.6f type=%.6f n_sim=%d sigma_x0=%.3g space=%s"
                        % (
                            ep + 1, int(max_epochs),
                            float(evs.get("mean", float("nan"))),
                            float(evs.get("var", float("nan"))),
                            float(evs.get("type", float("nan"))),
                            int(n_sim_trials_eval),
                            float(sigma_x0),
                            str(var_loss_space),
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "[eval] ep=%d/%d best_metric=%s score=%.6f | global total=%.6f psth=%.6f type=%.6f J=%.6g (lam_ct=%.4g, lam_J=%.4g) Jf=%.6g Jmax=%.6g | bestSel=%.6f@ep%d"
                        % (
                            ep + 1, int(max_epochs), str(best_metric),
                            float("nan") if sel_val is None else float(sel_val),
                            eval_total, eval_psth, eval_type, eval_J,
                            float(lambda_celltype), float(lam_J_eff), eval_J_frob, eval_J_maxabs,
                            float(best_score), int(best_ep),
                        ),
                        flush=True,
                    )

            history_rows.append(
                {
                    "epoch": int(ep + 1),
                    "unit_key": str(batch.get("unit_key", "NA")),
                    "train_total": float(total_val),
                    "train_psth": float(psth_val),
                    "train_type": float(type_val),
                    "train_J": float(loss_J_val),
                    "lambda_J_eff": float(lam_J_eff),
                    "J_frob": float(J_frob_val),
                    "J_maxabs": float(J_maxabs_val),
                    "eval_total": float(eval_total),
                    "eval_psth": float(eval_psth),
                    "eval_type": float(eval_type),
                    "eval_J": float(eval_J),
                }
            )

            if do_eval:
                _flush_history_safe()

            if best_state is not None and int(save_best_every) > 0 and ((ep + 1) % int(save_best_every) == 0):
                _atomic_torch_save(best_state, best_path)
            if int(save_latest_every) > 0 and ((ep + 1) % int(save_latest_every) == 0):
                _atomic_torch_save(net.state_dict(), latest_path)
    except Exception:
        _flush_history_safe()
        raise

    # ----------------------------
    # 6) Save final best + metadata
    # ----------------------------
    if best_state is None:
        # Can happen if best_metric is eval_* and eval_every<=0.
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_ep = int(max_epochs)
        if str(best_metric) == "train_step_total":
            best_score = float(best_total)
        else:
            best_score = float("nan")

    _atomic_torch_save(best_state, best_path)
    _atomic_torch_save(best_state, model_path)  # legacy alias for eval scripts expecting .pt
    _flush_history_safe()

    meta_out = {
        "animal": animal,
        "registry_csv": registry_csv,
        "n_exc_virtual": int(n_exc_virtual),
        "n_obs_inh": int(observed_sign_stats["n_obs_inh"]),
        "n_obs_exc": int(observed_sign_stats["n_obs_exc"]),
        "n_obs_total": int(observed_sign_stats["n_obs_total"]),
        "registry_has_explicit_cell_sign": bool(observed_sign_stats["registry_has_explicit_cell_sign"]),
        "n_total": int(n_total),
        "D_in": int(D_in),
        "recurrent_mode": str(recurrent_mode),
        "recurrent_rank": int(getattr(net, "recurrent_rank", recurrent_rank)),
        "random_bg_scale": float(random_bg_scale),
        "n_trainable_params": int(n_trainable_params),
        "n_recurrent_trainable_params": int(n_recurrent_trainable_params),
        "dt": float(dt),
        "tau": float(tau),
        "substeps": int(substeps),
        "nonlinearity": str(nonlinearity),
        "dale": bool(dale),
        "psth_bin_ms": float(psth_bin_ms),
        "sample_ignore_ms": float(sample_ignore_ms),
        "resp_sec": float(resp_sec),
        "best_total": float(best_total),
        "best_train_step_epoch": int(best_train_ep),
        "best_epoch": int(best_ep),
        "best_metric": str(best_metric),
        "best_score": float(best_score),
        "unit_sampling": str(unit_sampling),
        "max_sessions": None if max_sessions is None else int(max_sessions),
        "lambda_J": float(lambda_J),
        "lambda_J_warmup_epochs": int(lambda_J_warmup_epochs),
        "lambda_J_ramp_epochs": int(lambda_J_ramp_epochs),
        "lambda_celltype": float(lambda_celltype),
        "celltype_label_keys": [str(x) for x in celltype_label_keys],
        "celltype_registry_cols": [str(x) for x in celltype_registry_cols],
        "celltype_exclude": [str(x) for x in celltype_exclude],
        "celltype_min_count": int(celltype_min_count),
        "save_best_every": int(save_best_every),
        "save_latest_every": int(save_latest_every),
        "eval_every": int(eval_every),
        "best_ckpt_path": best_path,
        "legacy_model_alias": model_path,
        "train_config_path": params_path,
        "train_history_csv": history_csv_path,
        "loss_curves_png": history_png_path,
        "best_eval_stats": best_eval_stats,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[OK] Saved best model -> {best_path}", flush=True)
    print(f"[OK] Saved legacy alias -> {model_path}", flush=True)
    print(f"[OK] Saved meta -> {meta_path}", flush=True)
    print(f"[OK] Saved train config -> {params_path}", flush=True)
    print(f"[OK] Saved training history -> {history_csv_path}", flush=True)
    print(f"[OK] Saved loss curves -> {history_png_path}", flush=True)
