# current_rnn/model_current.py

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ALMCurrentRNN(nn.Module):
    """
    High-dimensional current-based RNN for ALM data.

    - State h_t ∈ R^N corresponds 1:1 to recorded neurons (after cell-type filtering).
    - Discrete-time dynamics (Euler):

        h_{t+1} = h_t + (dt / tau) * [ -h_t + J * phi(h_t) + W_in * u_t + b ]

      where:
        - J: recurrent coupling (full-rank or low-rank parameterization)
        - W_in: NxD_in input weight
        - u_t: external input at time t, shape [B, D_in]
        - phi: element-wise nonlinearity

    - Forward input u: shape [B, T, D_in]
    - Forward output: dict with
        - "h":    shape [B, T, N] (current / latent state)
        - "rate": shape [B, T, N] (phi(h))
    """

    def __init__(
        self,
        N: int,
        D_in: int,
        dt: float = 1.0,
        tau: float = 1.0,
        substeps: int = 1,
        nonlinearity: str = "tanh",
        device: Optional[torch.device] = None,
        dale_mask: Optional[torch.Tensor] = None,
        recurrent_mode: str = "full",
        recurrent_rank: int = 0,
        random_bg_scale: float = 0.0,
    ) -> None:
        super().__init__()

        self.N = int(N)
        self.D_in = int(D_in)
        self.dt = float(dt)
        self.tau = float(tau)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.nonlinearity = str(nonlinearity).lower()

        self.recurrent_mode = self._canonicalize_recurrent_mode(recurrent_mode)
        self.recurrent_rank = int(recurrent_rank)
        self.random_bg_scale = float(random_bg_scale)

        if self.recurrent_mode == "full":
            self.recurrent_rank = 0
        elif self.recurrent_rank <= 0:
            raise ValueError(
                f"recurrent_rank must be >= 1 for recurrent_mode={self.recurrent_mode!r}, got {recurrent_rank}"
            )

        if dale_mask is not None:
            if dale_mask.shape != (self.N, self.N):
                raise ValueError(
                    f"dale_mask must have shape ({self.N}, {self.N}), got {tuple(dale_mask.shape)}"
                )
            self.register_buffer("dale_mask", dale_mask.clone())
        else:
            self.dale_mask = None

        param_device = device if device is not None else (self.dale_mask.device if self.dale_mask is not None else None)

        if self.recurrent_mode == "full":
            J = torch.empty(self.N, self.N, device=param_device)
            nn.init.kaiming_uniform_(J, a=math.sqrt(5))
            J = J / math.sqrt(self.N)
            self.J = nn.Parameter(J)
        else:
            U = torch.empty(self.N, self.recurrent_rank, device=param_device)
            V = torch.empty(self.N, self.recurrent_rank, device=param_device)
            nn.init.kaiming_uniform_(U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(V, a=math.sqrt(5))
            # Keep the induced low-rank J = U V^T / N in the same rough scale family
            # as the dense initialization across different ranks.
            lr_scale = math.pow(float(max(self.N, 1)) / float(max(self.recurrent_rank, 1)), 0.25)
            U = U * lr_scale
            V = V * lr_scale
            self.J_lr_u = nn.Parameter(U)
            self.J_lr_v = nn.Parameter(V)

            if self.recurrent_mode == "random_plus_low_rank":
                J_bg = torch.empty(self.N, self.N, device=param_device)
                nn.init.kaiming_uniform_(J_bg, a=math.sqrt(5))
                J_bg = (self.random_bg_scale * J_bg) / math.sqrt(self.N)
                J_bg = self._project_with_dale(J_bg)
                self.register_buffer("J_background", J_bg)
            else:
                self.J_background = None

        W_in = torch.empty(self.N, self.D_in, device=param_device)
        nn.init.kaiming_uniform_(W_in, a=math.sqrt(5))
        W_in = W_in / math.sqrt(max(self.D_in, 1))
        self.W_in = nn.Parameter(W_in)

        self.b = nn.Parameter(torch.zeros(self.N, device=param_device))

        if device is not None:
            self.to(device)

    @staticmethod
    def _canonicalize_recurrent_mode(mode: str) -> str:
        s = str(mode).strip().lower()
        aliases = {
            "full": "full",
            "dense": "full",
            "full_rank": "full",
            "lowrank": "low_rank",
            "low_rank": "low_rank",
            "random_plus_low_rank": "random_plus_low_rank",
            "structured_low_rank": "random_plus_low_rank",
            "full_plus_low_rank": "random_plus_low_rank",
            "background_plus_low_rank": "random_plus_low_rank",
        }
        if s not in aliases:
            raise ValueError(
                "recurrent_mode must be one of full / low_rank / random_plus_low_rank, "
                f"got {mode!r}"
            )
        return aliases[s]

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "tanh":
            return torch.tanh(x)
        if self.nonlinearity == "relu":
            return F.relu(x)
        if self.nonlinearity == "softplus":
            return F.softplus(x)
        raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")

    def _project_with_dale(self, J: torch.Tensor) -> torch.Tensor:
        if self.dale_mask is None:
            return J
        pos_mask = self.dale_mask > 0
        neg_mask = self.dale_mask < 0
        if pos_mask.any():
            J = torch.where(pos_mask, torch.clamp_min(J, 0.0), J)
        if neg_mask.any():
            J = torch.where(neg_mask, torch.clamp_max(J, 0.0), J)
        return J

    def get_recurrent_matrix(self, apply_dale: bool = True) -> torch.Tensor:
        if self.recurrent_mode == "full":
            J_eff = self.J
        else:
            J_eff = torch.matmul(self.J_lr_u, self.J_lr_v.transpose(0, 1)) / float(self.N)
            if self.recurrent_mode == "random_plus_low_rank" and self.J_background is not None:
                J_eff = J_eff + self.J_background
        if apply_dale:
            J_eff = self._project_with_dale(J_eff)
        return J_eff

    def get_recurrent_rank(self) -> int:
        if self.recurrent_mode == "full":
            return int(self.N)
        return int(self.recurrent_rank)

    def recurrent_trainable_parameter_count(self) -> int:
        if self.recurrent_mode == "full":
            return int(self.J.numel())
        return int(self.J_lr_u.numel() + self.J_lr_v.numel())

    @torch.no_grad()
    def apply_dale_mask(self) -> None:
        if self.dale_mask is None:
            return
        if self.recurrent_mode == "full":
            self.J.data = self._project_with_dale(self.J.data)

    def forward(
        self,
        u: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        return_rate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if u.dim() != 3:
            raise ValueError(f"u must have shape [B, T, D_in], got {tuple(u.shape)}")

        B, T, D_in = u.shape
        if D_in != self.D_in:
            raise ValueError(f"Expected input dimension D_in={self.D_in}, got {D_in}")

        device = self.W_in.device
        dtype = self.W_in.dtype
        u = u.to(device=device, dtype=dtype)
        J_eff = self.get_recurrent_matrix(apply_dale=True)

        if h0 is None:
            h_t = torch.zeros(B, self.N, device=device, dtype=dtype)
        else:
            if h0.shape != (B, self.N):
                raise ValueError(
                    f"h0 must have shape [B, N]=[{B}, {self.N}], got {tuple(h0.shape)}"
                )
            h_t = h0.to(device=device, dtype=dtype)

        dt_sub = self.dt / self.substeps
        dt_over_tau_sub = dt_sub / self.tau
        sqrt_dt_sub = math.sqrt(dt_sub)
        noise_scale = 1.0 / math.sqrt(self.substeps) if self.substeps > 1 else 1.0

        h_seq = []
        for t in range(T):
            u_t = u[:, t, :]
            for _ in range(self.substeps):
                rate_t = self._phi(h_t)
                rec_t = torch.matmul(rate_t, J_eff.transpose(0, 1))
                inp_t = torch.matmul(u_t, self.W_in.transpose(0, 1))
                drift = -h_t + rec_t + inp_t + self.b
                h_t = h_t + dt_over_tau_sub * drift
                if noise_std > 0.0:
                    noise = (noise_std * noise_scale) * sqrt_dt_sub * torch.randn_like(h_t)
                    h_t = h_t + noise
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=1)
        out = {"h": h_seq}
        if return_rate:
            out["rate"] = self._phi(h_seq)
        return out
