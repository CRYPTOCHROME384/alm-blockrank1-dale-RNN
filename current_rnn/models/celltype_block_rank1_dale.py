import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse_scalar(x: float) -> float:
    x = float(max(x, 1e-12))
    return float(math.log(math.expm1(x)))


class _Rank1BlockParams(nn.Module):
    def __init__(
        self,
        n_post: int,
        n_pre: int,
        *,
        init_A: float,
        init_factor_scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.n_post = int(n_post)
        self.n_pre = int(n_pre)
        self.eps = float(eps)

        self.u_raw = nn.Parameter(torch.zeros(self.n_post))
        self.v_raw = nn.Parameter(torch.zeros(self.n_pre))
        self.A_raw = nn.Parameter(torch.tensor(_softplus_inverse_scalar(init_A), dtype=torch.float32))

        if float(init_factor_scale) > 0.0:
            nn.init.normal_(self.u_raw, mean=0.0, std=float(init_factor_scale))
            nn.init.normal_(self.v_raw, mean=0.0, std=float(init_factor_scale))

    def positive_u(self) -> torch.Tensor:
        return F.softplus(self.u_raw)

    def positive_v(self) -> torch.Tensor:
        return F.softplus(self.v_raw)

    def positive_A(self) -> torch.Tensor:
        return F.softplus(self.A_raw)

    def normalized_u(self) -> torch.Tensor:
        u = self.positive_u()
        return u / (u.mean() + self.eps)

    def normalized_v(self) -> torch.Tensor:
        v = self.positive_v()
        return v / (v.mean() + self.eps)


class CellTypeBlockRank1DaleCurrentRNN(nn.Module):
    """Current-based RNN with block rank-1 Dale-constrained recurrent structure.

    Recurrent matrix is defined blockwise:

      J^{a<-b}_{ij} = s_b * A^{a<-b} * u^{a<-b}_i * v^{a<-b}_j / N_b

    where `u` and `v` are positive and mean-normalized inside each block,
    and `s_b` is determined purely by presynaptic type `b`.
    """

    def __init__(
        self,
        *,
        N: int,
        D_in: int,
        neuron_type_index: torch.Tensor,
        type_names: List[str],
        type_signs: List[float],
        dt: float = 1.0,
        tau: float = 1.0,
        substeps: int = 1,
        nonlinearity: str = "tanh",
        block_rank: int = 1,
        factor_nonlinearity: str = "softplus",
        A_nonlinearity: str = "softplus",
        normalize_uv: bool = True,
        eps: float = 1e-8,
        init_A: float = 0.10,
        init_factor_scale: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.N = int(N)
        self.D_in = int(D_in)
        self.dt = float(dt)
        self.tau = float(tau)
        self.substeps = int(substeps)
        self.nonlinearity = str(nonlinearity).lower()
        self.block_rank = int(block_rank)
        self.factor_nonlinearity = str(factor_nonlinearity).lower()
        self.A_nonlinearity = str(A_nonlinearity).lower()
        self.normalize_uv = bool(normalize_uv)
        self.eps = float(eps)
        self.init_A = float(init_A)
        self.init_factor_scale = float(init_factor_scale)

        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        if self.block_rank != 1:
            raise ValueError(f"Only block_rank=1 is supported, got {self.block_rank}")
        if self.factor_nonlinearity != "softplus":
            raise ValueError(f"Only factor_nonlinearity='softplus' is supported, got {factor_nonlinearity!r}")
        if self.A_nonlinearity != "softplus":
            raise ValueError(f"Only A_nonlinearity='softplus' is supported, got {A_nonlinearity!r}")
        if not self.normalize_uv:
            raise ValueError("This implementation requires normalize_uv=True.")

        type_names = [str(x) for x in type_names]
        if len(type_names) == 0:
            raise ValueError("type_names must be non-empty.")
        if len(type_names) != len(type_signs):
            raise ValueError("type_names and type_signs must have the same length.")

        self.type_names = list(type_names)
        self.type_name_to_index = {name: i for i, name in enumerate(self.type_names)}
        self.num_types = int(len(self.type_names))

        neuron_type_index = torch.as_tensor(neuron_type_index, dtype=torch.long)
        if neuron_type_index.dim() != 1 or int(neuron_type_index.numel()) != self.N:
            raise ValueError(
                f"neuron_type_index must be 1D with length N={self.N}, got shape={tuple(neuron_type_index.shape)}"
            )
        if int(neuron_type_index.min().item()) < 0 or int(neuron_type_index.max().item()) >= self.num_types:
            raise ValueError("neuron_type_index contains out-of-range type ids.")

        sign_t = torch.as_tensor(type_signs, dtype=torch.float32)
        if sign_t.shape != (self.num_types,):
            raise ValueError(f"type_signs must have shape [{self.num_types}], got {tuple(sign_t.shape)}")
        bad = [self.type_names[i] for i, v in enumerate(sign_t.tolist()) if v not in (-1.0, 1.0)]
        if bad:
            raise ValueError(f"type_signs must be +/-1 only; bad types={bad}")

        self.register_buffer("neuron_type_index", neuron_type_index.clone())
        self.register_buffer("type_sign_tensor", sign_t.clone())

        self._type_buffer_names: List[str] = []
        type_counts = []
        for t in range(self.num_types):
            idx = torch.where(self.neuron_type_index == int(t))[0]
            if int(idx.numel()) <= 0:
                raise ValueError(f"Type {self.type_names[t]!r} has zero neurons.")
            buf_name = f"type_idx_{t}"
            self.register_buffer(buf_name, idx.clone())
            self._type_buffer_names.append(buf_name)
            type_counts.append(int(idx.numel()))
        self.register_buffer("type_count_tensor", torch.as_tensor(type_counts, dtype=torch.float32))

        self.blocks = nn.ModuleDict()
        self.block_pairs: List[Dict[str, Any]] = []
        for a in range(self.num_types):
            idx_a = self.type_indices(a)
            for b in range(self.num_types):
                idx_b = self.type_indices(b)
                key = self._block_key(a, b)
                self.blocks[key] = _Rank1BlockParams(
                    n_post=int(idx_a.numel()),
                    n_pre=int(idx_b.numel()),
                    init_A=float(self.init_A),
                    init_factor_scale=float(self.init_factor_scale),
                    eps=float(self.eps),
                )
                self.block_pairs.append(
                    {
                        "key": key,
                        "post_index": int(a),
                        "pre_index": int(b),
                        "post_type": str(self.type_names[a]),
                        "pre_type": str(self.type_names[b]),
                    }
                )

        param_device = device
        W_in = torch.empty(self.N, self.D_in, device=param_device)
        nn.init.kaiming_uniform_(W_in, a=math.sqrt(5))
        W_in = W_in / math.sqrt(max(self.D_in, 1))
        self.W_in = nn.Parameter(W_in)
        self.b = nn.Parameter(torch.zeros(self.N, device=param_device))

        if device is not None:
            self.to(device)

    @staticmethod
    def _block_key(post_index: int, pre_index: int) -> str:
        return f"post{int(post_index)}__pre{int(pre_index)}"

    def type_indices(self, type_index: int) -> torch.Tensor:
        return getattr(self, self._type_buffer_names[int(type_index)])

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "tanh":
            return torch.tanh(x)
        if self.nonlinearity == "relu":
            return F.relu(x)
        if self.nonlinearity == "softplus":
            return F.softplus(x)
        raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")

    def _block_factors(self, post_index: int, pre_index: int) -> Dict[str, torch.Tensor]:
        block = self.blocks[self._block_key(post_index, pre_index)]
        u_norm = block.normalized_u()
        v_norm = block.normalized_v()
        A_pos = block.positive_A()
        sign = self.type_sign_tensor[int(pre_index)]
        n_pre = float(self.type_indices(pre_index).numel())
        return {
            "u_norm": u_norm,
            "v_norm": v_norm,
            "A_pos": A_pos,
            "sign": sign,
            "N_pre": n_pre,
        }

    def _build_forward_block_cache(self) -> Dict[str, Any]:
        u_rows_by_post: List[List[torch.Tensor]] = [[] for _ in range(self.num_types)]
        v_rows_by_pre: List[List[torch.Tensor]] = [[] for _ in range(self.num_types)]
        gamma_scale_by_post: List[torch.Tensor] = []
        n_pre_inv: List[float] = []

        for b in range(self.num_types):
            n_pre_inv.append(1.0 / float(self.type_indices(b).numel()))

        for a in range(self.num_types):
            gamma_row: List[torch.Tensor] = []
            for b in range(self.num_types):
                fac = self._block_factors(a, b)
                u_rows_by_post[a].append(fac["u_norm"])
                v_rows_by_pre[b].append(fac["v_norm"])
                gamma_row.append(fac["sign"] * fac["A_pos"])
            gamma_scale_by_post.append(torch.stack(gamma_row, dim=0))

        return {
            "u_stack_by_post": [torch.stack(rows, dim=0) for rows in u_rows_by_post],
            "v_stack_by_pre": [torch.stack(rows, dim=0) for rows in v_rows_by_pre],
            "gamma_scale_by_post": gamma_scale_by_post,
            "n_pre_inv": n_pre_inv,
        }

    def _recurrent_input_from_rate(self, rate_t: torch.Tensor, block_cache: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if rate_t.dim() != 2 or int(rate_t.shape[1]) != self.N:
            raise ValueError(f"rate_t must have shape [B, N={self.N}], got {tuple(rate_t.shape)}")

        if block_cache is None:
            block_cache = self._build_forward_block_cache()

        rec = torch.zeros_like(rate_t)
        kappa_by_pre: List[torch.Tensor] = []
        for b in range(self.num_types):
            idx_b = self.type_indices(b)
            rate_b = rate_t.index_select(dim=1, index=idx_b)
            v_stack = block_cache["v_stack_by_pre"][b]
            kappa_b = torch.matmul(rate_b, v_stack.transpose(0, 1)) * float(block_cache["n_pre_inv"][b])
            kappa_by_pre.append(kappa_b)

        kappa_tensor = torch.stack(kappa_by_pre, dim=2)
        for a in range(self.num_types):
            idx_a = self.type_indices(a)
            gamma = kappa_tensor[:, a, :] * block_cache["gamma_scale_by_post"][a].view(1, -1)
            rec[:, idx_a] = torch.matmul(gamma, block_cache["u_stack_by_post"][a])

        return rec

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
        if int(D_in) != self.D_in:
            raise ValueError(f"Expected input dimension D_in={self.D_in}, got {D_in}")

        device = self.W_in.device
        dtype = self.W_in.dtype
        u = u.to(device=device, dtype=dtype)

        if h0 is None:
            h_t = torch.zeros(B, self.N, device=device, dtype=dtype)
        else:
            if tuple(h0.shape) != (B, self.N):
                raise ValueError(f"h0 must have shape [B, N]=[{B}, {self.N}], got {tuple(h0.shape)}")
            h_t = h0.to(device=device, dtype=dtype)

        block_cache = self._build_forward_block_cache()
        inp_seq = torch.matmul(u, self.W_in.transpose(0, 1))
        dt_sub = self.dt / self.substeps
        dt_over_tau_sub = dt_sub / self.tau
        sqrt_dt_sub = math.sqrt(dt_sub)
        noise_scale = 1.0 / math.sqrt(self.substeps) if self.substeps > 1 else 1.0

        h_seq = []
        for t in range(T):
            inp_t = inp_seq[:, t, :]
            for _ in range(self.substeps):
                rate_t = self._phi(h_t)
                rec_t = self._recurrent_input_from_rate(rate_t, block_cache=block_cache)
                drift = -h_t + rec_t + inp_t + self.b
                h_t = h_t + dt_over_tau_sub * drift
                if float(noise_std) > 0.0:
                    noise = float(noise_std) * noise_scale * sqrt_dt_sub * torch.randn_like(h_t)
                    h_t = h_t + noise
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=1)
        out = {"h": h_seq}
        if bool(return_rate):
            out["rate"] = self._phi(h_seq)
        return out

    def get_recurrent_rank(self) -> int:
        return 1

    def recurrent_trainable_parameter_count(self) -> int:
        total = 0
        for block in self.blocks.values():
            total += int(block.u_raw.numel() + block.v_raw.numel() + block.A_raw.numel())
        return int(total)

    def equivalent_full_rank_parameter_count(self) -> int:
        return int(self.N * self.N)

    def recurrent_regularization_loss(self, *, A_l2: float = 0.0, uv_l2: float = 0.0) -> torch.Tensor:
        acc = self.b.new_zeros(())
        if float(A_l2) <= 0.0 and float(uv_l2) <= 0.0:
            return acc

        n_blocks = max(len(self.block_pairs), 1)
        for pair in self.block_pairs:
            fac = self._block_factors(int(pair["post_index"]), int(pair["pre_index"]))
            if float(A_l2) > 0.0:
                acc = acc + float(A_l2) * fac["A_pos"].pow(2)
            if float(uv_l2) > 0.0:
                acc = acc + float(uv_l2) * (
                    (fac["u_norm"] - 1.0).pow(2).mean() + (fac["v_norm"] - 1.0).pow(2).mean()
                )
        return acc / float(n_blocks)

    def recurrent_regularization_stats(self) -> Dict[str, float]:
        frob_sq = 0.0
        maxabs = 0.0
        for pair in self.block_pairs:
            fac = self._block_factors(int(pair["post_index"]), int(pair["pre_index"]))
            A_pos = float(fac["A_pos"].detach().cpu().item())
            u_norm = fac["u_norm"].detach().cpu()
            v_norm = fac["v_norm"].detach().cpu()
            N_pre = float(fac["N_pre"])

            block_frob_sq = (A_pos ** 2) * float(u_norm.pow(2).sum().item()) * float(v_norm.pow(2).sum().item()) / (N_pre ** 2)
            block_maxabs = A_pos * float(u_norm.max().item()) * float(v_norm.max().item()) / N_pre
            frob_sq += block_frob_sq
            maxabs = max(maxabs, block_maxabs)

        mean_sq = frob_sq / float(max(self.N * self.N, 1))
        return {
            "J_mean_sq": float(mean_sq),
            "J_frob": float(math.sqrt(max(frob_sq, 0.0))),
            "J_maxabs": float(maxabs),
        }

    @torch.no_grad()
    def apply_dale_mask(self) -> None:
        return

    @torch.no_grad()
    def materialize_J_for_debug(self) -> torch.Tensor:
        J = torch.zeros(self.N, self.N, device=self.W_in.device, dtype=self.W_in.dtype)
        for pair in self.block_pairs:
            a = int(pair["post_index"])
            b = int(pair["pre_index"])
            idx_a = self.type_indices(a)
            idx_b = self.type_indices(b)
            fac = self._block_factors(a, b)
            block_J = fac["sign"] * fac["A_pos"] * torch.outer(fac["u_norm"], fac["v_norm"]) / float(fac["N_pre"])
            J[idx_a[:, None], idx_b[None, :]] = block_J
        return J

    @torch.no_grad()
    def dale_violation_count(self, tol: float = 1e-12) -> int:
        J = self.materialize_J_for_debug().detach()
        count = 0
        for b in range(self.num_types):
            idx_b = self.type_indices(b)
            col_block = J.index_select(dim=1, index=idx_b)
            sign = float(self.type_sign_tensor[b].item())
            if sign > 0:
                count += int((col_block < -float(tol)).sum().item())
            else:
                count += int((col_block > float(tol)).sum().item())
        return int(count)

    @torch.no_grad()
    def numerical_block_ranks(self, tol: float = 1e-5) -> Dict[str, int]:
        ranks: Dict[str, int] = {}
        for pair in self.block_pairs:
            a = int(pair["post_index"])
            b = int(pair["pre_index"])
            idx_a = self.type_indices(a)
            idx_b = self.type_indices(b)
            fac = self._block_factors(a, b)
            block_J = fac["sign"] * fac["A_pos"] * torch.outer(fac["u_norm"], fac["v_norm"]) / float(fac["N_pre"])
            svals = torch.linalg.svdvals(block_J)
            if int(svals.numel()) == 0:
                rank = 0
            else:
                thr = float(tol) * float(svals.max().item())
                rank = int((svals > thr).sum().item())
            ranks[f"{pair['post_type']}<-{pair['pre_type']}"] = int(rank)
        return ranks

    @torch.no_grad()
    def block_parameter_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for pair in self.block_pairs:
            a = int(pair["post_index"])
            b = int(pair["pre_index"])
            fac = self._block_factors(a, b)
            u_norm = fac["u_norm"].detach().cpu()
            v_norm = fac["v_norm"].detach().cpu()
            rows.append(
                {
                    "block": f"{pair['post_type']}<-{pair['pre_type']}",
                    "post_type": str(pair["post_type"]),
                    "pre_type": str(pair["pre_type"]),
                    "presyn_sign": int(float(fac["sign"].item())),
                    "n_post": int(u_norm.numel()),
                    "n_pre": int(v_norm.numel()),
                    "N_b": int(fac["N_pre"]),
                    "A_pos": float(fac["A_pos"].detach().cpu().item()),
                    "u_norm_mean": float(u_norm.mean().item()),
                    "u_norm_std": float(u_norm.std(unbiased=False).item()),
                    "u_norm_min": float(u_norm.min().item()),
                    "u_norm_max": float(u_norm.max().item()),
                    "v_norm_mean": float(v_norm.mean().item()),
                    "v_norm_std": float(v_norm.std(unbiased=False).item()),
                    "v_norm_min": float(v_norm.min().item()),
                    "v_norm_max": float(v_norm.max().item()),
                }
            )
        return rows
