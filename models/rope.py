import torch
import torch.nn as nn


class ROPE(nn.Module):
    """
    Rotary position embedding for attention q/k tensors.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, D/2)

        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)

        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len

    def _get_cos_sin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        needs_rebuild = (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        )
        if needs_rebuild:
            self._build_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return cos, sin

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to a tensor shaped (..., L, D).
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected last dim {self.head_dim}, got {x.shape[-1]} for tensor {tuple(x.shape)}"
            )
        cos, sin = self._get_cos_sin(x)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to q and k.

        Expected shapes: (..., L, D) where D == head_dim.
        """
        if q.shape != k.shape:
            raise ValueError(f"q and k must have same shape, got {tuple(q.shape)} vs {tuple(k.shape)}")
        return self.apply(q), self.apply(k)
