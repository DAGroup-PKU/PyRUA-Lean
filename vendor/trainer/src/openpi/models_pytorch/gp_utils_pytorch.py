"""PyTorch Gaussian Process utilities for structured noise sampling in Flow Matching.

Direct translation of the JAX gp_utils.py to PyTorch.
Provides kernel functions (SE, OU, PE), unconditional GP noise sampling,
and GPR (Gaussian Process Regression) conditional posterior sampling.

Designed for action_horizon=50, with execution_horizon=15 in mind.
Recommended length_scale values: SE=5.0, OU=3.0, PE=2.0
"""

from typing import Literal

import torch

# Type alias for supported kernel names
KernelType = Literal["SE", "OU", "PE"]


def squared_exponential_kernel(
    tau: torch.Tensor,
    tau_prime: torch.Tensor,
    length_scale: float,
) -> torch.Tensor:
    """Squared Exponential (SE) kernel: K_SE(τ, τ') = exp(-d² / (2 * ℓ²))"""
    d = tau - tau_prime
    return torch.exp(-d**2 / (2.0 * length_scale**2))


def ornstein_uhlenbeck_kernel(
    tau: torch.Tensor,
    tau_prime: torch.Tensor,
    length_scale: float,
) -> torch.Tensor:
    """Ornstein-Uhlenbeck (OU) kernel: K_OU(τ, τ') = exp(-|d| / ℓ)"""
    d = tau - tau_prime
    return torch.exp(-torch.abs(d) / length_scale)


def periodic_kernel(
    tau: torch.Tensor,
    tau_prime: torch.Tensor,
    length_scale: float,
) -> torch.Tensor:
    """Periodic (PE) kernel: K_PE(τ, τ') = exp(-2 * sin²(d) / ℓ²)"""
    d = tau - tau_prime
    return torch.exp(-2.0 * torch.sin(d) ** 2 / length_scale**2)


def _apply_kernel(
    tau_i: torch.Tensor,
    tau_j: torch.Tensor,
    kernel_type: KernelType,
    length_scale: float,
) -> torch.Tensor:
    """Apply the specified kernel function to the given time points."""
    if kernel_type == "SE":
        return squared_exponential_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "OU":
        return ornstein_uhlenbeck_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "PE":
        return periodic_kernel(tau_i, tau_j, length_scale)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}. Choose from 'SE', 'OU', 'PE'.")


def build_gp_covariance(
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Build the GP covariance matrix for a given action horizon.

    Returns:
        Covariance matrix K of shape (action_horizon, action_horizon).
    """
    tau = torch.arange(action_horizon, dtype=torch.float32)
    tau_i = tau[:, None]  # (T, 1)
    tau_j = tau[None, :]  # (1, T)

    K = _apply_kernel(tau_i, tau_j, kernel_type, length_scale)

    # Add jitter for numerical stability
    K = K + jitter * torch.eye(action_horizon)
    return K


def sample_gp_noise(
    shape: tuple[int, ...],
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample temporally-correlated noise using a Gaussian Process prior.

    Replaces i.i.d. N(0,I) with GP-structured noise: z = L @ eps, L = cholesky(K).

    Args:
        shape: Full noise shape (batch, action_horizon, action_dim).
        action_horizon: Number of timesteps; must match shape[-2].
        kernel_type: "SE" (smooth), "OU" (rough), or "PE" (periodic).
        length_scale: Kernel length scale ℓ.
        jitter: Small diagonal term for numerical stability in Cholesky.
        device: Device to create tensors on.

    Returns:
        GP-structured noise of shape `shape`.
    """
    assert shape[-2] == action_horizon, (
        f"shape[-2]={shape[-2]} must equal action_horizon={action_horizon}"
    )

    K = build_gp_covariance(action_horizon, kernel_type, length_scale, jitter)
    K = K.to(device=device)

    # Cholesky decomposition: K = L L^T
    L = torch.linalg.cholesky(K)

    # Sample standard Gaussian
    eps = torch.randn(shape, dtype=torch.float32, device=device)

    # Apply temporal correlation: z = L @ eps along the action_horizon axis
    # L: (T, T), eps: (..., T, D) -> noise: (..., T, D)
    noise = torch.einsum("ij,...jd->...id", L, eps)

    return noise


def compute_gpr_posterior(
    past_times: torch.Tensor,
    future_times: torch.Tensor,
    y_p: torch.Tensor,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GPR posterior mean and Cholesky factor given past executed actions.

    All linear algebra is done in float64 for numerical stability (the SE kernel
    matrix K_pp can have condition number ~1e6, making float32 solve unreliable).

    Args:
        past_times:   (P,)       integer time indices for past steps
        future_times: (F,)       integer time indices for future steps
        y_p:          (B, P, D)  past executed actions (normalised)
        kernel_type:  "SE"/"OU"/"PE"
        length_scale: kernel ℓ
        jitter:       diagonal regularisation for Cholesky stability

    Returns:
        mu    (B, F, D)   posterior mean  μ_{f|p} = K_fp K_pp⁻¹ y_p
        L_ff  (F, F)      lower Cholesky of posterior covariance Σ_{f|p}
    """
    P = past_times.shape[0]
    F = future_times.shape[0]
    device = y_p.device

    # --- Build kernel matrices in float64 for stability ---
    tp_i = past_times[:, None].double()     # (P, 1)
    tp_j = past_times[None, :].double()     # (1, P)
    tf_i = future_times[:, None].double()   # (F, 1)
    tf_j = future_times[None, :].double()   # (1, F)
    tp_col = past_times[None, :].double()   # (1, P)
    tf_row = future_times[:, None].double() # (F, 1)

    K_pp = _apply_kernel(tp_i, tp_j, kernel_type, length_scale).to(device=device)     # (P, P)
    K_ff = _apply_kernel(tf_i, tf_j, kernel_type, length_scale).to(device=device)     # (F, F)
    K_fp = _apply_kernel(tf_row, tp_col, kernel_type, length_scale).to(device=device) # (F, P)

    K_pp = K_pp + jitter * torch.eye(P, dtype=torch.float64, device=device)
    K_ff = K_ff + jitter * torch.eye(F, dtype=torch.float64, device=device)

    # --- Cholesky solve in float64: K_pp⁻¹ y_p ---
    y_p_f64 = y_p.to(dtype=torch.float64)
    L_pp = torch.linalg.cholesky(K_pp)  # (P, P)

    # alpha = K_pp⁻¹ y_p: broadcasts (P,P) x (B,P,D) -> (B,P,D)
    alpha = torch.linalg.solve(K_pp, y_p_f64)  # (B, P, D)

    # --- Posterior mean: μ_{f|p} = K_fp @ alpha ---
    mu = torch.einsum("fp,bpd->bfd", K_fp, alpha)  # (B, F, D)

    # --- Posterior covariance: Σ_{f|p} = K_ff - K_fp K_pp⁻¹ K_pf ---
    V = torch.linalg.solve(L_pp, K_fp.T)  # (P, F)
    Sigma_ff = K_ff - V.T @ V              # (F, F)

    # Add extra jitter to ensure Sigma_ff is positive definite for Cholesky.
    # With real robot data, Sigma_ff diagonal can be ~1e-3, so jitter=1e-5 is
    # insufficient. Use 1e-3 to match the JAX training fix.
    extra_jitter = 1e-3
    Sigma_ff = Sigma_ff + extra_jitter * torch.eye(F, dtype=torch.float64, device=device)

    # Cholesky for sampling
    L_ff = torch.linalg.cholesky(Sigma_ff)  # (F, F)

    # Cast back to float32
    return mu.float(), L_ff.float()


def sample_gpr_noise(
    y_p: torch.Tensor,
    past_horizon: int,
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Sample initial noise from the GPR posterior conditioned on past actions.

    x_0 ~ N(μ_{f|p}, Σ_{f|p})  where μ and Σ are determined by the GP kernel
    and the observed y_p (past executed actions).

    Args:
        y_p:            (B, P, D)  past executed actions, where P = past_horizon
        past_horizon:   number of past steps P (e.g. 15)
        action_horizon: number of future steps F (e.g. 50)
        kernel_type:    "SE" (recommended), "OU", or "PE"
        length_scale:   GP kernel ℓ
        jitter:         diagonal regularisation for numerical stability

    Returns:
        noise (B, action_horizon, D) — GPR-conditioned initial noise
    """
    device = y_p.device

    # Time grid: past=[0..P-1], future=[P..P+F-1]
    past_times = torch.arange(past_horizon, dtype=torch.float32, device=device)
    future_times = torch.arange(past_horizon, past_horizon + action_horizon, dtype=torch.float32, device=device)

    mu, L_ff = compute_gpr_posterior(
        past_times, future_times, y_p,
        kernel_type=kernel_type,
        length_scale=length_scale,
        jitter=jitter,
    )

    # Sample: x_0 = μ + L_ff @ ε,  ε ~ N(0, I)
    batch_size = y_p.shape[0]
    D = y_p.shape[2]
    eps = torch.randn(batch_size, action_horizon, D, dtype=torch.float32, device=device)
    noise = mu + torch.einsum("ij,bjd->bid", L_ff, eps)

    return noise
