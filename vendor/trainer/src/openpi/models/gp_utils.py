"""Gaussian Process utilities for structured noise sampling in Flow Matching.

Provides three kernel functions (SE, OU, PE), unconditional GP noise sampling,
and GPR (Gaussian Process Regression) conditional posterior sampling.

Unconditional GP: replaces i.i.d. Gaussian noise with temporally-correlated noise.
GPR: conditions the initial noise on past executed actions y_p, so the noise
     starting point reflects the temporal context of the previous action chunk.
     This achieves cross-chunk temporal coherence without OT Coupling.

Designed for action_horizon=50, with execution_horizon=15 in mind.
Recommended length_scale values: SE=5.0, OU=3.0, PE=2.0
"""

from typing import Literal

import jax
import jax.numpy as jnp

# Type alias for supported kernel names
KernelType = Literal["SE", "OU", "PE"]


def squared_exponential_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Squared Exponential (SE) kernel: infinitely smooth samples.

    K_SE(τ, τ') = exp(-d² / (2 * ℓ²))

    Suitable for smooth time series. Stronger temporal correlations over longer
    horizons compared to OU. Recommended length_scale=5.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls correlation range. Higher = smoother.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-d**2 / (2.0 * length_scale**2))


def ornstein_uhlenbeck_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Ornstein-Uhlenbeck (OU) kernel: related to Brownian motion.

    K_OU(τ, τ') = exp(-|d| / ℓ)

    Produces rougher, more jagged samples. Decays faster than SE.
    Recommended length_scale=3.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls correlation range. Higher = longer memory.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-jnp.abs(d) / length_scale)


def periodic_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Periodic (PE) kernel: captures repeating patterns.

    K_PE(τ, τ') = exp(-2 * sin²(d) / ℓ²)

    Suitable for time series with periodic behavior (e.g., cyclic motions).
    Recommended length_scale=2.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls smoothness of periodicity.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-2.0 * jnp.sin(d) ** 2 / length_scale**2)


def build_gp_covariance(
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> jax.Array:
    """Build the GP covariance matrix for a given action horizon.

    Time points are uniformly spaced in [0, action_horizon - 1].
    The resulting matrix K has shape (action_horizon, action_horizon).

    Args:
        action_horizon: Number of time steps (e.g., 50).
        kernel_type: One of "SE", "OU", "PE".
        length_scale: Kernel length scale parameter ℓ.
        jitter: Small diagonal term for numerical stability.

    Returns:
        Covariance matrix K of shape (action_horizon, action_horizon).
    """
    tau = jnp.arange(action_horizon, dtype=jnp.float32)
    # Build pairwise difference matrix via broadcasting
    tau_i = tau[:, None]   # (T, 1)
    tau_j = tau[None, :]   # (1, T)

    if kernel_type == "SE":
        K = squared_exponential_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "OU":
        K = ornstein_uhlenbeck_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "PE":
        K = periodic_kernel(tau_i, tau_j, length_scale)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}. Choose from 'SE', 'OU', 'PE'.")

    # Add jitter for numerical stability
    K = K + jitter * jnp.eye(action_horizon)
    return K


def sample_gp_noise(
    rng: jax.Array,
    shape: tuple[int, ...],
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> jax.Array:
    """Sample temporally-correlated noise using a Gaussian Process prior.

    Replaces i.i.d. `jax.random.normal(...)` with GP-structured noise.
    The temporal correlation is applied along the action_horizon dimension.
    Each action dimension is sampled independently from the same GP.

    Noise is sampled via: z = L @ eps, where L = cholesky(K), eps ~ N(0,I)

    Args:
        rng: JAX random key.
        shape: Full noise shape (batch, action_horizon, action_dim).
        action_horizon: Number of timesteps; must match shape[-2].
        kernel_type: "SE" (smooth), "OU" (rough), or "PE" (periodic).
        length_scale: Kernel length scale ℓ. Recommended values:
            SE=5.0, OU=3.0, PE=2.0 for action_horizon=50, exec_horizon=15.
        jitter: Small diagonal term for numerical stability in Cholesky.

    Returns:
        GP-structured noise of shape `shape`, with temporal correlations
        along the action_horizon axis.
    """
    assert shape[-2] == action_horizon, (
        f"shape[-2]={shape[-2]} must equal action_horizon={action_horizon}"
    )

    # Build covariance matrix K: (action_horizon, action_horizon)
    K = build_gp_covariance(action_horizon, kernel_type, length_scale, jitter)

    # Cholesky decomposition: K = L L^T, L lower-triangular
    # shape: (action_horizon, action_horizon)
    L = jnp.linalg.cholesky(K)

    # Sample standard Gaussian: shape (batch, action_horizon, action_dim)
    eps = jax.random.normal(rng, shape)

    # Apply temporal correlation: z = L @ eps along the action_horizon axis
    # eps: (..., T, D) -> einsum over T dimension
    # L: (T, T), eps: (..., T, D) -> noise: (..., T, D)
    noise = jnp.einsum("ij,...jd->...id", L, eps)

    return noise


def compute_gpr_posterior(
    past_times: jax.Array,
    future_times: jax.Array,
    y_p: jax.Array,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Compute GPR posterior mean and Cholesky factor given past executed actions.

    All time points live on a shared integer grid (e.g. past=[0..P-1],
    future=[P..P+F-1]) so the kernel sees the correct temporal distances.

    Args:
        past_times:   (P,)         integer time indices for past steps
        future_times: (F,)         integer time indices for future steps
        y_p:          (B, P, D)    past executed actions (normalised)
        kernel_type:  "SE"/"OU"/"PE"
        length_scale: kernel ℓ
        jitter:       diagonal regularisation for Cholesky stability

    Returns:
        mu    (B, F, D)   posterior mean  μ_{f|p} = Σ_fp Σ_pp⁻¹ y_p
        L_ff  (F, F)      lower Cholesky of posterior covariance Σ_{f|p}
    """
    P = past_times.shape[0]
    F = future_times.shape[0]

    # --- Build kernel matrices (batch-independent) ---
    # Shape helpers: (P,1)x(1,P), (F,1)x(1,F), (F,1)x(1,P)
    tp_i, tp_j = past_times[:, None].astype(jnp.float32), past_times[None, :].astype(jnp.float32)
    tf_i, tf_j = future_times[:, None].astype(jnp.float32), future_times[None, :].astype(jnp.float32)
    tp_col, tf_row = past_times[None, :].astype(jnp.float32), future_times[:, None].astype(jnp.float32)

    if kernel_type == "SE":
        K_pp = squared_exponential_kernel(tp_i, tp_j, length_scale)   # (P, P)
        K_ff = squared_exponential_kernel(tf_i, tf_j, length_scale)   # (F, F)
        K_fp = squared_exponential_kernel(tf_row, tp_col, length_scale)  # (F, P)
    elif kernel_type == "OU":
        K_pp = ornstein_uhlenbeck_kernel(tp_i, tp_j, length_scale)
        K_ff = ornstein_uhlenbeck_kernel(tf_i, tf_j, length_scale)
        K_fp = ornstein_uhlenbeck_kernel(tf_row, tp_col, length_scale)
    elif kernel_type == "PE":
        K_pp = periodic_kernel(tp_i, tp_j, length_scale)
        K_ff = periodic_kernel(tf_i, tf_j, length_scale)
        K_fp = periodic_kernel(tf_row, tp_col, length_scale)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}. Choose from 'SE', 'OU', 'PE'.")

    K_pp = K_pp + jitter * jnp.eye(P)
    K_ff = K_ff + jitter * jnp.eye(F)

    # --- Cholesky solve: Σ_pp⁻¹ y_p  (numerically stable) ---
    # L_pp: (P, P),  solve via triangular systems
    L_pp = jnp.linalg.cholesky(K_pp)  # (P, P)

    # alpha = Σ_pp⁻¹ y_p, computed per-batch via two triangular solves
    # y_p: (B, P, D) -> alpha: (B, P, D)
    # Using jnp.linalg.solve on batched y_p:
    #   K_pp @ alpha = y_p  =>  we need to broadcast K_pp (P,P) over batch
    alpha = jnp.linalg.solve(K_pp, y_p)  # (B, P, D)  — JAX broadcasts (P,P) x (B,P,D)

    # --- Posterior mean: μ_{f|p} = K_fp @ alpha ---
    # K_fp: (F, P),  alpha: (B, P, D)  ->  mu: (B, F, D)
    mu = jnp.einsum("fp,bpd->bfd", K_fp, alpha)

    # --- Posterior covariance: Σ_{f|p} = K_ff - K_fp Σ_pp⁻¹ K_pf ---
    # K_pf = K_fp.T,  K_fp Σ_pp⁻¹ K_pf computed via Cholesky:
    #   V = L_pp⁻¹ K_pf  =>  K_fp Σ_pp⁻¹ K_pf = V.T V
    V = jnp.linalg.solve(L_pp, K_fp.T)  # (P, F)
    Sigma_ff = K_ff - V.T @ V           # (F, F)

    # Cholesky for sampling
    L_ff = jnp.linalg.cholesky(Sigma_ff)  # (F, F)

    return mu, L_ff


def sample_gpr_noise(
    rng: jax.Array,
    y_p: jax.Array,
    past_horizon: int,
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> jax.Array:
    """Sample initial noise from the GPR posterior conditioned on past actions.

    Replaces unconditional GP sampling with a data-conditioned distribution:
        x_0 ~ N(μ_{f|p}, Σ_{f|p})
    where μ and Σ are determined by the GP kernel and the observed y_p.

    This ensures the noise starting point is informed by the previous chunk's
    executed trajectory, achieving cross-chunk temporal continuity.

    Training and inference use the identical formula (no train-inference gap).

    Args:
        rng:           JAX random key
        y_p:           (B, P, D)  past executed actions, where P = past_horizon
        past_horizon:  number of past steps P (e.g. 15)
        action_horizon: number of future steps F (e.g. 50)
        kernel_type:   "SE" (recommended), "OU", or "PE"
        length_scale:  GP kernel ℓ; controls how far past influences future
        jitter:        diagonal regularisation for numerical stability

    Returns:
        noise (B, action_horizon, D)  — GPR-conditioned initial noise
    """
    # Time grid: past=[0..P-1], future=[P..P+F-1]
    past_times   = jnp.arange(past_horizon, dtype=jnp.float32)
    future_times = jnp.arange(past_horizon, past_horizon + action_horizon, dtype=jnp.float32)

    mu, L_ff = compute_gpr_posterior(
        past_times, future_times, y_p,
        kernel_type=kernel_type,
        length_scale=length_scale,
        jitter=jitter,
    )

    # Sample: x_0 = μ + L_ff @ ε,  ε ~ N(0, I)
    # mu: (B, F, D),  L_ff: (F, F)
    batch_size = y_p.shape[0]
    D = y_p.shape[2]
    eps = jax.random.normal(rng, (batch_size, action_horizon, D))
    noise = mu + jnp.einsum("ij,bjd->bid", L_ff, eps)

    return noise
