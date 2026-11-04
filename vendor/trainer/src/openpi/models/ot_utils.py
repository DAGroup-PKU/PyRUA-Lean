"""Mini-batch Optimal Transport utilities for Flow Matching noise coupling.

Implements Sinkhorn-based OT to re-pair GP noise samples with actions within
a training batch. This reduces the expected flow-matching transport cost by
ensuring each action gets a "nearby" noise sample rather than a random one.

Usage:
    noise = sample_gp_noise(rng, actions.shape, ...)      # GP noise
    noise = apply_ot_coupling(actions, noise, reg=0.1)    # OT re-pair
    # Both GP temporal structure and OT coupling are preserved.
"""

import jax
import jax.numpy as jnp


def sinkhorn(
    cost_matrix: jax.Array,
    reg: float = 0.1,
    num_iters: int = 20,
) -> jax.Array:
    """Sinkhorn algorithm for entropic-regularised optimal transport.

    Solves the regularised OT problem:
        min_P  <C, P>  -  reg * H(P)
        s.t.   P 1 = a,  Pᵀ 1 = b   (uniform marginals a=b=1/n)

    Uses the log-domain formulation for numerical stability.

    Args:
        cost_matrix: (n, n) pairwise cost matrix C[i, j].
        reg:         Entropy regularisation parameter ε > 0.
                     Larger ε → softer (more uniform) plan.
                     Smaller ε → closer to hard assignment.
        num_iters:   Number of Sinkhorn iterations.

    Returns:
        Transport plan P of shape (n, n) with uniform marginals.
    """
    n = cost_matrix.shape[0]
    log_a = jnp.full((n,), -jnp.log(n))  # log(1/n)
    log_b = jnp.full((n,), -jnp.log(n))

    # Log-kernel: M[i,j] = -C[i,j] / reg
    log_K = -cost_matrix / reg

    # Sinkhorn in log-domain: alternate updating u, v
    log_u = jnp.zeros(n)
    log_v = jnp.zeros(n)

    def sinkhorn_step(carry, _):
        log_u, log_v = carry
        # log_u  ← log_a - logsumexp(log_K + log_v, axis=1)
        log_u = log_a - jax.scipy.special.logsumexp(log_K + log_v[None, :], axis=1)
        # log_v  ← log_b - logsumexp(log_K + log_u, axis=0)
        log_v = log_b - jax.scipy.special.logsumexp(log_K + log_u[:, None], axis=0)
        return (log_u, log_v), None

    (log_u, log_v), _ = jax.lax.scan(sinkhorn_step, (log_u, log_v), None, length=num_iters)

    # Transport plan: P[i,j] = exp(log_u[i] + log_K[i,j] + log_v[j])
    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = jnp.exp(log_P)
    return P


def compute_cost_matrix(actions: jax.Array, noise: jax.Array) -> jax.Array:
    """Compute pairwise squared L2 cost between actions and noise samples.

    Cost encourages pairing action aᵢ with a noise nⱼ that lies close to it
    in action space, minimising the expected ODE path length.

    Args:
        actions: (batch, action_horizon, action_dim) ground-truth actions.
        noise:   (batch, action_horizon, action_dim) GP noise samples.

    Returns:
        Cost matrix of shape (batch, batch).
        C[i, j] = mean squared distance between aᵢ and nⱼ (averaged over T, D).
    """
    # Flatten (T, D) → vector per sample for distance computation
    a_flat = actions.reshape(actions.shape[0], -1)   # (B, T*D)
    n_flat = noise.reshape(noise.shape[0], -1)        # (B, T*D)

    # Squared L2: C[i,j] = ||a_i - n_j||² / (T*D)  (normalised)
    diff = a_flat[:, None, :] - n_flat[None, :, :]   # (B, B, T*D)
    cost = jnp.mean(diff ** 2, axis=-1)               # (B, B)
    return cost


def ot_reindex(noise: jax.Array, plan: jax.Array) -> jax.Array:
    """Re-pair noise samples according to a (soft) transport plan.

    Performs a soft assignment: the re-paired noise for sample i is the
    weighted sum of all noise samples according to row i of the plan.

    For a near-hard plan (small reg), this is approximately a permutation.
    For a softer plan, it produces an interpolated noise that may be slightly
    smoother but still GP-structured.

    Args:
        noise: (batch, action_horizon, action_dim) original GP noise.
        plan:  (batch, batch) transport plan; rows sum to 1/batch.

    Returns:
        Re-indexed noise of shape (batch, action_horizon, action_dim).
    """
    B = noise.shape[0]
    # Normalise plan rows to sum to 1 (instead of 1/B)
    weights = plan * B                # (B, B), each row sums to 1
    # Weighted sum: reindexed[i] = Σⱼ weights[i,j] * noise[j]
    reindexed = jnp.einsum("ij,jkl->ikl", weights, noise)
    return reindexed


def apply_ot_coupling(
    actions: jax.Array,
    noise: jax.Array,
    reg: float = 0.1,
    num_iters: int = 20,
) -> jax.Array:
    """Apply mini-batch OT coupling: re-pair GP noise with actions.

    Full pipeline:
      1. Compute pairwise cost matrix C[i,j] = ||aᵢ - nⱼ||²
      2. Run Sinkhorn to get transport plan P
      3. Re-index noise according to P

    The GP temporal structure within each noise sample is preserved because
    OT only permutes (or softly mixes) samples across the batch dimension.

    Args:
        actions:   (batch, action_horizon, action_dim) ground-truth actions.
        noise:     (batch, action_horizon, action_dim) GP-sampled noise.
        reg:       Sinkhorn regularisation (default 0.1).
                   Smaller → harder assignment, closer to permutation.
                   Larger  → softer/smoother, more stable gradient.
        num_iters: Sinkhorn iterations (20 is typically sufficient).

    Returns:
        OT-reindexed noise, same shape as input `noise`.
        Each action aᵢ is now paired with a noise sample that is close to
        it in action space, reducing the expected FM transport cost.
    """
    cost = compute_cost_matrix(actions, noise)   # (B, B)
    plan = sinkhorn(cost, reg=reg, num_iters=num_iters)  # (B, B)
    return ot_reindex(noise, plan)               # (B, T, D)
