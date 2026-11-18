import time
import jax
import jax.numpy as jnp
from speculative_engine.rejection_sampling import verify_and_sample_chain


def theoretical_speedup(alpha: float, k: int, cost_ratio_c: float) -> float:
    """
    Leviathan et al. analytical speedup:
    E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)
    Speedup   = E[tokens] / (1 + c * K)
    """
    expected_tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
    return expected_tokens / (1.0 + cost_ratio_c * k)


def run_benchmark():
    vocab_size = 32_000
    n_steps = 2_000
    cost_ratio_c = 0.08  # Draft model is ~12.5x cheaper per pass than target

    print(
        f"{'K':>3} | {'Target Alpha':>12} | {'Empirical E[tok]':>16} | "
        f"{'Analytical E[tok]':>17} | {'Net Speedup':>11} | {'Verify us/step':>14}"
    )
    print("-" * 85)

    for k in (2, 4, 6, 8):
        for alpha in (0.60, 0.75, 0.90):
            target_probs = jnp.full(
                (k + 1, vocab_size),
                (1.0 - alpha) / (vocab_size - 1),
                dtype=jnp.float32,
            ).at[:, 0].set(alpha)

            draft_probs = jnp.zeros((k, vocab_size), dtype=jnp.float32).at[:, 0].set(1.0)
            draft_tokens = jnp.zeros((k,), dtype=jnp.int32)
            keys = jax.random.split(jax.random.PRNGKey(0), n_steps)

            # Warmup XLA compilation
            _ = jax.vmap(
                lambda key: verify_and_sample_chain(
                    key, draft_tokens, draft_probs, target_probs, vocab_size
                )
            )(keys[:10])

            t0 = time.perf_counter()
            _, valid_counts = jax.block_until_ready(
                jax.vmap(
                    lambda key: verify_and_sample_chain(
                        key, draft_tokens, draft_probs, target_probs, vocab_size
                    )
                )(keys)
            )
            elapsed_us = ((time.perf_counter() - t0) / n_steps) * 1e6

            emp_tokens = float(jnp.mean(valid_counts))
            ana_tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
            speedup = theoretical_speedup(alpha, k, cost_ratio_c)

            print(
                f"{k:>3} | {alpha:>12.2f} | {emp_tokens:>16.3f} | "
                f"{ana_tokens:>17.3f} | {speedup:>10.2f}x | {elapsed_us:>14.2f}"
            )


if __name__ == "__main__":
    run_benchmark()