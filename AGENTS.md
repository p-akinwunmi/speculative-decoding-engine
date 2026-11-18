# AGENTS.md — speculative-decoding-engine

## Rules
- Keep all functions in `src/speculative_engine/` pure and compatible with `jax.jit`.
- Use `jnp.int32` for token IDs, `parent_indices`, `block_tables`, and `context_lens`.
- Use `jnp.float32` for probabilities and KV tensors.
- Use `-1` for empty/padded slots in `block_tables` and `emitted_tokens`.

## Commands
- Run tests: `pytest -v`
- Run benchmark: `python benchmarks/bench_speculative_speedup.py`