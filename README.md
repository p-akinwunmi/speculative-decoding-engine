# speculative-decoding-engine

Exact speculative decoding verification, tree-structured causal masking, and paged KV-cache rollback implemented in pure, XLA-compilable JAX (`jax.jit`, `jax.vmap`, `jax.lax.scan`).

## Overview
Autoregressive LLM decoding is memory-bandwidth bound because each forward pass loads full model parameters to emit a single token. This engine breaks that bottleneck by verifying $K$ draft tokens—or an entire branching tree of $M$ candidate tokens—in a single target-model forward pass while preserving exact distributional equivalence with standard autoregressive sampling.

## Architecture & Modules

```text
speculative-decoding-engine/
├── AGENTS.md                        # Shape, dtype, and XLA purity invariants
├── pyproject.toml                   # Editable package configuration
├── src/speculative_engine/
│   ├── rejection_sampling.py        # Multi-step rejection & normalized residual sampling
│   ├── tree_attention.py            # Parent-pointer ancestor closures & RoPE position IDs
│   └── kv_rollback.py               # Paged KV-pool compaction & block-table reclamation
├── tests/
│   ├── test_distributional_parity.py # Pearson Chi-Square statistical parity test (N=100k)
│   └── test_tree_kv_rollback.py      # Branch isolation & block reclamation verification
└── benchmarks/
    └── bench_speculative_speedup.py  # Acceptance length (tau) & speedup profiler
