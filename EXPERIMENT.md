# Experiment 1: Live Executable Geometry During Inference

## Hypothesis

A model hidden state can traverse Glamin executable geometry and receive a
residual result that changes inference behavior without modifying model weights
or restarting the process.

## First Milestone

Before integrating a model, prove the native runtime boundary:

1. Start Glamin from C++ through a stable C ABI.
2. Load and pin geometry generation A.
3. Traverse it with a fixed input vector and record the result.
4. Mount geometry generation B while the process remains live.
5. Verify existing requests remain pinned to A and new requests use B.
6. Restore A and verify deterministic rollback.

## Intended Runtime Equation

```text
h' = h + gate * glamin(h, trace, generation)
```

## Boundary Status

The live native boundary passes this milestone with real Glamin flat indexes
behind immutable generations. The model boundary also now runs inside inference:
a pinned llama.cpp evaluation callback captures Qwen3's `l_out-18` hidden state,
projects it into a 16-dimensional Glamin space, resolves a generation-qualified
residual, writes it into the live tensor, and lets later layers continue.

With the same loaded model and prompt, generation A produced top token ` the`,
generation B produced ` for`, and rollback to A reproduced the original logits
exactly.

## Deferred Beyond This Boundary

- Adapter training
- Token-by-token memory access
- Online memory writes
- External actions and capabilities
- Durable Glamin traces

## Current Persistence Checkpoint

The model hook now mounts `gx1-hook-v1` artifacts that checksum-bind the model
and tensor contract, projection, Glamin compiler files, residual labels, and
residual vectors. Failed validation cannot publish or activate a partial hook
generation. Artifact signatures and durable deployment metadata remain future
control-plane work.

## Current Associative-Memory Checkpoint

A 24-row Glamin generation now stores eight activation-derived associations,
with three independently derived surface-form keys and view-conditioned actions
per association. At Qwen3's penultimate transformer layer, the callback searches
every prompt-token state, automatically selects the entity-bearing address, and
applies the retrieved action to the final answer position. All 16 held-out
positive prompts—including eight unseen natural questions—produce their target
as the rank-one next token. A calibrated maximum-distance gate abstains on 16
controls, four of them held out, and preserves their logits exactly. This does
not yet show broad multi-relation generalization, multi-token recall, or a
learned gate. The artifact hash-binds its all-token selection policy, and the
model-agnostic writer derives key positions without entity annotations.
