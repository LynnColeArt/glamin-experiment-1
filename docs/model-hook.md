# Fixed Model Hidden-State Hook

## Implemented Boundary

The model hook runs synchronously inside a pinned `llama.cpp` graph evaluation.
It targets one named transformer-layer tensor. Addressing can use the last
token, an explicitly selected earlier token, or an automatic scan of every
token row. Scan mode chooses the candidate with the smallest Glamin distance;
residual delivery mutates only the last token in the current batch:

```text
q  = normalize(project_in(h_address))
id = glamin_search(q, pinned_generation, k = 1)
if distance(id) <= maximum_distance:
    r  = residual_ledger[pinned_generation, id]
    h_action' = h_action + gate * r
else:
    h_action' = h_action
```

`llama.cpp` calls the adapter immediately after the target tensor is computed.
The scheduler has not evaluated later nodes at that point. The adapter reads the
float32 address and action rows, traverses Glamin, writes the validated residual
back to the action row in the same tensor, and returns control so the remaining
transformer layers consume the modified state.

No llama.cpp source modification is required. The integration uses its public
evaluation callback and backend tensor APIs.

## Pinned Demonstration Contract

The first real-model run used:

| Field | Value |
| --- | --- |
| llama.cpp revision | `ecd99d6a9acbc436bad085783bcd5d0b9ae9e9e9` |
| Model | Qwen3 4B, Q4_K_M GGUF |
| Model SHA-256 | `fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3` |
| Hidden width | 2560 |
| Transformer layers | 36 |
| Hook tensor | `l_out-18` |
| Glamin query width | 16 |
| Query normalization | L2 |
| Retrieval | Flat L2, one neighbor |
| Injection scope | Last token, once per request |
| Device | CPU |

The demonstration keeps one model object loaded, creates a request context for
each trial, and performs A -> B -> A geometry activation in the same process.
For the fixed prompt, generation A produced top token ` the`, generation B
produced ` for`, and rollback reproduced A exactly. The measured maximum logit
deltas were `14.3216` for A-to-B and `0` for A-to-rollback.

This is an integration probe, not a trained memory. The sparse input projection
and sinusoidal generation-B residual are deterministic experimental fixtures.

## Persistent Address and Payload Contract

Glamin currently returns the selected row label and distance. A host-side
`ResidualPayloadLedger` maps `(generation, label)` to the full hidden-width
residual. Qualifying the address by generation prevents the same row number in
two immutable generations from silently resolving to the same payload.

This separation is deliberate: Glamin owns executable addressing and live
selection, while a ledger owns exact payloads. The `gx1-hook-v1` artifact now
persists and hash-binds that ledger with its projection, model SHA-256,
llama.cpp revision, architecture, target tensor, address-selection policy,
Glamin space, and geometry files. The complete format and mount protocol are
specified in the
[persistent hook artifact](hook-artifact.md) document.

Inference receives a `PinnedHookGeneration`, not separately selected parts.
That object keeps the Glamin generation, projection, payload ledger, target
tensor, and artifact identity aligned for the lifetime of the request.

## Failure Semantics

The callback aborts the current decode and records a diagnostic if any of these
conditions fail:

- The target tensor is missing, non-float32, noncontiguous, or has the wrong
  hidden width.
- The projection and Glamin space dimensions differ.
- Glamin traversal fails or does not return exactly one result.
- The selected generation-qualified payload is missing or has the wrong width.
- Projection or residual injection produces a non-finite value.
- A manifest, component checksum, tensor shape, or active-model contract does
  not match before mount.

The hidden vector is updated only after retrieval and payload validation
complete, so lookup failures do not partially apply a residual.
Distance abstention also leaves the vector unchanged and reports that no memory
was applied.

## Deliberate Limitations

- CPU-only and Qwen3-only in the demonstrated contract.
- One synchronous tensor hook; automatic selection performs one Glamin search
  per prompt-token row and mutates only the final prompt token, not token-level
  generation.
- Fixed projection and gate, not learned adapters.
- Nearest-neighbor selection, not yet Glamin trace or mini-app execution.
- Request contexts are recreated between trials; model weights remain loaded.
- No durable audit trace; the current latency measurement is a single-machine
  experiment rather than a benchmark suite.
- Artifacts are checksummed but not yet signed or authenticated.

The activation-derived adapter, automatic address scan, and held-out natural
questions are recorded in the
[associative-memory experiment](activation-memory.md). The next useful boundary
is broader multi-relation semantic generalization, followed by batched scanning,
a learned dense projection, and a contextual gate.
