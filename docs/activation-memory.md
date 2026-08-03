# Activation-Derived Associative Memory

## Question

Can a small set of actual associations be addressed from live model state,
inject the right specific answer, and abstain on unrelated prompts?

The activation-memory experiment tests that boundary without an autograd-based
training stack. It derives memory directly from grouped model runs while
separating the state used to address memory from the state changed by memory:

```text
construction candidates = hidden(every token in each construction prompt)
query state   = hidden(final answer position in query prompt)
teacher state = hidden(final answer position with the association in context)
key per view  = argmax(distance to nearest other-association/control state)
address candidates = hidden(every token in an inference prompt)
address       = nearest memory over all projected candidate states
view action   = teacher state - query state
```

Projection dimensions are selected by variance across every token state in the
construction prompts and calibration negatives. For each construction view,
the writer selects the token state with the greatest separation from all token
states belonging to other associations and controls. Views belonging to the
same association are linked logically but are not forced into one activation
cluster: lookup syntax and natural language may occupy different local
manifolds. Each key retains its view-conditioned teacher-minus-query action.
No entity string or token position is given to key derivation.

The complete generation is persisted as one `gx1-hook-v1` artifact and searched
through Glamin. The artifact now hash-binds `all_token_rows` as its address
selection policy; constructing `LlamaGlaminHook` from the pinned generation
activates scanning without an experiment-side flag.

At inference time the Qwen3 `l_out-34` tensor still contains every prompt row.
The callback projects every row, asks Glamin for each row's nearest memory,
selects the globally nearest result, and applies its residual only to the final
answer-position row. No entity marker or token index is supplied during
inference. Thus the system can discover *where the memory request is represented*
and act at another location in the same live tensor.

## Abstention

Nearest-neighbor search always returns something, which is unsafe for sparse
memory. The hook artifact therefore includes a maximum squared-L2 distance.
The residual is applied only when the nearest key falls inside that radius.
Otherwise the hook reports the nearest address and distance but applies a zero
gate and leaves the hidden state unchanged.

The experimental radius is calibrated without using the final test queries or
held-out negatives. Sixteen validation prompts cover lookup syntax and natural
questions. The gate is placed 75 percent of the way from the largest accepted
validation distance toward the nearest of 12 calibration negatives or
cross-association keys, retaining 25 percent of the observed negative gap as a
safety margin. This remains a small synthetic calibration set, not a production
thresholding method.

## Qwen3 Result

The passing scaled run used Qwen3 4B Q4_K_M, a 256-dimensional variance-selected
projection, and the penultimate transformer tensor `l_out-34`.

| Split | Count | Result |
| --- | ---: | --- |
| Construction: three views for each of eight associations | 24 | 24/24 target rank one |
| Validation: lookup and natural-question forms | 16 | 16/16 target rank one |
| Held-out positive: unseen retrieval and natural-question forms | 16 | 16/16 target rank one |
| Negative controls | 12 | 12/12 abstained; logits exactly unchanged |
| Held-out negative controls | 4 | 4/4 abstained; logits exactly unchanged |

The eight synthetic associations map Arcturus, Bellatrix, Cygnus, Draco,
Eridanus, Fornax, Gemini, and Hydra to distinct one-token values. The model
weights remain frozen. Construction selected 24 entity-bearing keys, took about
`694 ms`, and produced a `2,893,875` byte artifact. Average memory-enabled
inference across the measured run was about `200 ms` on the test machine.

The largest validation distance was `0.0183476`, the nearest calibration
negative began at `0.0456469`, and the resulting gate was `0.0388221`. The
hardest held-out positive was the Gemini natural question at `0.0371393`, still
inside the independently calibrated gate. The nearest held-out negative was
`0.0505747`, outside it. Negative prompts still produced nearest candidates,
but the distance gate prevented every action.

The address position and layer both mattered. Addressing natural questions
from their final `Answer:` state selected the correct associations, but an
unknown-name question was closer than the positive validation questions because
that state was dominated by response format. Addressing from the entity token
restored a safe margin. `l_out-35` exposes only the output-pruned final row, so
the experiment moved one layer earlier to `l_out-34`, which retains the entity
row while still delivering the residual reliably through the final block.

## What This Establishes

- An all-token nearest search can discover the entity-bearing state and address
  a specific persisted memory across lookup syntax and unseen natural questions.
- Grouped association labels plus negatives are sufficient for this writer to
  derive relevant token positions across three construction views without
  entity annotations.
- Surface forms can occupy separate activation neighborhoods while remaining
  linked to the same logical association and a locally appropriate action.
- The mounted artifact, rather than the caller, can choose all-token scanning.
- Addressing and action can occur at different token positions inside one live
  transformer tensor.
- The selected residual can make the model emit the associated next token.
- A distance-gated hook can abstain and preserve unrelated inference exactly.
- Memory effectiveness depends on where its state is injected, even when
  retrieval itself is correct.

## What It Does Not Establish

- Broad semantic generalization across relations, domains, or unconstrained
  dialogue. This test still has one synthetic relation and eight associations.
- Multi-token factual answers or continued generation quality.
- A learned dense projection, output adapter, or contextual gate.
- Construction without known association grouping.
- Robust threshold calibration beyond the small 12-negative calibration set.
- Large-memory scaling or resistance to key collisions.

The next experiment should add multiple relations, batch the all-token searches,
and measure retrieval, abstention, answer quality, and latency as neighborhood
density and linguistic freedom increase.
