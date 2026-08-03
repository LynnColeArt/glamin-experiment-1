# Factorized Entity–Relation Memory

## Question

Can Glamin authorize a specific memory action from separately retrieved entity
and relation evidence, rather than requiring one vector to encode the complete
tuple?

The preceding [multi-relation probe](multi-relation-probe.md) showed that a
single nearest-vector address could not safely separate valid paraphrases from
unknown entity–relation tuples. This experiment replaces that address with a
bounded three-stage join inside the llama.cpp inference callback:

```text
all token states ──> entity Glamin space ──> entity evidence + distance gate
                 └─> relation Glamin space -> relation evidence + distance gate

(entity, relation) ──> exact reviewed tuple ledger
                    └─> nearest registered action context + distance gate
                        └─> residual injection at the final token row
```

Entity and relation searches may select different token rows. Passing both
factor gates does not confer action authority: the exact tuple must exist, and
the current action-position state must also be close to a registered action
prototype. Failure at any stage leaves the tensor unchanged.

## Frozen-Model Probe

The frozen Qwen3 4B Q4_K_M model used four entities and two relations. Six of
the eight possible tuples were registered:

| Entity | Color | Material |
| --- | --- | --- |
| Arcturus | ` blue` | ` cedar` |
| Bellatrix | ` amber` | absent |
| Cygnus | absent | ` copper` |
| Draco | ` violet` | ` maple` |

`Bellatrix/material` and `Cygnus/color` are deliberately absent even though
both of their individual factors are known. Each registered tuple has
canonical, structured, and natural-question action contexts. Targets and
teacher prompts were required to produce one-token, rank-one answers before a
residual was admitted.

The independently calibrated factor boundaries were:

| Factor | Maximum valid distance | Nearest negative boundary |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Relation | `0.0376144` | `0.0559928` |

## Result

- All 18 registered action contexts selected the intended entity, relation,
  and contextual residual and produced the target token at rank one.
- Both absent known-factor tuples passed the two factor gates but failed the
  exact tuple join. Neither changed any logit.
- Twelve fresh request/continuation paraphrases recalled zero answers. Every
  prompt nevertheless remained an exact no-op: some failed the relation gate;
  the others failed the action-context gate by a wide margin.

This proves a stronger safety property than the monolithic probe: broad factor
evidence and specific action authority are now distinct decisions. The system
can recognize both pieces of a tuple without inventing a fact or applying an
ill-fitting residual.

It does not yet prove open-ended linguistic recall. Action residuals remain
strongly view-conditioned, and the current action prototype uses raw full-width
hidden-state distance. The `1.0` action radius is intentionally strict; exact
registered contexts have distance zero, while exploratory contexts were tens
of thousands of squared-distance units away.

## Implementation Boundary

`FactorizedLayerMemoryHook` owns two immutable Glamin generation pins, two
independent projection and distance contracts, a logical row-to-factor ledger,
an exact tuple ledger, contextual action variants, and a third distance gate.
`LlamaFactorizedGlaminHook` executes that join synchronously on the configured
layer tensor and writes only the final token row.

The contextual variant set is currently host-resident. Moving it into a
persisted, tuple-scoped Glamin space is the next executable-geometry step. That
artifact must bind both factor spaces, row ledgers, tuple membership, action
prototypes, residuals, all three gates, and the model/layer contract as one
atomic generation.

## Next Experiment

The next probe should improve form generalization without tuning against the
twelve exploratory prompts:

1. derive a form-invariant relation projection from a separate development set;
2. normalize or project the action-context prototypes before local selection;
3. calibrate the action gate on positive variants and hard context negatives;
4. freeze a new paraphrase evaluation set before running it;
5. persist and live-swap the complete factorized generation.
