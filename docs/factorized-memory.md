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
both of their individual factors are known. The initial probe gave each
registered tuple canonical, structured, and natural-question action contexts.
Targets and teacher prompts were required to produce one-token, rank-one
answers before a residual was admitted.

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

## Projected Action Probe

The next probe attempted to improve form generalization without tuning against
the twelve exploratory prompts. Those prompts became part of an explicit
development set, and a new evaluation set was written before the first run.

The development-only changes were:

1. add request and continuation forms to each tuple, producing 30 registered
   action views;
2. use one canonical teacher state for all five surface forms of a tuple;
3. project action states from the model width to 256 selected dimensions and
   L2-normalize them before tuple-local nearest-variant selection;
4. calibrate the action radius on one separate positive form per tuple and six
   hard negatives; and
5. add the request and continuation forms to relation calibration.

Cross-tuple action prototypes are not gate negatives. The entity–relation join
has already isolated the tuple before action selection, so the action gate is
calibrated against explicit hard negatives instead of re-solving tuple
identity.

The resulting development boundaries were:

| Space | Maximum valid distance | Nearest negative boundary |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Relation | `0.0543132` | `0.0559928` |
| Projected action | `0.0177528` | `0.0198972` |

All 30 registered views produced their target at rank one. Both absent
known-factor tuples still left every logit unchanged.

The frozen evaluation then tested two previously unseen forms per tuple:

- `Please provide the REL recorded for ENTITY.\nResponse:`
- `Recorded REL for ENTITY:`

The result was **4/12 rank-one recalls**. The first form recalled Arcturus/color,
Arcturus/material, Bellatrix/color, and Cygnus/material. Draco/color activated
the intended memory but reached rank two. Draco/material exceeded the action
radius. All six compact forms exceeded the relation radius and were exact
no-ops.

This is evidence of partial form generalization, not completion of the
milestone. Projection collapsed raw action distances into a calibrated local
space and generalized beyond registered strings, but the relation address is
still brittle to compact syntax and a single canonical action target does not
guarantee rank-one behavior for every tuple. The frozen prompts must remain an
evaluation set; subsequent work should create a new development/evaluation
split.

## Next Experiment

The next probe should:

1. derive a relation representation that is less sensitive to prompt syntax,
   using a new development corpus rather than the frozen prompts above;
2. test whether tuple-local action targets should be prototypes or a learned
   low-rank transform instead of one canonical residual per form;
3. add same-factor, wrong-intent negatives that can actually reach the action
   gate;
4. freeze another paraphrase evaluation set before running it; and
5. persist and live-swap the complete factorized generation.
