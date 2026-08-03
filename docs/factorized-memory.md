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

## Association-Signal Relation Probe

A third probe created another development/evaluation split. It added an
`association_signal` projection selector that prefers dimensions whose relation
identity is stable across construction views while suppressing dimensions
dominated by within-relation form variance. Two new syntax families expanded
relation development, and one field-oriented form expanded each tuple's action
development. The resulting 36 registered action views all produced rank-one
targets.

This probe also added six same-entity, same-relation wrong-intent controls of
the form “write a metaphor about these words; do not retrieve a stored value.”
All six passed both factor gates and the exact tuple join, reached the action
gate, and abstained with exactly unchanged logits. The two absent known-factor
tuples also remained exact no-ops.

The development boundaries were:

| Space | Maximum valid distance | Nearest negative boundary |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Association-signal relation | `1.09326` | `1.28097` |
| Projected action | `0.0172099` | `0.0186327` |

The frozen set then tested possessive and symbolic forms:

- `From memory, give ENTITY's REL.\nValue:`
- `ENTITY :: REL :: stored value =`

Relation retrieval accepted the intended relation on **12/12** unseen prompts.
Entity retrieval accepted 9/12; the three failures were symbolic Arcturus and
Cygnus forms. Every one of the nine prompts reaching the exact tuple join was
rejected by the action gate, producing **0/12 end-to-end recalls** and exact
no-ops throughout.

The distances locate the next boundary more precisely. Frozen retrieval forms
were `0.038–0.075` from their nearest tuple-local action prototype, while the
same-factor wrong-intent controls were farther away at `0.117–0.150`. The
`0.0172` action radius was instead constrained by legacy calibration controls
outside the negative class demonstrated to reach the action gate. This is
post-evaluation evidence of a potentially usable intent gap, not authorization
to retune against this frozen set.

## Runtime-Topology-Aware Action Probe

The fourth probe implemented the gate-local calibration suggested above. The
builder now accepts association-scoped validation and calibration negatives.
Because runtime searches action variants only after the exact entity–relation
join, each action positive and wrong-intent control is compared only with the
six variants of its reachable tuple. Entity and relation calibration remains
global. Legacy controls that cannot reach the action gate were removed from
that gate's calibration corpus but remain tested at their actual factor or
tuple boundary.

The new development corpus added two unseen lookup forms and two tuple-matched
wrong intents for every stored tuple. The resulting boundaries were:

| Space | Maximum valid distance | Nearest negative boundary |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Association-signal relation | `1.09326` | `1.28097` |
| Tuple-scoped projected action | `0.0823871` | `0.0974977` |

All 36 registered actions still produced rank-one targets. All 12 development
lookup forms reached the intended action; 7/12 produced the target at rank one
with the canonical residual strength of `1.0`. A development-only strength
sweep improved the aggregate to 11/12 but made the remaining Draco/color miss
worse, showing that the failure was not merely a weak intervention. The frozen
run therefore retained the unscaled residual.

All 12 tuple-matched metaphor and spelling controls passed both factor gates
and the exact tuple join, then failed the action gate at distances from
`0.0974977` through `0.164`. Every control left every logit exactly unchanged.
The two absent known-factor tuples also remained exact no-ops.

Before its first run, the next evaluation set was frozen with two new forms:

- `Consult stored memory for ENTITY; requested property: REL.\nAnswer:`
- `What value does memory associate with ENTITY under REL?\nValue:`

The intended entity, relation, tuple, and action were selected on **12/12**
prompts. Action distances ranged from `0.0121598` to `0.0546432`, safely below
the independently calibrated radius. The residual produced the stored target
at rank one on **9/12** prompts: all six `consult` forms and three of six
`associate` forms. The misses were Bellatrix/color at rank 7, Draco/color at
rank 70, and Draco/material at rank 7. No mechanism or threshold was changed
after observing this frozen result.

This establishes generalization of the complete authorization path on this
small split: broad factor evidence can resolve a specific tuple, and a live
action gate can distinguish novel retrieval language from tuple-matched wrong
intent. It does not establish a form-invariant memory action. The stored
teacher-minus-query residual remains view-conditioned; a nearby action address
can authorize the right tuple while its selected residual fails to transport
the new hidden state to the intended answer state.

## Two-Stage Target-State Probe

The fifth probe separated authorization from action in both code and model
depth. A new callback observes `l_out-34`, performs the entity, relation, exact
tuple, and action-context checks, and deliberately leaves that tensor
unchanged. If and only if all checks pass, the callback later replaces the
final row of `l_out-35` with the tuple's canonical teacher state. The target
state is stored in a separate tuple ledger, and a gate of `1.0` performs exact
interpolation to that state.

The comparison kept the existing early residual path as its baseline. Before
any run, two development and two evaluation forms were written for every
tuple. The development forms expanded factor and action calibration; the
evaluation forms were then frozen:

- `Retrieve from long-term memory: ENTITY; requested attribute REL.\nAnswer:`
- `Archive[ENTITY] / REL / value =>`

The expanded development boundary changed only the relation radius:

| Space | Maximum valid distance | Nearest negative boundary |
| --- | ---: | ---: |
| Entity | `0.0510935` | `0.0694399` |
| Association-signal relation | `1.12635` | `1.28097` |
| Tuple-scoped projected action | `0.0823871` | `0.0974977` |

On the 12 development prompts, both paths authorized 12/12. The residual
baseline produced 9/12 rank-one targets; late target-state transfer produced
12/12. All 12 tuple-matched wrong intents and both absent tuples remained exact
no-ops through the two-tensor callback.

The preregistered frozen success criterion required 12/12 correct routes and
12/12 rank-one target-state actions. It **failed**:

| Frozen measurement | Result |
| --- | ---: |
| Correct authorization routes | 10/12 |
| Early residual rank-one recall | 6/12 |
| Late target-state rank-one recall | 10/12 |
| Late target-state recall, conditional on authorization | 10/10 |

All six long-term-memory forms routed, and both mechanisms reached rank one.
Four compact archive forms routed; the residual missed all four while the late
target state reached rank one on all four. The two Arcturus archive prompts
failed the entity gate at distance `0.0617394`, beyond its independently
calibrated `0.0510935` radius. Their relation evidence was accepted, no action
was authorized, and both paths left every logit unchanged.

For every authorized frozen prompt, target-state logits reproduced the
canonical teacher logits within a maximum absolute difference of
`3.8147e-06`. No prompt, threshold, representation, or mechanism was changed
after the first frozen run.

This provides strong conditional evidence that action location was the payload
boundary: once the tuple and intent are authorized, a later target state is
substantially more form-invariant than an early view-conditioned residual. It
does not meet the end-to-end criterion because compact entity addressing still
fails for one entity. Full-state replacement is also an aggressive one-token
intervention; multi-token continuation quality and general-capability effects
remain unmeasured.

## Next Experiment

The failed archive prompts must remain frozen. A new split should test an
association-signal or otherwise form-stable entity representation against
compact syntax while retaining the two-stage target action and every existing
negative control. Once end-to-end routing generalizes independently, the
complete two-stage factorized contract should be persisted and live-swapped as
one generation.
