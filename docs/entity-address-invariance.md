# Entity-Address Invariance Probe

## Status

Protocol frozen before the first model evaluation. Results are intentionally
absent from this document until that run occurs.

## Motivation

The two-stage target-state probe reproduced the intended target on every
authorized frozen prompt, but two compact Arcturus prompts failed the entity
gate. This probe tests whether the entity projection, rather than the later
action, is the remaining end-to-end boundary.

The previous two-stage evaluation prompts remain immutable. They are a known
regression set, not development or independent evaluation evidence.

## Controlled Comparison

Two entity memories are built from exactly the same construction views,
development views, calibration negatives, labels, and query width:

1. the existing maximum-variance coordinate projection; and
2. an association-signal projection that favors coordinates separating entity
   groups while suppressing within-entity surface-form variation.

Both candidates use the same association-signal relation memory, exact tuple
ledger, tuple-local action memory, action gate, target-state ledger, tensor
locations, and full-state interpolation. Only the entity projection strategy
differs.

## Development Split

Each of the six registered tuples has three development forms, for 18 prompts:

- prose: `From retained memory, return the REL registered for object <ENTITY>.`
- record: `Object record { name: "ENTITY", field: "REL" }`
- path: `memory/object/ENTITY/attribute/REL/value=`

These forms may calibrate the candidate entity, relation, and tuple-local action
gates. The association-signal system must route all 18 and produce the stored
target at rank one before the frozen split is run. The existing 12 wrong-intent
controls and two missing-tuple controls must remain exact no-ops.

## Frozen Evaluation Split

Each registered tuple has three previously unused forms, for 18 positive
prompts:

- question: `Which memorized REL is attached to the object named ENTITY?`
- card: `Lookup card [object=ENTITY][attribute=REL]`
- URI: `vault://records/ENTITY?field=REL#value=`

Six additional controls apply the same three forms to the known-negative entity
names `Rigel` and `Sirius`. Each must fail the entity gate and leave logits
exactly unchanged.

The frozen success criterion is all of the following:

- 18/18 association-signal prompts route to the correct entity, relation, and
  reviewed tuple;
- 18/18 produce the stored target as the rank-one next token through the later
  target-state action; and
- 6/6 unknown-entity controls are exact no-ops rejected by the entity gate.

Variance-projection results are recorded on the same prompts as a controlled
baseline but are not part of the pass condition. Only after the new frozen run
is complete is the association-signal system measured on the earlier two-stage
prompts; that measurement is a known regression result, not fresh evidence.

No prompt, representation, threshold, negative set, or success criterion may
change after the first frozen run. A failure is recorded as a failure and must
motivate a separate experiment.
