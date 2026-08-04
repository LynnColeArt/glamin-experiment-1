# Gate-Local Authorization Invariance

## Status

Protocol frozen before the first model evaluation. Results are intentionally
absent until the development preflight and one-shot frozen stages run.

Development preflight amendment, 2026-08-04: the first execution stopped
before `relation_prototype_evaluation` while constructing the variance action
baseline. Its expanded positive and negative neighborhoods overlap (maximum
validation distance `0.104436`, minimum negative distance `0.097498`), so the
strict builder correctly refused to construct a supposedly separating gate.
To retain this negative baseline without changing its inputs or threshold
formula, the builder gained an explicit permissive-calibration option. The
default remains strict, the variance comparison alone opts out, and the new
authorization-signal candidate remains strict. No frozen prompt was evaluated
or changed.

## Motivation

Association-signal entity addressing identified 18/18 positive entities and
rejected 6/6 unknown entities, while nine prompts still failed at relation or
action-context gates. This experiment changes those representations one at a
time before composing them.

The model weights, association-signal entity memory, exact tuple ledger,
two-stage tensor locations, target-state ledger, target-state interpolation,
and all earlier frozen prompts remain unchanged.

## New Representation Primitives

The relation candidate retains the association-signal projection but collapses
its selected construction views into one normalized centroid per relation. It
is compared with the existing nearest-view keys using identical construction,
development, and negative activations.

The action candidate adds an `authorization_signal` projection. For each
reviewed tuple and activation coordinate, it compares the mean retrieval state
with the mean tuple-scoped wrong-intent state and scores between-class signal
against within-class variation. The highest-scoring coordinates form the
projection. It is compared with variance projection using identical positive
views, negatives, validation scope, key construction, and gate interpolation.

No gradient-based training or model-weight update occurs.

## Development Preflight

The relation stage has two development forms per registered tuple (12 total):

- `Retrieve memory for ENTITY. Requested property category: REL.`
- `Stored object: ENTITY\nPredicate label: REL\nValue:`

The centroid candidate must identify and accept the correct relation on 12/12.

The authorization stage has two development forms per tuple (12 total):

- `Memory retrieval authorized: return REL for ENTITY.`
- `READ fact(entity=ENTITY, relation=REL) =>`

The authorization-signal candidate must reach and pass the action gate on
12/12. The existing 12 tuple-scoped metaphor and spelling negatives must reach
that gate, reject, and leave logits exactly unchanged. Development may correct
construction defects without reading or changing a frozen set.

## Frozen Relation Stage

Two new relation forms per registered tuple produce 12 positives:

- `For ENTITY, recall the value whose property type is REL.`
- `record(subject=ENTITY, predicate=REL) -> object:`

The existing negative relation names `temperature` and `age` appear in both
forms, producing four negative controls. Success requires 12/12 correct
centroid relation matches and 4/4 relation-gate rejections. Nearest-view results
are recorded as a baseline but are not a pass condition.

## Frozen Authorization Stage

Two new positive forms per registered tuple produce 12 positives:

- `Execute stored-value recall for ENTITY / REL.`
- `memory.get(ENTITY, REL) ->`

Two new wrong-intent forms per tuple produce 12 negatives: label summarization
and alphabetization with explicit instructions not to retrieve. Success
requires 12/12 authorization-signal acceptances, 12/12 rank-one target-state
results, and 12/12 exact no-op negatives that reach the action gate. Variance
results are recorded as a baseline.

## Frozen Composition Stage

The successful candidates, if any, are composed on a third untouched split:

- `Consult the fact ledger; subject ENTITY, property REL.`
- `resolve-memory{subject:ENTITY,predicate:REL}=>`

Success requires all of the following:

- 12/12 correct routes and rank-one stored targets;
- 6/6 fresh formatting-only wrong intents rejected at the action gate;
- 4/4 prompts for the two known missing tuples left as exact no-ops;
- 4/4 prompts for new entity names `Vega` and `Altair` rejected by the entity
  gate; and
- 4/4 prompts for new relation names `weight` and `origin` rejected by the
  relation gate.

Each local stage and the composition stage receives an independent verdict. No
frozen prompt, representation, threshold, control, or criterion may change
after the first frozen run. Failure is retained as evidence and must motivate a
separate experiment.
