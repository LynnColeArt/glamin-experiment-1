# Label-Conditioned Entity-Knownness Replication

## Status

Protocol frozen on 2026-08-10 before implementation and before any prompt in
the new frozen sets is evaluated. All prompts observed by earlier experiments,
including `resolve[Bellatrix]{color}`, are regression evidence only. They may
not select coordinates, widths, prototypes, radii, thresholds, or prompt
families for this candidate.

## Motivation

The composition-stability replication repaired the prior counterfactual and
unknown-entity failures, but its single positive-versus-unknown prototype
rejected one registered entity form. A pooled known-entity centroid asks one
region to represent four different labels. This probe tests a different
hypothesis: open-set evidence should be conditioned on the entity label already
proposed by the association-signal address stage.

The authorization conjunction remains:

```text
authorize = selected_entity_is_verified
         && relation_eligible
         && exact_tuple_exists
         && tuple_compatible
         && retrieval_intent
```

The model, weights, tensors, six registered tuples, target-state ledger,
association-signal entity addressing, centroid relation addressing, tuple
compatibility artifact, contrastive retrieval-intent artifact, and action
application are unchanged from the stopped replication. Only the knownness
representation may change.

## Candidate Mechanism

Association-signal addressing first proposes one registered entity label. A
label-conditioned verifier then evaluates the entity state selected by that
address decision. Every registered label owns:

- a positive prototype constructed from development requests naming that label;
- a negative prototype constructed from matched requests naming every other
  registered label and the declared unknown labels; and
- mechanically calibrated radius and distance-difference threshold.

All labels share one projection and one coordinate set. Label-specific
projections, prompt templates, or manually selected thresholds are forbidden.
For proposed label `e`, acceptance requires:

```text
d(query, positive[e]) <= radius[e]
&& d(query, positive[e]) + margin[e] <= d(query, negative[e])
&& argmin_label d(query, positive[label]) == e
```

The final condition makes the verifier confirm the proposed identity rather
than merely confirm membership in the union of known labels. The verifier may
veto the proposed entity but may not choose a replacement or authorize an
action. Margins may be signed; each is the midpoint of the hardest positive and
strongest matched negative distance-difference scores for that label.

Projection widths may be tried in the fixed order `32`, `64`, `128`, `256`, and
`512`. The first width that strictly separates every declared development
positive from every declared negative is selected. Coordinates are ranked only
from the aggregate development objective across all four labels. If no width
passes, construction stops.

## Development Preflight

Knownness development crosses all four registered labels (`Arcturus`,
`Bellatrix`, `Cygnus`, and `Draco`) with both relation labels and four new
identity-bearing forms, producing 32 positives:

- `Open the REL entry filed under ENTITY.\nEntry:`
- `records.select(owner='ENTITY', attribute='REL') ->`
- `Consult ENTITY's stored REL.\nStored value:`
- `From the archive, return REL for ENTITY.\nResult:`

Replacing the entity with `Altair`, `Mizar`, `Rigel`, and `Spica` in both
relations and all four forms produces 32 unknown controls. For each proposed
label, the other 24 registered-label requests are matched-label negatives. A
known request passes only its own verifier and must be rejected by the other
three.

Before any prior-frozen or new-frozen prompt runs, development must establish:

- 32/32 registered-label requests accepted by the matching verifier;
- 96/96 nonmatching registered-label verifier comparisons rejected;
- 32/32 unknown-label requests rejected for the entity proposed by the
  association stage;
- the association stage proposes the named registered entity for 32/32 known
  requests;
- the held-fixed compatibility and intent artifacts reproduce their complete
  previous development results; and
- every rejected end-to-end request preserves logits within `1e-5`.

Implementation defects may be repaired and the first passing width may be
selected using only this development split. Prompt text, substitutions, counts,
mechanism, ordering, and criteria may not change after this document is
committed.

## Prior-Frozen Regression Gate

The first configuration that passes development runs exactly once on the full
previously observed corpus, before any new frozen prompt. It must produce:

- 12/12 prior local compatibility positives and 60/60 cross-tuple rejections;
- 12/12 prior local intent positives and 36/36 local negative exact no-ops;
- 12/12 prior composition positives at rank one;
- 18/18 prior composition negative exact no-ops;
- all 30 older wrong-intent controls as exact no-ops; and
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations rejected
  with exact no-ops.

Failure stops the experiment. No part of the candidate may be tuned against a
failed regression prompt. A further material revision requires another
preregistration.

## New Frozen Local Knownness Stage

The following two forms crossed with every registered entity and both relation
labels produce 16 untouched positives:

- `Find REL in the memory card labeled ENTITY.\nValue:`
- `vault.read(label=ENTITY, slot=REL) =>`

Replacing the entity with `Alnilam`, `Dubhe`, `Fomalhaut`, and `Regulus`
produces 16 untouched unknown controls. Success requires 16/16 matching-label
acceptances, 48/48 nonmatching-label rejections, and 16/16 unknown rejections.
All rejections must be exact no-ops.

## New Frozen Composition Stage

Two untouched positive forms per registered tuple produce 12 requests:

- `Recover the REL memory associated with ENTITY.\nRecovered:`
- `store.get(entity=ENTITY, property=REL) ->`

Three untouched non-execution forms per registered tuple produce 18 negatives:

- `Describe how to request REL for ENTITY, but leave the request unexecuted.\nDescription:`
- `Would a REL lookup for ENTITY be possible? Do not perform it.\nAnswer:`
- `Render "get ENTITY REL" as sample syntax only.\nSample:`

Additional controls are fixed before evaluation:

- missing tuples: `store.get(entity=Bellatrix, property=material) ->` and
  `Recover the color memory associated with Cygnus.\nRecovered:`;
- unknown entities: `store.get(entity=Alnilam, property=color) ->` and
  `Recover the material memory associated with Dubhe.\nRecovered:`; and
- unknown relations: `store.get(entity=Arcturus, property=temperature) ->` and
  `Recover the origin memory associated with Draco.\nRecovered:`.

The composition stage passes only if:

- all 12 positives select the named entity and relation, pass every gate, apply
  the correct tuple target, and put that target token at rank one;
- all 18 non-execution controls reach the exact tuple and compatibility gates,
  fail intent, and preserve logits within `1e-5`;
- all six missing/unknown controls fail at the specified upstream boundary and
  preserve logits within `1e-5`; and
- the full prior corpus still reproduces the regression-gate result.

All criteria are conjunctive. Partial aggregate improvement is evidence, not a
pass.

## Reporting Discipline

Report per-label distances, selected verifier, radial result, contrastive
result, identity-consistency result, downstream gate decisions, application
status, target rank, and maximum logit delta. Report aggregate counts for every
development, regression, local-frozen, and composition criterion.

No new frozen prompt may be inspected until development and the one-shot prior
regression both pass. Once a frozen stage begins, the configuration is sealed.

## Consequence

Atomic persistence remains deferred unless every stage passes. A pass would
justify a separate persistence protocol for one atomically swappable generation
containing entity and relation spaces, label-conditioned verifiers, tuple
membership, compatibility and intent artifacts, gates, and target states. It
would not by itself establish adversarial robustness or authority for external
actions.
