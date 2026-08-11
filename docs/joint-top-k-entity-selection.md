# Joint Top-K Entity Selection

## Status

Protocol frozen on 2026-08-11 before implementation and before any prompt in
its development or frozen sets is evaluated. At base commit `03aa42e`, a
repository-wide provenance search found no occurrence of the development
unknown labels `Pollux`, `Castor`, `Alpheratz`, or `Mirfak`. All prompts and
labels observed by prior experiments remain regression-only. The frozen unknown
labels `Rasalhague`, `Merak`, `Nunki`, and `Schedar` were also absent at the base
commit. The sealed development preflight later reached 32/32 correct top-two
inclusions but only 28/32 correct selections. It stopped before the prior-frozen
regression and before every new frozen prompt.

## Motivation

The corpus-disjoint label-conditioned verifier passed development but reached
only 10/12 prior positives. One Bellatrix request proposed the correct label but
failed verification; another committed to Arcturus before the verifier ran. A
veto applied after single-nearest selection cannot recover from either an
unhelpful token state or the wrong first label.

This experiment tests a structural change rather than a wider threshold:
evaluate a bounded set of entity labels and token states jointly, then authorize
only one candidate that independently satisfies association and verification.

## Fixed Inputs

The model, weights, tensors, registered entity and relation labels, six tuples,
target states, relation addressing, width-128 tuple compatibility, width-64
contrastive intent, and action application remain unchanged. The width-32
corpus-disjoint label-conditioned verifier is rebuilt from its recorded clean
development corpus and calibration rules without modification.

No prior positive, negative, or failed distance may influence candidate count,
scoring, thresholds, or prompt construction.

## Candidate Mechanism

For every address token state and every entity-space row, compute association
distance and retain the best distance per distinct entity label. The two labels
with the smallest distances form the fixed candidate set. `k=2` is immutable.

For each candidate label `e`, consider every address token state `s`. Define:

```text
association_ok(e, s) = association_distance(e, s) <= entity_radius

verification_ok(e, s) = verifier_positive_distance(e, s) <= verifier_radius[e]
                     && verifier_positive_distance(e, s)
                          + verifier_margin[e]
                          <= verifier_negative_distance(e, s)
                     && nearest_verifier_label(s) == e

joint_score(e, s) = association_distance(e, s) / entity_radius
                  + verifier_positive_distance(e, s) / verifier_radius[e]
```

A label is eligible when at least one state satisfies both predicates. Its
representative is the eligible state with the smallest joint score. The unique
eligible label with the smallest representative score is selected. An exact
score tie, no eligible label, non-finite score, or missing verifier rejects the
entity request. The selector may choose the second association label but may
not inspect relation, tuple, intent, payload, logits, or target token.

The selected label and representative state feed the unchanged relation,
tuple, compatibility, intent, and action stages. No fallback occurs after a
downstream rejection.

## Development Preflight

Four new forms crossed with all four registered labels and both relations
produce 32 known-label requests:

- `Load the REL cell belonging to ENTITY.\nCell:`
- `memory_index[ENTITY].read(REL) ->`
- `Return ENTITY's archived REL field.\nArchive value:`
- `Query record owner ENTITY for attribute REL.\nResult:`

Replacing the entity with `Pollux`, `Castor`, `Alpheratz`, and `Mirfak`
produces 32 corpus-disjoint unknown controls.

Development passes only if:

- the named registered label appears in the top two for 32/32 known requests;
- joint selection chooses and verifies that label for 32/32 known requests;
- no candidate is selected for 32/32 unknown requests;
- all 32 unknown end-to-end requests are exact no-ops within `1e-5`;
- the held-fixed verifier reproduces its prior clean development result;
- held-fixed compatibility and intent reproduce their complete development
  results; and
- all scores and per-stage diagnostics are finite.

The candidate count, scoring equation, thresholds, prompt text, substitutions,
counts, and criteria may not change after this document is committed.
Implementation defects may be repaired using only this development split. A
development failure stops the experiment.

## Development Result

The implementation was sealed at commit `5ff8b48` and failed the conjunctive
development criterion:

| Measurement | Result |
| --- | ---: |
| Named label included in top two | 32/32 |
| Named label selected and verified | 28/32 |
| Known-request diagnostics finite | 32/32 |
| Unknown requests rejected | 32/32 |
| Unknown end-to-end exact no-ops | 32/32 |
| Unknown-request diagnostics finite | 32/32 |

All four misses used the `memory-index` form. Bellatrix/color,
Bellatrix/material, Draco/color, and Draco/material each included the named
label in the association top two, but no candidate state satisfied the joint
association and verifier conditions. Their selector therefore rejected with no
eligible joint score rather than substituting another label.

This is a development failure, not a regression or frozen result. Per the
frozen protocol, candidate count, scoring, thresholds, and prompts were not
tuned after observing it. The prior-frozen regression, untouched local frozen
stage, and untouched composition stage were never evaluated.

## Prior-Frozen Regression Gate

The first implementation that passes development is sealed and evaluated once
on the full previously observed corpus. It must produce:

- 12/12 prior local compatibility positives and 60/60 cross-tuple rejections;
- 12/12 prior local intent positives and 36/36 local negative exact no-ops;
- 12/12 prior composition positives routed to rank one;
- 18/18 prior composition negative exact no-ops;
- 30/30 older wrong-intent exact no-ops; and
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations rejected
  with exact no-ops.

Failure stops the experiment. The candidate may not be tuned against a failed
regression observation, including either prior Bellatrix result.

## Untouched Local Frozen Stage

Two forms crossed with all registered labels and both relations produce 16
known-label requests:

- `Read REL from the profile headed ENTITY.\nValue:`
- `catalog.entity(ENTITY).field(REL).get() =>`

Replacing the entity with `Rasalhague`, `Merak`, `Nunki`, and `Schedar` produces
16 unknown controls. Success requires 16/16 correct top-two inclusion and joint
selection, 16/16 unknown rejections, and exact no-ops on every rejection.

## Untouched Composition Stage

Two forms per registered tuple produce 12 positives:

- `Fetch the REL datum in ENTITY's retained record.\nDatum:`
- `archive.lookup(entity=ENTITY, relation=REL) ->`

Three forms per tuple produce 18 non-execution controls:

- `Outline a request for ENTITY's REL without sending it.\nOutline:`
- `Check whether ENTITY could have a REL lookup; do not retrieve it.\nCheck:`
- `Show "lookup ENTITY REL" as inert example text.\nExample:`

Additional controls are:

- missing tuples: `archive.lookup(entity=Bellatrix, relation=material) ->` and
  `Fetch the color datum in Cygnus's retained record.\nDatum:`;
- unknown entities: `archive.lookup(entity=Rasalhague, relation=color) ->` and
  `Fetch the material datum in Merak's retained record.\nDatum:`; and
- unknown relations: `archive.lookup(entity=Arcturus, relation=density) ->` and
  `Fetch the birthplace datum in Draco's retained record.\nDatum:`.

All 12 positives must select the named factors, pass every gate, apply the
correct target state, and rank the target token first. All 18 non-execution
controls must be exact no-ops; controls that identify a registered tuple must
reach and fail intent. Missing and unknown controls must fail at their specified
upstream boundary and remain exact no-ops. The full prior corpus must continue
to pass. Every criterion is conjunctive.

## Reporting Discipline

Report the top-two labels and association distances, every eligible state's
association and verifier distances, joint scores, selected label and state,
downstream gate results, application state, target rank, maximum logit delta,
and aggregate counts. Development must pass before regression; regression must
pass before any untouched prompt. Once either boundary begins, its configuration
is immutable.

## Consequence

Atomic persistence remains deferred unless every stage passes. A pass would
show that bounded joint candidate-state selection resolves the observed recall
boundary under these corpora; it would not establish adversarial robustness or
authority for external actions.
