# Adaptive Entity-Evidence Tuning

## Status

Adaptive engineering lane opened on 2026-08-16 from merged baseline `cec70fa`.
The stopped sequence-evidence experiment and its one-shot result remain
immutable history. This lane explicitly permits repeated evaluation, diagnostic
instrumentation, parameter tuning, and architectural changes on every prompt
that had already been executed before this branch was created.

Results on reused prompts are optimization measurements, not unbiased
regression evidence.

## Reusable Tuning Corpus

The following are now development data and may be run repeatedly:

- all construction, calibration, and held-family development prompts from
  sequence-level multi-prototype entity evidence;
- the stopped joint top-two development corpus;
- the clean corpus-disjoint conditioned-knownness development corpus;
- all prior local compatibility and intent prompts;
- all prior composition positives and negatives;
- all older wrong-intent, missing-tuple, unknown-entity, and unknown-relation
  controls; and
- the observed Bellatrix/color `ledger` and `resolve` failures.

Every iteration must report complete candidate diagnostics and the full tuning
corpus aggregate. A change may be retained only if it improves known recall
without losing any previously passing exact no-op or downstream authority
criterion.

## Sealed Evaluation Corpus

The exact local and composition stages documented under “Untouched Local
Frozen Stage” and “Untouched Composition Stage” in
[Joint Top-K Entity Selection](joint-top-k-entity-selection.md) remain sealed.
Their prompts, labels, ordering, and expected results may not be executed during
tuning. They will be evaluated once only after a configuration passes the full
reusable corpus and is committed as locked.

## First Tuning Hypothesis

The registered entity vocabulary contains only four labels. Restricting the
sequence evidence gate to the two closest association labels can discard a
label that has strong multi-prototype identity evidence elsewhere in the
sequence. The first adaptive iteration therefore evaluates all four registered
labels while retaining:

- the fixed association radius;
- the width-64 evidence projection and all 24 prototypes;
- the calibrated evidence radius and identity margin;
- the normalized association-plus-evidence score;
- unique-best selection and fail-closed ties; and
- every relation, compatibility, intent, action, and payload boundary.

This isolates candidate breadth. If a named label still fails, diagnostics will
identify association radius, evidence radius, identity margin, or score
competition before another mechanism changes.

## Iteration Results

Evaluating all four labels retained the complete fresh-development result and
every no-op control, but the reused prior corpus remained at 30/32 joint-known
selections and 11/12 composition routes. All three misses had the same isolated
cause:

- the Bellatrix sequence-evidence distance was inside its calibrated radius;
- the gap to the nearest competing label was far above the calibrated identity
  margin; and
- only the association distance was outside the legacy association radius.

For example, `resolve/Bellatrix/color` had evidence distance `0.246388` and
identity gap `1.45154`, while its association distance `1.1014` alone prevented
eligibility. The two `memory-index/Bellatrix/*` requests had evidence distance
`0.220927` and identity gap `1.29914` with the same failure mode.

## Second Tuning Hypothesis

Once every registered label is evaluated, association is a candidate-proposal
and ranking signal rather than independent proof of identity. Requiring it to
also pass its radius duplicates authority and can veto strong, unambiguous
sequence evidence. The second iteration therefore:

- retains association distance in the normalized joint score and diagnostics;
- removes association-radius acceptance from sequence-evidence eligibility;
- keeps the calibrated sequence-evidence radius and cross-label identity margin
  as hard gates; and
- leaves all downstream relation, compatibility, intent, action, and payload
  gates unchanged.

This is an authority correction, not verifier retraining. Unknown requests must
still fail both calibrated evidence safeguards and remain exact no-ops.

## Locked Configuration

The second iteration satisfied the complete reusable-corpus objective:

- fresh development: 32/32 known selections and 32/32 unknown exact no-ops;
- prior joint corpus: 32/32 known selections and 32/32 unknown exact no-ops;
- conditioned corpus: 32/32 known selections and 32/32 unknown exact no-ops;
- prior local corpus: 12/12 compatibility, 60/60 cross checks, 12/12 intent,
  and 36/36 negative exact no-ops;
- prior composition corpus: 12/12 routes at rank one and 18/18 negative exact
  no-ops; and
- older safety controls: 30/30 negatives plus 4/4 each for missing tuples,
  unknown entities, and unknown relations, all exact no-ops.

The configuration is therefore locked with four-label candidate evaluation,
association retained in ranking, and sequence-evidence radius plus identity
margin as the hard identity gates. No verifier retraining or threshold change
was required. The sealed harness may be added only after this lock is committed.

## Tuning Objective

A lock candidate must achieve all of the following on the reusable corpus:

- every registered-entity request selects the named entity;
- all 12 prior composition positives route and rank the target first;
- every local and composition negative remains an exact no-op;
- every older wrong-intent, missing-tuple, unknown-entity, and unknown-relation
  control retains its prior exact no-op result; and
- all candidate and downstream diagnostics are finite.

Once a candidate satisfies this objective, commit its complete configuration
before adding or executing the sealed evaluation harness. Any failure on the
sealed corpus ends that locked evaluation; it may inform a later adaptive lane
but may not be used to relabel the failed lock as an unbiased pass.
