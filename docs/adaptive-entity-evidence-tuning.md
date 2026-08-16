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
