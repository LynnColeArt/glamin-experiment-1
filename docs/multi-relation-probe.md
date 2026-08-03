# Multi-Relation Addressing Probe

## Question

Can one Glamin memory distinguish two facts about the same entity while still
generalizing across unseen prompt forms and abstaining on unknown entity–relation
tuples?

The probe replaced the passing one-relation/eight-entity dataset with four
entities and two relations per entity. For example, `Arcturus/color` mapped to
` blue`, while `Arcturus/material` mapped to ` cedar`. All eight targets remained
distinct one-token values. The frozen Qwen3 model produced the intended target
at rank one for every teacher prompt.

The calibration controls deliberately included both kinds of incomplete tuple:
a known relation with an unknown entity and a known entity with an unknown
relation. Held-out positive and negative prompts were fixed before the first run
and were never used because no candidate construction passed calibration.

## Attempts and Results

| Address construction | Calibration result |
| --- | --- |
| 256-dimensional raw all-token rows, three relation-before-entity views | `Arcturus/color` selected `Arcturus/material`; distance `0.031080` |
| 256-dimensional cumulative prefix means | Same wrong association; distance `0.003049` |
| Add a fourth entity-before-relation construction view | Same wrong association; distance `0.003082` |
| Restrict construction keys to completed-prompt prefix means | Same wrong association; distance `0.003354` |
| Widen prefix-mean projection to 512 dimensions | Same wrong association; distance `0.004830` |
| 512-dimensional final causal state | Correct validation associations, but no safe abstention boundary |

The final-state representation produced a maximum valid calibration distance of
`0.035589`, while an actual control was only `0.008579` from a memory. Relative
confidence did not repair the overlap: the minimum valid margin over the nearest
competing association was `0.004031`, while a negative control had a larger,
wrong-confidence margin of `0.036758`.

## Interpretation

This is a negative result, not a failed implementation. A single nearest-vector
address can represent identity well enough for the one-relation experiment, but
it does not reliably encode an open-set entity–relation tuple. Unknown tuples
can be closer—and more confidently so—than valid paraphrases. Increasing ANN
capacity, pooling more context, adding causal-order views, and comparing the
nearest competing association did not change that boundary.

The next design should factor the address rather than demand that one vector do
all the work:

1. retrieve entity evidence and relation evidence independently;
2. join those results through reviewed geometry or a bounded Glamin mini-app;
3. resolve the resulting tuple to an action only when both components agree;
4. calibrate abstention on the joined result.

A learned supervised projection or contextual gate is another plausible path,
but it would answer a different question and require a real training and
evaluation protocol. The current experiment intentionally does not tune against
its held-out prompts.

## Infrastructure Retained

The probe added an artifact-bound `prefix_mean_rows` address policy and grouped
nearest-competitor diagnostics to the model-agnostic builder. Both are tested
mechanisms even though neither made this particular representation separable.
The primary executable remains the last fully passing one-relation experiment.

The subsequent [factorized memory experiment](factorized-memory.md) implements
the proposed entity/relation split and exact tuple join inside inference.
