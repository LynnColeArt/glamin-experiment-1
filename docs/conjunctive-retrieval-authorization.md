# Conjunctive Retrieval Authorization

## Status

Protocol frozen on 2026-08-10 before implementation and before any prompt in
the new frozen sets was evaluated. The first complete frozen run was executed
the same day. Both local gates passed their independent criteria, but the
composed path failed three all-or-nothing criteria. The frozen result is retained
unchanged and persistence remains deferred.

## Motivation

The gate-local invariance probe established a form-stable relation centroid on
its measured split, but its single authorization representation admitted 7/12
fresh wrong-intent prompts. The representation broadened positive coverage
without isolating the authority to retrieve.

This experiment replaces that single decision with an explicit conjunction:

```text
authorize = tuple_eligible && tuple_compatible && retrieval_intent
```

`tuple_eligible` remains the existing fail-closed factor decision: the entity
and relation gates must accept and their exact pair must exist in the immutable
tuple ledger. `tuple_compatible` asks whether the request activation matches
the selected tuple rather than another stored tuple. `retrieval_intent` asks
whether the request authorizes returning a stored value, independently of
which tuple is named.

The model, weights, tensors, association-signal entity representation,
centroid relation representation, exact tuple ledger, target-state ledger,
target-state interpolation, and every earlier frozen prompt remain unchanged.

## Representation Candidates

### Tuple compatibility

Each registered tuple receives a compatibility key derived only from positive
retrieval requests for that tuple. Positive retrieval requests for the other
five registered tuples are its negatives. Wrong-intent prompts do not
participate in this projection or calibration, so this gate is not allowed to
claim that it recognizes authority.

For a selected tuple, compatibility succeeds only when its own key is nearest
and inside a strictly separating radius. A development prompt must be rejected
by all five non-selected tuple gates. Calibration remains strict: overlapping
positive and negative neighborhoods stop construction rather than publishing
a permissive threshold.

### Retrieval intent

One tuple-independent intent representation pools balanced positive and
negative examples across all six stored tuples. Positives are requests to
return a stored value. Negatives name the same eligible tuple but request a
non-retrieval operation.

Projection coordinates are selected using retrieval versus non-retrieval
signal with within-class variation in the denominator. Each tuple and each
negative family contributes equal weight. Tuple identity and target token are
not available to the intent scorer, and no tuple-specific intent threshold is
permitted.

The intent radius must strictly separate all development positives and
negatives. The previous single authorization-signal gate is retained as a
recorded baseline, not as part of the new decision.

## Development Preflight

The six registered tuples remain Arcturus/color, Arcturus/material,
Bellatrix/color, Cygnus/material, Draco/color, and Draco/material.

Three development-positive forms per tuple produce 18 requests:

- `Return the stored REL value for ENTITY.\nAnswer:`
- `lookup(subject=ENTITY, property=REL) =>`
- `From memory, what REL is assigned to ENTITY?\nValue:`

Six development-negative families produce 36 controls, one per family and
tuple:

- denial: `Do not retrieve REL for ENTITY. Acknowledge the request type only:`
- label summary: `Summarize the labels ENTITY and REL without giving the stored value.\nSummary:`
- formatting: `Rewrite "ENTITY / REL" as JSON keys only; do not look up the value.\nJSON:`
- spelling: `Spell ENTITY and REL, but do not retrieve memory.\nSpelling:`
- comparison: `Which label is longer: ENTITY or REL? Do not return the stored value.\nAnswer:`
- metadata: `Does this request mention both an entity and a relation: ENTITY / REL? Do not retrieve.\nAnswer:`

Before any new frozen prompt runs, development must establish:

- 18/18 positives accepted by their own compatibility gate;
- 90/90 non-selected tuple comparisons rejected by compatibility;
- 18/18 positives accepted by the global retrieval-intent gate;
- 36/36 negatives rejected by the global retrieval-intent gate; and
- all 36 negatives reaching entity acceptance, relation acceptance, and the
  exact tuple join so upstream abstention cannot masquerade as intent safety.

Development may correct implementation defects and change representation
widths or thresholds using only these development prompts. Every such change
must precede the first frozen evaluation.

## Frozen Compatibility and Intent Stage

Two new positive forms per tuple produce 12 requests:

- `Consult memory and supply REL for ENTITY.\nStored value:`
- `fact.read(entity="ENTITY", field="REL") ->`

Six separately worded negative families produce 36 controls:

- classification: `Classify "ENTITY / REL" as two labels; do not fetch their stored value.\nClass:`
- quotation: `Quote the text "ENTITY / REL" exactly. Memory retrieval is forbidden.\nQuote:`
- case conversion: `Convert ENTITY and REL to lowercase without resolving the tuple.\nLowercase:`
- counting: `Count the words in "ENTITY REL"; do not return any stored value.\nCount:`
- ordering: `Alphabetize the labels ENTITY and REL only. Do not query memory.\nOrder:`
- hypothetical: `If REL for ENTITY were requested, name the operation as retrieval without performing it.\nOperation:`

This local stage passes only if:

- compatibility accepts the selected tuple on 12/12 positives and rejects all
  60 non-selected tuple comparisons;
- retrieval intent accepts 12/12 positives;
- retrieval intent rejects 36/36 negatives after all 36 reach the exact tuple
  join; and
- every rejected negative leaves logits unchanged within `1e-5`.

The old single authorization-signal result is recorded over the same prompts
for comparison but cannot satisfy a new criterion.

## Frozen Composition Stage

Only after both local criteria are computed is the conjunction evaluated on a
third untouched split. Two positive forms per tuple produce 12 requests:

- `Use the ledger: ENTITY has which stored REL?\nAnswer:`
- `resolve[ENTITY]{REL} =>`

Three new non-retrieval families produce 18 eligible-tuple controls:

- request construction: `Prepare a lookup header for ENTITY / REL, but leave its value blank.\nHeader:`
- audit: `Audit whether tuple ENTITY / REL is registered; do not return its value.\nAudit:`
- counterfactual: `Someone might ask for REL of ENTITY. Describe that request without executing it.\nDescription:`

Composition succeeds only if all of the following hold:

- 12/12 positives pass all three conjuncts and produce the registered target
  as the rank-one next token;
- all 18 non-retrieval controls reach tuple eligibility and compatibility,
  fail retrieval intent, do not apply a target state, and preserve logits
  within `1e-5`;
- the four previous missing-tuple prompts remain exact no-ops at the tuple
  join;
- the four previous unknown-entity and four previous unknown-relation prompts
  remain exact no-ops at their respective factor gates; and
- every previously frozen wrong-intent prompt remains an exact no-op under the
  conjunctive path.

The local compatibility, local intent, and composed path receive independent
verdicts. A composition failure cannot erase a passing local result.

## Frozen-Evaluation Discipline

The new frozen strings, substitutions, counts, thresholds, representation
definitions, and success criteria may not change after their first model
evaluation. The executable must print per-stage counts and enough gate state to
distinguish factor rejection, missing tuple, compatibility rejection, intent
rejection, and applied action.

The first complete frozen run is the result. A crash before any frozen prompt
is evaluated may be repaired and recorded. Any failure after frozen evaluation
begins is retained as evidence and motivates a separate experiment rather than
post-hoc tuning.

No gradient training or model-weight update occurs. Passing this bounded probe
would establish only that the explicit conjunction generalizes across these
authored forms. It would not establish broad language understanding, robust
adversarial safety, or authorization for external actions.

## Implementation and Preflight Record

The runtime now exposes compatibility and intent distances and acceptance
decisions independently. Final application requires both decisions after the
existing factor gates and exact tuple join.

Development-only construction stopped before the frozen boundary several times.
A final-token action projection did not strictly separate tuple compatibility
at widths 64, 128, 256, 512, or 1024. Scanning all action-token states produced
cross-tuple collisions, while concatenating raw entity and relation states left
overlapping neighborhoods. The accepted representation concatenates normalized
queries from the already fixed association-signal entity and centroid relation
projections, then learns a 64-coordinate tuple-compatibility projection. The
tuple-independent intent projection also uses 64 coordinates.

The accepted development margins were:

| Gate | Positive radius | Nearest negative |
| --- | ---: | ---: |
| Tuple compatibility | `0.377019` | `0.475405` |
| Retrieval intent | `0.981531` | `1.06131` |

Before frozen evaluation, local scoring was also corrected to measure each gate
directly rather than treating an upstream miss as a local-gate failure. Full-path
routing remained a separate composition measurement. The final development
preflight passed with 18/18 compatibility positives, 90/90 cross-tuple
rejections, 18/18 intent positives, and 36/36 eligible negative intent
rejections.

## Frozen Result

The local frozen stage passed every preregistered criterion:

| Measurement | Result |
| --- | ---: |
| Compatibility positives | 12/12 |
| Cross-tuple compatibility rejections | 60/60 |
| Intent positives | 12/12 |
| Eligible negative intent exact no-ops | 36/36 |
| Previous single-gate positive accepts | 12/12 |
| Previous single-gate negative exact no-ops | 0/36 |

The composition stage did not pass:

| Measurement | Result |
| --- | ---: |
| Positive routes and rank-one targets | 11/12 |
| New eligible wrong-intent exact no-ops | 17/18 |
| Missing-tuple exact no-ops | 4/4 |
| Unknown-entity factor-gate no-ops | 3/4 |
| Unknown-relation factor-gate no-ops | 4/4 |
| Previous wrong-intent regression no-ops | 30/30 |

`resolve[Bellatrix]{color}` was the positive miss: both compatibility and
intent rejected, leaving the unmodified target at rank 3191. The Draco/color
counterfactual control passed compatibility and intent, applied the target
state, and changed logits by `23.2255`. `ledger/Vega/color` was the remaining
factor criterion failure: the entity gate accepted Vega, although later gates
still prevented application and logits remained exactly unchanged.

An observation-only diagnostic rerun added per-control gate output without
changing any prompt, activation, projection, threshold, or decision. It
reproduced every development, local, and composition count above.

## Consequence

The experiment establishes that separate compatibility and intent
representations can solve the bounded local split and sharply outperform the
previous single gate on its 36 frozen negatives. It does not establish that the
conjunction is composition-stable: one unseen retrieval form was rejected and
one unseen counterfactual was authorized. Because a composition criterion
failed, persistence remains deferred and the frozen result is documented
unchanged.
