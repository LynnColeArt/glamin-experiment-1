# Composition-Stable Retrieval Authorization Replication

## Status

Protocol frozen on 2026-08-10 before implementation and before any prompt in
the new frozen sets is evaluated. The previously observed conjunctive prompts
are regression evidence only. They may not be used to select coordinates,
thresholds, margins, widths, or prompt families.

## Motivation

The first conjunctive experiment passed both local frozen gates but failed its
strict composition criterion. One unseen retrieval form was rejected by both
compatibility and intent, one counterfactual was authorized, and one unknown
entity passed the upstream entity gate. Those observations identify three
failure categories, not examples to tune against.

This replication asks whether three independently calibrated decisions can
remain stable when composed:

```text
authorize = known_entity
         && relation_eligible
         && exact_tuple_exists
         && tuple_compatible
         && retrieval_intent
```

The model, weights, tensors, target-state ledger, target-state interpolation,
registered tuples, entity labels, relation labels, association-signal entity
addressing, centroid relation addressing, and every prior frozen prompt remain
unchanged.

## Candidate Mechanism

### Known-entity veto

Association-signal addressing still selects the candidate entity. A new
tuple-independent knownness scorer then distinguishes requests containing one
of the four registered entity labels from requests containing an unseen label.
It uses balanced development views and a positive-versus-unknown discriminant;
it may veto a candidate but may not replace the selected entity or authorize an
action.

Acceptance requires both a positive radius and a contrastive margin:

```text
d_known <= r_known && d_known + m_known <= d_unknown
```

The radius and contrastive threshold must strictly separate every development
positive and unknown control. The calibrated `m_known` may be signed because it
is the midpoint between the hardest positive and negative distance-difference
scores; no sign constraint was preregistered. If no tested development width
separates the scores, construction stops before regression or frozen evaluation.

### Tuple compatibility

The passing mechanism is retained: normalized association-signal entity and
centroid relation queries are concatenated and projected to tuple keys. The
projection is rebuilt from the new development-positive forms below so compact
and prose retrieval syntax contribute balanced evidence. Wrong-intent prompts
remain excluded from compatibility construction.

The selected tuple must own the nearest compatibility key within its calibrated
radius. Every positive must also be rejected by all five non-selected tuple
keys.

Development clarified radius calibration before any prior-frozen or new frozen
prompt was evaluated. Compatibility keys and projection coordinates remain
positive-only. The radius is the midpoint between the hardest own-tuple distance
across all 72 eligible development requests and the nearest wrong-tuple key.
This uses wrong-intent requests only to require that they reach the intent gate;
they do not contribute to a compatibility key or projection score.

### Contrastive retrieval intent

The prior positive-radius intent gate is replaced by two tuple-independent
prototypes: retrieval and non-execution. Projection coordinates are selected
from balanced positive and negative development families with tuple identity
held out of the score. Acceptance requires:

```text
d_retrieve <= r_retrieve && d_retrieve + m_intent <= d_non_execution
```

Both inequalities must strictly separate all development examples. The
negative prototype is a veto signal, not an action class. No tuple-specific
intent prototype, threshold, or margin is permitted.

Development clarified the conjunctive calibration before any prior-frozen or
new frozen prompt was evaluated. The radius encloses the hardest development
positive and uses half of any remaining gap to the nearest negative; if those
radial neighborhoods overlap, the radius stops at the hardest positive. The
signed prototype threshold must then reject every overlapping negative. Thus
every positive must satisfy both conditions and every negative must fail at
least one; radial separation is not treated as a hidden third criterion.
All 24 declared development positives contribute equally to the retrieval
prototype and are also checked by the preflight. There is no held-out claim
inside the development split; generalization is reserved for the frozen stages.

Widths may be selected from `64`, `128`, `256`, and `512` using development
only. Thresholds and margins must be derived mechanically from the hardest
development positive and nearest development negative with a nonzero gap. A
contrastive margin is a signed distance-difference threshold; its sign is not a
success criterion.

## Development Preflight

The six registered tuples remain Arcturus/color, Arcturus/material,
Bellatrix/color, Cygnus/material, Draco/color, and Draco/material.

Four positive forms per tuple produce 24 requests:

- `Read memory at ENTITY::REL.\nStored:`
- `memory.fetch(subject="ENTITY", field="REL") =>`
- `Give the saved REL belonging to ENTITY.\nValue:`
- `Retrieve from the ledger (ENTITY, REL).\nAnswer:`

Eight non-execution families produce 48 eligible-tuple negatives:

- plan: `Plan how one would retrieve REL for ENTITY without doing it.\nPlan:`
- conditional: `If REL for ENTITY were needed later, describe the request only.\nDescription:`
- quoted command: `Quote the command "fetch ENTITY REL" without executing it.\nQuote:`
- template: `Draft a blank request template for ENTITY / REL; leave the value empty.\nTemplate:`
- permission: `State whether retrieving REL for ENTITY would be allowed, but do not retrieve it.\nDecision:`
- explanation: `Explain what a REL lookup for ENTITY would mean without performing one.\nExplanation:`
- syntax rewrite: `Rewrite "retrieve ENTITY REL" as pseudocode only; do not run it.\nCode:`
- mention detection: `Does this sentence mention a possible lookup of ENTITY and REL? Do not look it up.\nAnswer:`

Knownness development uses the same four positive forms for all registered
tuples. Replacing the entity with `Deneb` and `Polaris` for both relations and
all four forms produces 16 unknown-entity controls.

Before any previously frozen or newly frozen prompt runs, development must
establish:

- 24/24 registered entities accepted by knownness and 16/16 unknown entities
  rejected;
- 24/24 positives accepted by their own compatibility key;
- 120/120 non-selected tuple comparisons rejected;
- 24/24 positives accepted by contrastive retrieval intent;
- 48/48 non-execution controls reaching the exact tuple and compatibility
  decisions, then rejected by intent; and
- every rejection preserving logits within `1e-5`.

Implementation defects may be repaired and widths may be selected using only
this development split. Prompt text, substitutions, counts, mechanisms, and
criteria may not change after this document is committed.

## Prior-Frozen Regression Gate

The first configuration that passes development is evaluated once on the
already observed frozen corpus before any new frozen prompt. This stage does
not measure generalization; it prevents a new mechanism from discarding known
behavior.

It must produce:

- 12/12 prior local compatibility positives and 60/60 cross-tuple rejections;
- 12/12 prior local intent positives and 36/36 local negative no-ops;
- 12/12 prior composition positives at rank one, including the previously
  rejected Bellatrix/color form;
- 18/18 prior composition negative no-ops, including the previously authorized
  Draco/color counterfactual;
- all 30 older wrong-intent controls as exact no-ops;
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations rejected
  at their specified gates with exact no-ops.

Failure stops the experiment. The configuration may not be tuned against a
failed regression prompt; a materially revised approach requires a new
preregistration.

## New Frozen Local Stage

Two positive forms per tuple produce 12 requests:

- `stored_record(ENTITY).REL ->`
- `Answer from memory: for ENTITY, return the recorded REL.\nAnswer:`

Six non-execution families produce 36 eligible-tuple controls:

- simulation: `Simulate writing a request for ENTITY / REL, but do not access memory.\nSimulation:`
- rationale: `Explain why REL for ENTITY could be retrieved, without retrieving it.\nReason:`
- yes/no classification: `Is "get REL for ENTITY" a retrieval instruction? Answer yes or no without executing it.\nAnswer:`
- placeholder: `Create a placeholder for the future REL value of ENTITY; keep it blank.\nPlaceholder:`
- translation: `Translate "fetch ENTITY REL" into plain English without carrying it out.\nTranslation:`
- delegation: `Describe how another system could retrieve REL for ENTITY; do not retrieve it here.\nDescription:`

Four unknown-entity controls use `Capella` and `Aldebaran`, the `color`
relation, and both positive forms.

This stage passes only if:

- knownness accepts 12/12 registered positives and rejects 4/4 unknowns;
- compatibility accepts 12/12 selected tuples and rejects 60/60 cross-tuples;
- intent accepts 12/12 positives;
- all 36 negatives reach known entity, relation, exact tuple, and compatibility,
  then fail intent with exact no-ops.

## New Frozen Composition Stage

Two untouched positive forms per tuple produce 12 requests:

- `vault.get<ENTITY, REL>() =>`
- `Using stored facts, complete REL(ENTITY):`

Three untouched non-execution families produce 18 eligible-tuple controls:

- pseudocode: `Write pseudocode that would fetch REL for ENTITY, but do not run it.\nCode:`
- restatement: `A user could request REL for ENTITY. Restate that request without answering it.\nRestatement:`
- instruction audit: `Check whether "retrieve ENTITY REL" is an instruction; do not execute it.\nAudit:`

The two missing tuples, Bellatrix/material and Cygnus/color, are tested with
both positive forms for four controls. `Fomalhaut` and `Regulus` replace the
entity in both positive forms with relation `color` for four unknown-entity
controls. `density` and `era` replace the relation for Arcturus in both forms
for four unknown-relation controls.

Composition succeeds only if:

- 12/12 positives pass every gate, apply the registered target state, and put
  the target token at rank one;
- 18/18 non-execution controls reach compatibility, fail intent, do not apply,
  and preserve logits within `1e-5`;
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations fail at
  their specified gates and preserve logits within `1e-5`; and
- every prompt in the prior-frozen regression gate reproduces its required
  result under the final composed path.

Local knownness, compatibility, intent, and composition receive independent
verdicts. A later failure cannot erase an earlier passing local result.

## Frozen-Evaluation Discipline

The first complete new frozen run is the result. No new frozen prompt may be
evaluated until development and the prior-frozen regression gate both pass.
After new frozen evaluation begins, no prompt, substitution, projection,
prototype, width, threshold, margin, tensor, or success criterion may change.

The executable must print per-stage counts and per-failure gate state. A crash
before the first new frozen prompt may be repaired and recorded. A failure after
that point remains evidence and motivates another preregistered experiment.

No gradient training or model-weight update occurs. Passing would establish
only bounded generalization across these authored forms and synthetic labels.
It would not establish adversarial robustness, general language understanding,
or authority for external actions.

## Consequence

Atomic persistence remains deferred unless every development, regression,
local-frozen, and composition criterion passes. If they all pass, a subsequent
protocol may persist and live-swap the entity space, relation space, knownness
veto, tuple ledger, compatibility representation, contrastive intent
representation, gates, and target states as one generation.
