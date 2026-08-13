# Sequence-Level Multi-Prototype Entity Evidence

## Status

Protocol frozen on 2026-08-13 before implementation and before any prompt in
its construction, calibration, or development sets is evaluated. At base commit
`9f4793f`, a repository-wide provenance search found no occurrence of the
calibration unknown labels `Alhena`, `Sadr`, `Zosma`, or `Kochab`, nor the
development unknown labels `Hamal`, `Markab`, `Kaus`, or `Ankaa`. Exact prompt
searches also found no occurrence of the construction, calibration, or
development forms instantiated with a registered label and relation.

Every prompt previously evaluated by an entity experiment is regression-only.
The local and composition prompts left untouched by the stopped joint top-two
experiment remain frozen and unavailable until this candidate passes both
development and the one-shot prior regression.

## Motivation

Joint top-two association placed the named label in its candidate set for
32/32 development requests and rejected 32/32 unknown controls exactly, but its
same-state conjunction selected only 28/32. All four failures used the compact
`memory_index` form for Bellatrix or Draco. Candidate generation is therefore
not the observed bottleneck. The label-conditioned verifier is structurally
brittle because it represents each entity with one centroid and requires the
same token state to carry both association and verification evidence.

This experiment replaces that verifier rather than widening it. It tests
whether multiple syntax-view prototypes and sequence-wide identity evidence
can preserve open-set abstention while allowing association and identity to be
witnessed by different prompt-token states.

## Fixed Inputs

The model, weights, tensors, four entity labels, two relation labels, six
registered tuples, target states, association-signal entity space, centroid
relation space, width-128 tuple compatibility artifact, width-64 contrastive
intent artifact, action selection, and payload application remain unchanged.
Entity candidates remain the two distinct labels with the smallest best
association distances. `k=2` and the existing entity association radius are
immutable.

The stopped single-centroid verifier is not reused or widened. No previously
observed prompt, failure distance, target rank, logit, or frozen result may
influence projection width, prototypes, thresholds, scoring, or prompt text.

## Evidence Artifact

The evidence projection has fixed width 64 and uses association-signal feature
selection. It is trained only from the construction corpus below. Each complete
prompt token sequence is one construction view. Automatic separated-state
selection chooses one state from that sequence, producing one normalized
prototype for that entity and syntax view. With three forms, two relations, and
four entities, the artifact contains 24 prototypes: six per entity. Prototypes
are not averaged across views.

The fixed construction forms are:

- `Retrieve the REL value indexed for ENTITY.\nValue:`
- `memory.catalog("ENTITY").read("REL") =>`
- `Inspect ENTITY's retained record and return the REL slot.\nSlot:`

The projection and prototypes are calibrated with two separate known template
families crossed with all registered entities and relations, producing 16
positive sequences:

- `Fetch attribute REL for registry subject ENTITY.\nAttribute:`
- `registry.lookup(subject=ENTITY, field=REL) ->`

Replacing the entity in those forms with `Alhena`, `Sadr`, `Zosma`, and
`Kochab` produces 16 calibration unknown sequences. Construction and
calibration forms are disjoint. Unknown labels do not contribute prototypes.

For a projected sequence `S`, label `e`, and the prototype set `P_e`, select
the state and own-label prototype with minimum squared distance:

```text
(s_e, p_e) = argmin over s in S, p in P_e of distance(s, p)
own(e, S) = distance(s_e, p_e)
competitor(e, S) = min over p in P_not_e of distance(s_e, p)
identity_gap(e, S) = competitor(e, S) - own(e, S)
```

The absolute evidence radius is the midpoint between the hardest calibration
positive distance and nearest calibration-unknown distance. The identity
margin is the midpoint between the weakest positive identity gap and strongest
unknown identity gap. Construction succeeds only with strict separation at
both boundaries and 16/16 correct nearest-label calibration positives.
Thresholds are global rather than label-specific.

## Runtime Selection

For each of the two association candidate labels `e`, compute independently:

```text
association(e) = minimum association distance over all token states
evidence(e) = own(e, complete prompt sequence)

eligible(e) = association(e) <= fixed_association_radius
           && evidence(e) <= calibrated_evidence_radius
           && identity_gap(e) >= calibrated_identity_margin

score(e) = association(e) / fixed_association_radius
         + evidence(e) / calibrated_evidence_radius
```

The association witness and evidence witness may be different token states.
The unique eligible candidate with the lowest score is selected. A tie, missing
prototype, non-finite diagnostic, or absence of an eligible label rejects the
request. Selection may not inspect relation evidence, tuple membership, intent,
payload, target state, logits, or output token. No fallback occurs after a
downstream rejection.

This component is named an entity evidence gate, not a verifier: it estimates
whether activation geometry supports a registered identity and does not verify
the truth of a request.

## Held-Family-Out Development

Four new template families crossed with all registered entities and relations
produce 32 known development requests:

- `Read REL in ENTITY's account record.\nValue:`
- `entity_db[ENTITY].get(REL) ->`
- `Use ENTITY as the catalog key and return REL.\nResult:`
- `Find the REL slot for record owner ENTITY.\nSlot:`

Replacing the entity with `Hamal`, `Markab`, `Kaus`, and `Ankaa` produces 32
corpus-disjoint development unknown controls. These template families and
unknown labels are absent from construction and calibration.

Development passes only if:

- construction produces exactly six prototypes for each registered entity;
- calibration has strict radial and identity-gap separation;
- the named entity appears in the top two for 32/32 known requests;
- the evidence gate selects the named entity for 32/32 known requests;
- all 32 development unknowns are rejected with exact end-to-end no-ops within
  `1e-5`;
- held-fixed compatibility and intent artifacts reproduce their complete
  development results; and
- every association distance, evidence distance, competitor distance, identity
  gap, threshold, and score is finite.

Width, candidate count, projection strategy, key strategy, aggregation,
threshold interpolation, prompts, substitutions, and criteria may not change
after this document is committed. Implementation defects may be repaired using
only construction, calibration, and development data. A model failure stops the
experiment.

## One-Shot Prior Regression

The first implementation that passes development is sealed and evaluated once
on previously observed data. It must produce:

- 32/32 correct selections and 32/32 unknown exact no-ops on the stopped joint
  top-two development corpus;
- 32/32 correct selections and 32/32 unknown exact no-ops on the clean
  corpus-disjoint conditioned-knownness development corpus;
- 12/12 prior local compatibility positives and 60/60 cross-tuple rejections;
- 12/12 prior local intent positives and 36/36 local negative exact no-ops;
- 12/12 prior composition positives routed to rank one;
- 18/18 prior composition negative exact no-ops;
- 30/30 older wrong-intent exact no-ops; and
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations rejected
  with exact no-ops.

Failure stops the experiment. The evidence mechanism and artifact may not be
tuned against a failed regression observation.

## Still-Untouched Frozen Stages

After a complete prior-regression pass, evaluate the exact local and composition
stages preregistered in
[Joint Top-K Entity Selection](joint-top-k-entity-selection.md). Their prompts,
unknown labels, counts, ordering, and conjunctive criteria are inherited
verbatim. They have never been evaluated. No additional frozen prompt may be
introduced after development begins.

## Reporting Discipline

Report both candidate labels and association distances; the selected evidence
state and prototype; own and competitor distances; identity gap; radial and
margin decisions; final score; downstream gate decisions; application state;
target rank; maximum logit delta; and aggregate counts. Report rejected
candidates as fully as accepted candidates so a failure can be attributed to a
specific boundary without rerunning or relaxing it.

## Consequence

Atomic persistence remains deferred unless every stage passes. A pass would
show that bounded sequence-level, multi-view entity evidence resolves the
observed syntax boundary under these corpora. It would not establish factual
truth, adversarial robustness, or authority for external actions.
