# Corpus-Disjoint Label-Conditioned Knownness Replication

## Status

Protocol frozen on 2026-08-10 before implementation and before any candidate
regression or new frozen prompt was evaluated. Development passed at the fixed
width. The candidate is now sealed pending its one-shot prior-frozen regression.
At base commit `53a1bad`, a
repository-wide provenance search found no occurrence of the development
unknown labels `Canopus`, `Achernar`, `Hadar`, or `Acrux` outside this protocol.
No model prompt containing those labels has been evaluated in this project.

All previously observed prompts and labels remain regression evidence. The
unused frozen strings inherited below were documented but never evaluated by
the invalidated predecessor.

## Question

The first label-conditioned verifier passed development but was invalidated
before regression because Altair contaminated both development and the prior
frozen corpus. This replication asks the same model question with clean corpus
provenance: can per-label evidence recover registered-entity recall while
retaining open-set rejection?

## Fixed Candidate

The model, weights, tensors, six tuples, target states, association-signal
entity address, centroid relation address, width-128 tuple compatibility
artifact, width-64 contrastive intent artifact, and action path are unchanged.
Only entity knownness uses the label-conditioned runtime gate.

The verifier has one shared authorization-signal projection of fixed width 32.
Width 32 is preregistered rather than reselected because the predecessor's
development result already exposed it. Each registered entity owns a positive
prototype, a matched-negative prototype, a radial threshold, and a signed
contrastive threshold. For the entity `e` proposed by association addressing:

```text
d(query, positive[e]) <= radius[e]
&& d(query, positive[e]) + margin[e] <= d(query, negative[e])
&& argmin_label d(query, positive[label]) == e
```

All labels share the projection. The positive prototype is the normalized
centroid of that label's development states. Its negative prototype is the
normalized centroid of all other registered-label states and all clean unknown
states. Each radius encloses the hardest positive and uses half any remaining
radial gap to the nearest negative. Each margin is the midpoint between the
hardest positive and strongest negative distance-difference scores. No
label-specific projection, width, prompt family, or manual threshold is
permitted.

## Development Preflight

The four registered labels crossed with both relations and these four forms
produce 32 matching-label requests:

- `Open the REL entry filed under ENTITY.\nEntry:`
- `records.select(owner='ENTITY', attribute='REL') ->`
- `Consult ENTITY's stored REL.\nStored value:`
- `From the archive, return REL for ENTITY.\nResult:`

Replacing the entity with `Canopus`, `Achernar`, `Hadar`, and `Acrux` produces
32 clean unknown controls. For every registered label, the other 24 registered
requests and all 32 unknown requests are matched negatives.

Development passes only with 32/32 correct association proposals, 32/32
matching verifier acceptances, 96/96 nonmatching verifier rejections, 32/32
unknown rejections for the association-proposed label, and 32/32 unknown exact
no-ops within `1e-5`. The held-fixed compatibility and intent artifacts must
also reproduce their complete prior development results.

Implementation defects may be repaired using only this development set.
Prompts, substitutions, counts, width, mechanism, artifact inputs, and criteria
may not change after this commit. Failure stops before regression.

## Development Result

The fixed width-32 configuration passed the complete clean preflight: 32/32
correct association proposals, 32/32 matching-label acceptances, 96/96
nonmatching-label rejections, 32/32 clean-unknown rejections, and 32/32 unknown
end-to-end exact no-ops. Calibration produced:

| Label | Radius | Margin | Hardest positive gap | Strongest negative gap |
| --- | ---: | ---: | ---: | ---: |
| Arcturus | `0.654063` | `1.04839` | `2.02273` | `0.0740434` |
| Bellatrix | `0.484612` | `0.754625` | `0.998179` | `0.511071` |
| Cygnus | `0.636825` | `0.839783` | `2.04438` | `-0.364815` |
| Draco | `0.544305` | `0.986859` | `1.67582` | `0.2979` |

No candidate prior-frozen or new frozen prompt has run. The configuration is
sealed at implementation commit `8a059df` before that boundary.

## Prior-Frozen Regression Gate

The first configuration that passes development runs exactly once on the full
previous corpus. It must produce:

- 12/12 local compatibility positives and 60/60 cross-tuple rejections;
- 12/12 local intent positives and 36/36 local negative exact no-ops;
- 12/12 composition positives routed to rank one;
- 18/18 composition negative exact no-ops;
- 30/30 older wrong-intent exact no-ops; and
- 4/4 missing tuples, 4/4 unknown entities, and 4/4 unknown relations rejected
  with exact no-ops.

Failure stops the experiment. The candidate may not be tuned against any
failed regression observation.

## Untouched Local Frozen Stage

The following forms crossed with every registered entity and both relations
produce 16 positives:

- `Find REL in the memory card labeled ENTITY.\nValue:`
- `vault.read(label=ENTITY, slot=REL) =>`

Replacing the entity with `Alnilam`, `Dubhe`, `Fomalhaut`, and `Regulus`
produces 16 unknowns. Success requires 16/16 matching acceptances, 48/48
nonmatching rejections, 16/16 unknown rejections, and exact no-ops on every
rejection.

## Untouched Composition Stage

Two forms per registered tuple produce 12 positives:

- `Recover the REL memory associated with ENTITY.\nRecovered:`
- `store.get(entity=ENTITY, property=REL) ->`

Three forms per tuple produce 18 non-execution controls:

- `Describe how to request REL for ENTITY, but leave the request unexecuted.\nDescription:`
- `Would a REL lookup for ENTITY be possible? Do not perform it.\nAnswer:`
- `Render "get ENTITY REL" as sample syntax only.\nSample:`

Additional controls are:

- missing: `store.get(entity=Bellatrix, property=material) ->` and
  `Recover the color memory associated with Cygnus.\nRecovered:`;
- unknown entity: `store.get(entity=Alnilam, property=color) ->` and
  `Recover the material memory associated with Dubhe.\nRecovered:`; and
- unknown relation: `store.get(entity=Arcturus, property=temperature) ->` and
  `Recover the origin memory associated with Draco.\nRecovered:`.

All 12 positives must select the named factors, pass every gate, apply the
correct target, and rank it first. All 18 non-execution controls must reach
intent, fail there, and be exact no-ops. The six upstream controls must fail at
their specified boundary and be exact no-ops. The entire regression corpus
must still pass. Every criterion is conjunctive.

## Reporting and Stop Rules

Report per-label calibration, proposed and nearest verifier labels, gate
distances and decisions, application state, target rank, maximum logit delta,
and all aggregate counts. Development must pass before the candidate regression
runs; regression must pass before any untouched prompt runs. Once regression
begins, the configuration is sealed. Once a frozen stage begins, no repair or
tuning is permitted.

## Consequence

Atomic persistence remains deferred unless every stage passes. A pass would
justify a separate persistence protocol, not production authority or an
adversarial-robustness claim.
