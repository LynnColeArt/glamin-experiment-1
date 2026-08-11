# Corpus-Disjoint Label-Conditioned Knownness Replication

## Status

Protocol frozen on 2026-08-10 before implementation and before any candidate
regression or new frozen prompt was evaluated. Development passed at the fixed
width. The first and only candidate prior-frozen run then failed positive recall
at 10/12. The experiment stopped; no new frozen prompt was evaluated. At base
commit `53a1bad`, a
repository-wide provenance search found no occurrence of the development
unknown labels `Canopus`, `Achernar`, `Hadar`, or `Acrux` outside this protocol.
Before this candidate's development run, no model prompt containing those
labels had been evaluated in this project.

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

The configuration was sealed at implementation commit `8a059df`; the regression
harness was sealed at commit `d6207d2` before the one-shot run.

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

## Prior-Frozen Regression Result

The sealed candidate ran the prior corpus exactly once and failed its
conjunctive criterion:

| Measurement | Result |
| --- | ---: |
| Local compatibility positives | 12/12 |
| Local cross-tuple rejections | 60/60 |
| Local intent positives | 12/12 |
| Local negative exact no-ops with full intermediate reach | 36/36 |
| Composition routes and rank-one targets | 10/12 |
| Composition negatives with full intermediate reach and exact no-op | 17/18 |
| Older wrong-intent exact no-ops | 30/30 |
| Missing-tuple exact no-ops | 4/4 |
| Unknown-entity base-address acceptance followed by veto and no-op | 1/4 |
| Unknown-relation exact no-ops | 4/4 |

The two positive failures were both Bellatrix/color. On the `ledger` form,
association addressing selected Bellatrix and the Bellatrix prototype was the
nearest verifier, but its radial or contrastive condition rejected the request;
the unchanged target ranked 504. On the `resolve` form, association addressing
proposed Arcturus and its verifier was nearest, so the identity-confirming gate
correctly refused to substitute Bellatrix; the unchanged target ranked 3191.

The `17/18` and `1/4` values above are stricter reach-qualified diagnostics in
the sealed harness, not counts of unauthorized state changes. A composition
negative rejected before intent does not satisfy that diagnostic even though it
abstains, and an unknown rejected by base entity addressing does not satisfy the
"accepted then vetoed" diagnostic. Every printed unknown-entity case reported
`known_entity=no` and `applied=no`; the fixed intent artifact cannot apply an
action on a path that stops upstream. The decisive preregistered failure is the
10/12 positive route and rank-one result, so no ambiguity in those diagnostic
subcounts changes the stop decision.

The candidate is retained unchanged and may not be tuned against either
Bellatrix observation.

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

Atomic persistence remains deferred. Label conditioning cleanly separates its
development corpus but does not solve the frozen entity-address recall boundary:
one Bellatrix request fails verification and another fails before verification.
A materially different approach requires a fresh preregistration. The untouched
local and composition sets above remain unevaluated.
