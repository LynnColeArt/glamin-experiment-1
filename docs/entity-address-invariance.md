# Entity-Address Invariance Probe

## Status

Frozen evaluation complete. The entity-specific hypothesis passed, but the
preregistered end-to-end criterion failed at 9/18 routes.

## Motivation

The two-stage target-state probe reproduced the intended target on every
authorized frozen prompt, but two compact Arcturus prompts failed the entity
gate. This probe tests whether the entity projection, rather than the later
action, is the remaining end-to-end boundary.

The previous two-stage evaluation prompts remain immutable. They are a known
regression set, not development or independent evaluation evidence.

## Controlled Comparison

Two entity memories are built from exactly the same construction views,
development views, calibration negatives, labels, and query width:

1. the existing maximum-variance coordinate projection; and
2. an association-signal projection that favors coordinates separating entity
   groups while suppressing within-entity surface-form variation.

Both candidates use the unchanged association-signal relation memory, exact
tuple ledger, tuple-local action memory, action gate, target-state ledger,
tensor locations, and full-state interpolation from the preceding probe. Only
the entity projection strategy and independently calibrated entity gate differ.

## Development Split

Each of the six registered tuples has three development forms, for 18 prompts:

- prose: `From retained memory, return the REL registered for object <ENTITY>.`
- record: `Object record { name: "ENTITY", field: "REL" }`
- path: `memory/object/ENTITY/attribute/REL/value=`

These forms may calibrate only the candidate entity gates. Before the frozen
split is run, the association-signal system must accept and identify the correct
entity on all 18. End-to-end routing and rank are development diagnostics rather
than prerequisites because the downstream gates are deliberately unchanged.
The existing 12 wrong-intent controls and two missing-tuple controls must remain
exact no-ops.

## Frozen Evaluation Split

Each registered tuple has three previously unused forms, for 18 positive
prompts:

- question: `Which memorized REL is attached to the object named ENTITY?`
- card: `Lookup card [object=ENTITY][attribute=REL]`
- URI: `vault://records/ENTITY?field=REL#value=`

Six additional controls apply the same three forms to the known-negative entity
names `Rigel` and `Sirius`. Each must fail the entity gate and leave logits
exactly unchanged.

The frozen success criterion is all of the following:

- 18/18 association-signal prompts route to the correct entity, relation, and
  reviewed tuple;
- 18/18 produce the stored target as the rank-one next token through the later
  target-state action; and
- 6/6 unknown-entity controls are exact no-ops rejected by the entity gate.

Variance-projection results are recorded on the same prompts as a controlled
baseline but are not part of the pass condition. Only after the new frozen run
is complete is the association-signal system measured on the earlier two-stage
prompts; that measurement is a known regression result, not fresh evidence.

No prompt, representation, threshold, negative set, or success criterion may
change after the first frozen run. A failure is recorded as a failure and must
motivate a separate experiment.

## Development Preflight

The initial implementation mistakenly rebuilt the downstream action gate with
the new development forms. Construction stopped before frozen evaluation
because maximum development distance `0.133199` crossed the wrong-intent
boundary `0.097498`. The implementation was corrected to reuse the preceding
probe's relation and action memories, as required by the controlled comparison.

With downstream gates held fixed, both entity projections identified 18/18
development entities. Both routed 12/18 end to end; all six path forms reached
the correct entity and relation but failed the unchanged action gate. This
development result was used only to correct the preflight boundary from
end-to-end success to entity-identification success. The frozen prompts and
frozen success criterion were not run or changed.

## Frozen Result

The preregistered end-to-end criterion **failed**:

| Measurement | Variance | Association signal |
| --- | ---: | ---: |
| Correct entity acceptance | 13/18 | 18/18 |
| Correct end-to-end routes | 9/18 | 9/18 |
| Rank-one stored targets | 9/18 | 9/18 |

Association-signal entity distance was at most `0.999444` on positives, within
the calibrated `1.10063` radius. All six unknown-entity controls were rejected
with exactly unchanged logits; their nearest distance was at least `1.13138`.
The method therefore achieved 18/18 positive entity identification and 6/6
negative entity abstention on this split. It also repaired the earlier known
regression set from 10/12 to 12/12 routes and rank-one targets.

The nine end-to-end failures occurred strictly downstream:

- all six URI forms passed the entity and relation gates but failed the
  unchanged action-context gate; and
- the three material-question forms passed the entity gate but failed the
  unchanged relation gate.

Every authorized later target-state action produced the stored target at rank
one, 9/9, and reproduced its canonical teacher logits within `2.86102e-06`.

## Interpretation

Association-signal projection solved the entity-address boundary measured by
this experiment, but did not satisfy the broader end-to-end criterion. The
controlled failure localizes the next work: relation and action-context
representations must generalize across compact forms without weakening the
existing wrong-intent boundary. The entity projection, exact tuple ledger, and
later target-state action should now remain fixed while those two gates are
tested on a new split.
