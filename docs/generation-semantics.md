# Geometry Generation Semantics

## Goal

Live geometry must change without exposing an active traversal to a partially
updated manifold. The runtime therefore treats a published geometry generation
as immutable.

## Terms

- **Generation:** A complete, immutable executable-geometry snapshot with a
  stable runtime identity and contract manifest.
- **Mounted:** Validated and available within a runtime, but not necessarily
  selected for new work.
- **Active:** The generation selected when a caller asks to pin the current
  geometry.
- **Pin:** A lease binding a request or trace segment to one generation.
- **Retired:** Ineligible for new pins but retained while existing pins remain.
- **Reclaimed:** Unloaded after retirement and release of the last pin.

## State Machine

```text
             validate
created ----------------> mounted
                            |
                            | activate
                            v
                          active <--------+
                            |             |
                            | supersede   | rollback/activate
                            v             |
                          mounted --------+
                            |
                            | retire
                            v
                          retired
                            |
                            | last pin released
                            v
                         reclaimed
```

Activation does not mutate either generation. It atomically changes the active
generation reference used by future pin operations.

## Required Invariants

1. A pin resolves to exactly one generation.
2. A generation cannot change after its first successful mount.
3. Activating B does not change the result of a traversal pinned to A.
4. Retirement does not invalidate existing pins.
5. Reclamation cannot occur while a pin exists.
6. A trace records the generation used for every traversal segment.
7. Rollback is activation of a previously mounted compatible generation, not
   mutation of the current generation.
8. Contract validation completes before a generation becomes mountable.

## Implementations

The C++ reference implementation models these rules with immutable generation
objects held by shared leases. The runtime serializes installation, activation,
and pin acquisition. Traversal occurs outside the runtime lock against the
pinned immutable object.

Glamin ABI version 3 introduced the same observable lifecycle for real flat
indexes, and ABI version 4 retains it. Generation creation freezes the bound
index. Activation affects future pins, retired generations remain searchable
through existing pins, and final unpin permits reclamation. The C++ adapter
exposes those handles through RAII generation stores and pins.

The native registry operations remain externally serialized in ABI version 4.
Atomic activation here means an indivisible observable transition in that
serialized call order, not yet a lock-free or concurrently callable operation.

## Active Trace Migration

Experiment 1 does not migrate a trace between generations. A trace segment
finishes on its pinned generation. A later segment may deliberately pin a newer
generation and record the transition.

Future migration requires:

- A compatible space and model contract.
- A named, versioned state transform.
- Validation against boundary and canonical-trace probes.
- An explicit trace event identifying the source and destination generation.

Silent migration is forbidden.

## Failure Handling

- Mount failure leaves the active generation unchanged.
- Activation of an unknown or retired generation fails.
- A traversal without a valid pin fails before reading geometry.
- A failed traversal does not corrupt its generation or trace.
- Process recovery selects a known-good generation from durable deployment
  metadata; it does not infer the active generation from whichever files exist.
