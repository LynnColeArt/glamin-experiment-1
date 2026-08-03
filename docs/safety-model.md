# Safety Model

## Scope

This document establishes the initial trust boundary for executable memory.
It is a design constraint, not a complete production threat model.

## Central Rule

> Information does not imply authority.

A document, retrieved memory, model output, or nearby vector may influence
reasoning. None may grant itself permission to perform an external effect.

## Trust Domains

| Domain | Examples | Default authority |
| --- | --- | --- |
| Untrusted information | Documents, messages, web content, retrieved episodes | Read and reason only |
| Model computation | Hidden states, generated tokens, proposed actions | Propose only |
| Executable geometry | Signed mini-apps, mints, corridors, traces | Traverse within declared contract |
| Capability broker | Filesystem, network, process, device, account actions | Enforce explicit grants |
| Control plane | Generation signing, activation, retirement, rollback | Administrative |

## Promotion Boundary

An episode cannot directly become executable geometry. Promotion follows an
explicit pipeline:

```text
episode
  -> candidate concept
  -> proposed procedure
  -> human-readable geometry specification
  -> validation and capability review
  -> probe and trace evaluation
  -> signature
  -> mounted generation
```

Automating stages later does not remove the boundary or its artifacts.

## Required Controls

- Separate document, semantic, and executable spaces.
- Versioned and deterministic cross-space transforms.
- Signed geometry manifests and verified embedder/model contracts.
- Explicit capability manifests for every mini-app that can propose effects.
- Least-privilege capability grants scoped to a request or trace.
- Immutable generation publication and rapid rollback.
- Append-only traversal and action audit records.
- Boundary probes that include abstention and adversarial inputs.
- No direct execution of text recovered from document memory.

## Initial Experiment Restrictions

Experiment 1 has no external actions, network access, online memory writes, or
automatic geometry authoring. Its reference behaviors are deterministic numeric
transforms compiled into the test process.

The first model experiment must remain equally constrained: Glamin may return a
residual vector, but it may not execute a system capability.

## Principal Risks

### Memory-to-code injection

Untrusted content could attempt to represent itself as an instruction. Hard
space separation and signed compilation prevent document writes from becoming
executable geometry.

### Behavioral drift

A geometry update may alter decisions beyond its intended neighborhood. Each
generation needs geometry, boundary, and canonical-trace diffs before activation.

### Stale representation

Hidden-state geometry may become invalid after a model, quantization, hook
layer, or adapter change. Contracts bind geometry to the exact representation
artifacts and reject incompatible loads.

### Confused authority

A valid mini-app may be invoked in a context with broader tools than intended.
The capability broker evaluates grants independently at execution time.

### Irreproducible intervention

Live editing can make behavior difficult to reconstruct. Generation identity,
activation events, pins, model contracts, and results must be present in the
trace.

## Non-Claims

This architecture does not make model reasoning inherently truthful, safe, or
interpretable. It makes some memories and procedures more addressable and
governable than undifferentiated weight updates. That is useful, but incomplete.
