# Documentation

## Vision and Scope

- [Project overview](project-overview.md) explains the neural-geometric runtime,
  executable memory, live geometry, and research hypothesis.
- [Architecture](architecture.md) defines component responsibilities and the
  intended inference data flow.
- [Roadmap](roadmap.md) breaks the work into falsifiable milestones.

## Contracts

- [Glamin C ABI](c-abi.md) distinguishes the implemented runtime, persistent
  index, and generation boundary from request and trace proposals.
- [Generation semantics](generation-semantics.md) defines mounting, activation,
  pinning, retirement, and rollback.
- [Fixed model hook](model-hook.md) records the implemented llama.cpp tensor
  callback, model contract, failure semantics, and observed Qwen3 result.
- [Persistent hook artifact](hook-artifact.md) defines the hash-bound model,
  projection, Glamin space, residual, publication, and pinning contract.
- [Activation-derived associative memory](activation-memory.md) records the
  scaled eight-association held-out recall and abstention experiment.
- [Multi-relation addressing probe](multi-relation-probe.md) records the
  calibrated boundary exposed by overlapping entity–relation tuples.
- [Factorized entity–relation memory](factorized-memory.md) records the bounded
  two-space tuple join, contextual action gate, and frozen-Qwen result.
- [Safety model](safety-model.md) separates information from executable
  authority and records the initial trust assumptions.

## Development

- [Development guide](development.md) contains build, test, and repository
  conventions.
- [Experiment definition](../EXPERIMENT.md) specifies the first demonstration
  and its success criteria.

## Current Status

The repository contains a C++17 reference runtime, a real C++ wrapper over
Glamin ABI version 4, and a pinned llama.cpp hidden-state adapter. The native
path searches Fortran-owned indexes, loads compiler-emitted persistent
artifacts, mounts complete hash-bound hook generations, and injects their
generation-qualified residuals into a live Qwen3 graph. Eight activation-derived
associations now produce their intended next tokens across 16 held-out prompts
while 16 controls abstain exactly. A second in-inference hook independently
retrieves entity and relation evidence, authorizes exact tuples, and selects a
bounded contextual action: 18 registered contexts pass at rank one, while two
missing known-factor tuples and 12 unfamiliar contexts abstain exactly. Broad
multi-relation paraphrase generalization, persisted factorized artifacts,
trained adapters, durable traces, and external procedural actions remain future
work.
