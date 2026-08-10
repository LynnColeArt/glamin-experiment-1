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
- [Entity-address invariance](entity-address-invariance.md) preregisters the
  variance-versus-association-signal entity projection comparison.
- [Gate-local authorization invariance](gate-local-authorization-invariance.md)
  records the preregistered relation prototypes, contrastive action
  authorization, independent composition test, and frozen result.
- [Conjunctive retrieval authorization](conjunctive-retrieval-authorization.md)
  records separate tuple-compatibility and retrieval-intent gates, passing
  local frozen results, and a narrowly failing one-shot composed evaluation.
- [Composition-stable retrieval authorization](composition-stable-retrieval-authorization.md)
  records a fresh replication whose development passed but whose known-entity
  veto failed one prior-frozen positive, stopping before new frozen evaluation.
- [Label-conditioned entity knownness](label-conditioned-entity-knownness.md)
  records a materially different verifier whose development passed but whose
  protocol stopped before regression on an Altair corpus-provenance collision.
- [Corpus-disjoint conditioned knownness](corpus-disjoint-conditioned-knownness.md)
  records a clean replication whose fixed width passed development with four
  repository-audited unknowns and is sealed for one-shot regression.
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
bounded contextual action. Its projected-action follow-up passes all 30
registered contexts at rank one and preserves exact abstention for two missing
known-factor tuples. A frozen unseen-paraphrase set produced 4/12 rank-one
recalls. A subsequent association-signal relation projection accepted the
intended relation on 12/12 new frozen prompts, while its deliberately strict
action gate produced 0/12 end-to-end recalls. Six tuple-matched wrong-intent
controls reached that gate and abstained exactly, exposing gate-local
calibration as the next boundary. Topology-aware action calibration now routes
12/12 further frozen paraphrases while 12/12 tuple-matched wrong intents and two
missing tuples remain exact no-ops. The selected residual produces rank-one
answers on 9/12. A two-stage follow-up authorizes at `l_out-34` and applies a
reviewed target state at `l_out-35`: it reached rank one on all 10 authorized
frozen prompts, compared with 6/10 for residuals, but missed its 12/12 criterion
because two compact Arcturus forms failed the upstream entity gate. Persisted
factorized artifacts, trained adapters, durable traces, and external procedural
actions remain future work. An entity-address comparison then selected 18/18
positive entities with association signal versus 13/18 with variance and
rejected 6/6 unknown entities exactly. End-to-end routing remained 9/18 because
the unchanged relation gate rejected three material questions and the action
gate rejected all six URI forms. The gate-local follow-up then passed its
relation stage at 12/12 positives and 4/4 unknown-relation rejections.
Authorization signal reached 11/12 positives but rejected only 5/12 fresh wrong
intents; composition reached 9/12 and failed the frozen authority criterion. A
subsequent conjunctive probe passed both independent local gates: compatibility
accepted 12/12 positives and rejected 60/60 cross-tuples, while intent accepted
12/12 positives and rejected 36/36 eligible negatives exactly. Composition
reached 11/12 rank-one targets and 17/18 new negative no-ops, but one
counterfactual was authorized and the existing entity factor remained 3/4 on
unknowns. Persistence therefore remains deferred.
The composition-stability replication then repaired the prior counterfactual
and all four unknown-entity controls while preserving every older no-op, but its
knownness veto rejected one registered Bellatrix form. It stopped at 11/12
prior-frozen composition positives; no new frozen prompt was evaluated.
The next protocol changed only knownness representation, assigning each label
its own positive and matched-negative prototypes under a shared projection. It
passed development, then stopped before candidate regression when provenance
review found that development had reused the frozen unknown label Altair.
The clean follow-up freezes width 32 and substitutes four labels absent from all
earlier corpora. Its development preflight passed every criterion; candidate
regression and new frozen evaluation remain unrun.
