# Roadmap

## Phase 0: Definition

- [x] Record the project hypothesis and non-goals.
- [x] Document the proposed architecture.
- [x] Define generation semantics and the initial safety boundary.
- [x] Draft the Glamin C ABI required by the experiment.

## Phase 1: Reference Generation Runtime

- [x] Scaffold a C++17 CMake project.
- [x] Implement immutable reference generations and pins.
- [x] Demonstrate live activation and rollback.
- [x] Test that old pins survive activation of a new generation.

Exit criterion: the observable generation contract is executable and tested
without claiming Glamin integration.

## Phase 2: Glamin C ABI

- [x] Review the ABI proposal against Glamin's internal types.
- [x] Implement runtime and diagnostic lifecycle functions in Glamin.
- [x] Implement runtime-owned flat-index add, search, and result functions.
- [x] Implement immutable generation and pin functions with synchronous search.
- [x] Implement contract-validated persistent flat artifact loading.
- [ ] Implement request and asynchronous traversal functions.
- [x] Add initial C conformance tests and C++ RAII wrappers.
- [x] Run the Phase 1 generation semantics against the Glamin backend.

Exit criterion: the C++ demo performs real Glamin traversal and hot generation
swaps through the public C ABI.

## Phase 3: Fixed Model Hook

- [x] Pin one `llama.cpp` revision and one small model architecture.
- [x] Capture a hidden state at one fixed transformer layer.
- [x] Define and record the experimental model/layer/adapter space contract.
- [x] Traverse Glamin once per request.
- [x] Inject a gated residual and measure logits and output.
- [x] Persist and hash-bind the model contract, projection, Glamin space, and
  residual payloads as one atomically activated hook generation.

Exit criterion: switching between two geometry generations predictably changes
model behavior without changing weights or restarting the process.

Observed: Qwen3 generation A -> B changed the top token from ` the` to ` for`;
A rollback reproduced the baseline logits exactly in the same process.

## Phase 4: Training and Evaluation

- [x] Derive an input projection, view-conditioned residuals, and a calibrated
  memory gate without updating model weights.
- [x] Demonstrate two activation-derived memories with distance abstention and
  unchanged unrelated controls.
- [x] Evaluate held-out prompt-prefix paraphrases and explicit hard negatives.
- [x] Evaluate held-out natural questions with entity-token addressing.
- [x] Select inference-time address positions by scanning all token states.
- [x] Persist the all-token selection policy in the hook artifact.
- [x] Derive paired-view key positions without entity annotations.
- [x] Generalize construction to three views and eight associations with
  held-out retrieval and natural-question phrasing.
- [x] Probe two overlapping relations per entity and record the single-vector
  addressing and abstention boundary.
- [x] Implement factorized entity/relation evidence, an exact tuple join, and a
  contextual action-distance gate inside inference.
- [ ] Generalize relation and action-context representations to frozen,
  previously unseen paraphrase sets.
- [ ] Persist and live-swap the complete factorized generation atomically.
- [ ] Add memory-disabled, generation-A, and generation-B baselines.
- [x] Measure artifact construction time, artifact size, inference latency, and
  exact abstention on a 16-prompt negative set.
- [ ] Measure broader behavioral specificity and general-capability retention.
- [ ] Add boundary probes, trace replay, and geometry-diff gates.

## Phase 5: Persistent Traces and Exact Memory

- [ ] Persist trace segments across inference requests.
- [ ] Add an immutable source ledger and episodic document space.
- [ ] Add explicit document-to-concept and concept-to-geometry transforms.
- [ ] Test cross-turn continuity with the model KV cache discarded between turns.

## Phase 6: Controlled Procedural Memory

- [ ] Package a bounded mini-app with source, vectors, contract, traces, and
  signature.
- [ ] Mount, inspect, retire, transplant, and roll back the mini-app live.
- [ ] Introduce a capability broker in proposal-only mode.
- [ ] Evaluate promotion from repeated traces to reviewed procedural geometry.

## Deferred

- Token-level Glamin traversal
- Shared GPU-resident hidden states and geometry
- Multiple model architectures
- Automatic online geometry authoring
- Unattended external actions
- Removal of the within-response KV cache
