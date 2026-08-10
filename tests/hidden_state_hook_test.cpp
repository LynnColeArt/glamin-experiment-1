#include "gx1/factorized_memory_hook.hpp"
#include "gx1/hidden_state_hook.hpp"
#include "gx1/hook_artifact.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gx1::HiddenStateHookConfig identity_config() {
    return gx1::HiddenStateHookConfig{
        2,
        2,
        {1.0F, 0.0F, 0.0F, 1.0F},
        0.25F,
        gx1::ProjectionNormalization::none,
    };
}

class HookArtifactFixture final {
public:
    HookArtifactFixture(
        std::string name,
        const float residual_sign,
        const std::size_t glamin_vector_count = 2,
        const gx1::AddressSelectionPolicy address_selection =
            gx1::AddressSelectionPolicy::all_token_rows)
        : directory_(std::filesystem::current_path() / std::move(name)) {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        const auto glamin = directory_ / "glamin";
        if (!std::filesystem::create_directories(glamin)) {
            throw std::runtime_error("failed to create hook artifact fixture");
        }

        const auto vector_bytes = glamin_vector_count * 2U * sizeof(float);
        std::ofstream layout(glamin / "vector_layout.json");
        layout << "{\"dtype\": \"float32\",\"endianness\": \"little\","
                  "\"total_vectors\": "
               << glamin_vector_count << ",\"total_bytes\": " << vector_bytes
               << ",\"spaces\":[{\"space_id\": \"fixture.space\",\"dim\": 2,"
                  "\"count\": "
               << glamin_vector_count
               << ",\"byte_stride\": 8,\"offset_bytes\": 0}]}\n";
        layout.close();

        std::vector<float> vectors(glamin_vector_count * 2U, 0.0F);
        if (glamin_vector_count > 1) {
            vectors[2] = 2.0F;
            vectors[3] = 2.0F;
        }
        std::ofstream vector_file(glamin / "vectors.bin", std::ios::binary);
        vector_file.write(
            reinterpret_cast<const char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
        vector_file.close();

        std::ofstream contracts(glamin / "contracts.json");
        contracts
            << "{\"spec_id\": \"fixture\",\"embedder\": {\"spec\": {"
               "\"id\": \"fixture\",\"version\": \"1\","
               "\"input_schema\": \"f32\",\"preprocess_chain\": [\"none\"],"
               "\"model_hash\": \"sha256:model\","
               "\"config_hash\": \"sha256:config\","
               "\"hardware_class\": \"cpu\",\"min_ram_mb\": 0,"
               "\"min_vram_mb\": 0},\"contract_hash\": \"sha256:embed\","
               "\"signature\": \"\"},\"spaces\":[{\"spec\": {"
               "\"space_id\": \"fixture.space\",\"dim\": 2,"
               "\"metric\": \"l2\",\"normalization\": \"none\"},"
               "\"contract_hash\": "
               "\"sha256:c761b2699931aba1583df083fa99155e82b3ad08545bcb2e2ce3521034f80f88\","
               "\"signature\": \"\"}]}\n";
        contracts.close();

        gx1::HookArtifactSpec spec;
        spec.model = model_contract();
        spec.query_dimension = 2;
        spec.metric = gx1::GlaminMetric::l2;
        spec.query_normalization = gx1::ProjectionNormalization::none;
        spec.gate = 0.5F;
        spec.maximum_distance = 0.75F;
        spec.address_selection = address_selection;
        spec.input_projection = {1.0F, 0.0F, 0.0F, 1.0F};
        spec.residual_labels = {0, 1};
        spec.residuals = {
            residual_sign * 2.0F, 0.0F,
            0.0F, residual_sign * 4.0F,
        };
        spec.glamin_space_id = "fixture.space";
        gx1::write_hook_artifact(directory_, spec);
    }

    ~HookArtifactFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    HookArtifactFixture(const HookArtifactFixture&) = delete;
    HookArtifactFixture& operator=(const HookArtifactFixture&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return directory_;
    }

    [[nodiscard]] static gx1::ModelHookContract model_contract() {
        return gx1::ModelHookContract{
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "llama-fixture-revision",
            "fixture-model",
            "l_out-1",
            2,
        };
    }

    void corrupt(const std::filesystem::path& relative_path) const {
        std::fstream file(directory_ / relative_path, std::ios::binary | std::ios::in |
                                                     std::ios::out);
        char byte = 0;
        file.read(&byte, 1);
        byte = static_cast<char>(static_cast<unsigned char>(byte) ^ 0x01U);
        file.seekp(0);
        file.write(&byte, 1);
    }

private:
    std::filesystem::path directory_;
};

void test_generation_qualified_residual_and_stable_pin() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation_a = generations.mount_flat("hook-a", 2, {0.0F, 0.0F});
    const auto generation_b = generations.mount_flat("hook-b", 2, {0.0F, 0.0F});

    auto payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    payloads->insert(generation_a, 0, {2.0F, 0.0F});
    payloads->insert(generation_b, 0, {-2.0F, 0.0F});

    generations.activate(generation_a);
    gx1::FixedLayerMemoryHook hook_a(
        generations.pin_active(),
        identity_config(),
        payloads);
    std::vector<float> hidden_a{0.0F, 1.0F};
    const auto result_a = hook_a.apply(hidden_a);
    expect(result_a.generation == generation_a, "hook A used the wrong generation");
    expect(result_a.label == 0, "hook A used the wrong payload label");
    expect(hidden_a == std::vector<float>({0.5F, 1.0F}), "hook A residual mismatch");

    generations.activate(generation_b);
    gx1::FixedLayerMemoryHook hook_b(
        generations.pin_active(),
        identity_config(),
        payloads);
    std::vector<float> hidden_b{0.0F, 1.0F};
    const auto result_b = hook_b.apply(hidden_b);
    expect(result_b.generation == generation_b, "hook B used the wrong generation");
    expect(hidden_b == std::vector<float>({-0.5F, 1.0F}), "hook B residual mismatch");

    std::vector<float> old_pin_hidden{0.0F, 1.0F};
    static_cast<void>(hook_a.apply(old_pin_hidden));
    expect(
        old_pin_hidden == std::vector<float>({0.5F, 1.0F}),
        "generation activation changed an existing hook pin");
}

void test_l2_projection_and_missing_payload_fail_closed() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat(
        "normalized-hook",
        2,
        {1.0F, 0.0F, 0.0F, 1.0F});
    generations.activate(generation);

    auto payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    payloads->insert(generation, 1, {0.0F, 4.0F});
    auto config = identity_config();
    config.query_normalization = gx1::ProjectionNormalization::l2;
    gx1::FixedLayerMemoryHook hook(
        generations.pin_active(),
        std::move(config),
        payloads);

    std::vector<float> hidden{3.0F, 4.0F};
    const auto result = hook.apply(hidden);
    expect(result.label == 1, "normalized projection selected the wrong neighbor");
    expect(hidden == std::vector<float>({3.0F, 5.0F}), "normalized residual mismatch");

    auto missing_payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    gx1::FixedLayerMemoryHook missing_hook(
        generations.pin_active(),
        identity_config(),
        missing_payloads);
    std::vector<float> unchanged{3.0F, 4.0F};
    try {
        static_cast<void>(missing_hook.apply(unchanged));
        throw std::runtime_error("missing residual payload was accepted");
    } catch (const std::out_of_range&) {
        expect(
            unchanged == std::vector<float>({3.0F, 4.0F}),
            "failed memory hook partially changed the hidden state");
    }
}

void test_memory_distance_abstention() {
    gx1::GlaminRuntime runtime(1);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat("abstention", 2, {0.0F, 0.0F});
    generations.activate(generation);

    auto payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    payloads->insert(generation, 0, {20.0F, 20.0F});
    auto config = identity_config();
    config.maximum_distance = 0.5F;
    gx1::FixedLayerMemoryHook hook(
        generations.pin_active(), std::move(config), payloads);

    std::vector<float> hidden{1.0F, 1.0F};
    const auto result = hook.apply(hidden);
    expect(!result.applied, "distant memory did not abstain");
    expect(result.gate == 0.0F, "abstained memory reported a nonzero gate");
    expect(hidden == std::vector<float>({1.0F, 1.0F}),
           "abstained memory changed the hidden state");
}

void test_automatic_nearest_address_candidate() {
    gx1::GlaminRuntime runtime(1);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat(
        "automatic-address", 2, {0.0F, 0.0F, 10.0F, 10.0F});
    generations.activate(generation);

    auto payloads = std::make_shared<gx1::ResidualPayloadLedger>();
    payloads->insert(generation, 0, {4.0F, 0.0F});
    payloads->insert(generation, 1, {0.0F, 4.0F});
    gx1::FixedLayerMemoryHook hook(
        generations.pin_active(), identity_config(), payloads);

    const std::vector<std::vector<float>> candidates{
        {5.0F, 5.0F},
        {0.5F, 0.5F},
        {9.9F, 9.9F},
    };
    std::vector<float> hidden{1.0F, 1.0F};
    const auto result = hook.apply_nearest(candidates, hidden);
    expect(result.address_candidate == 2U,
           "automatic addressing selected the wrong candidate state");
    expect(result.label == 1U,
           "automatic addressing selected the wrong memory row");
    expect(hidden == std::vector<float>({1.0F, 2.0F}),
           "automatic addressing applied the wrong residual payload");
}

void test_prefix_mean_address_candidates() {
    const auto means = gx1::prefix_mean_states({
        {2.0F, 4.0F},
        {4.0F, 8.0F},
        {6.0F, 12.0F},
    });
    expect(
        means == std::vector<std::vector<float>>({
                     {2.0F, 4.0F},
                     {3.0F, 6.0F},
                     {4.0F, 8.0F},
                 }),
        "prefix-mean addressing returned the wrong cumulative states");
}

void test_factorized_tuple_join_and_abstention() {
    gx1::GlaminRuntime runtime(1);
    gx1::GlaminGenerationStore generations(runtime);
    const auto entity_generation = generations.mount_flat(
        "entities", 2, {1.0F, 0.0F, 10.0F, 0.0F});
    const auto relation_generation = generations.mount_flat(
        "relations", 2, {0.0F, 1.0F, 0.0F, 10.0F});

    generations.activate(entity_generation);
    auto entity_pin = generations.pin_active();
    generations.activate(relation_generation);
    auto relation_pin = generations.pin_active();

    const gx1::FactorSearchConfig config{
        2,
        2,
        {1.0F, 0.0F, 0.0F, 1.0F},
        gx1::ProjectionNormalization::none,
        0.1F,
    };
    auto payloads = std::make_shared<gx1::TupleResidualLedger>();
    payloads->insert_variant(100U, 8U, {3.0F}, {2.0F, 4.0F});
    payloads->insert_variant(100U, 8U, {30.0F}, {20.0F, 40.0F});
    gx1::FactorizedLayerMemoryHook hook(
        std::move(entity_pin),
        config,
        {100U, 200U},
        std::move(relation_pin),
        config,
        {7U, 8U},
        0.5F,
        payloads,
        0.5F,
        gx1::FactorSearchConfig{
            2,
            1,
            {1.0F, 0.0F},
            gx1::ProjectionNormalization::none,
            0.5F,
        });

    const std::vector<std::vector<float>> candidates{
        {1.0F, 0.0F},
        {0.0F, 10.0F},
        {50.0F, 50.0F},
    };
    const auto authorized = hook.authorize_nearest(
        candidates, std::vector<float>{3.0F, 4.0F});
    expect(authorized.entity.factor_label == 100U &&
               authorized.relation.factor_label == 8U &&
               authorized.action_accepted && !authorized.applied &&
               authorized.gate == 0.0F,
           "authorization-only factorized join reported the wrong state");

    std::vector<float> hidden{3.0F, 4.0F};
    const auto joined = hook.apply_nearest(candidates, hidden);
    expect(joined.entity.factor_label == 100U,
           "factorized join selected the wrong entity");
    expect(joined.entity.address_candidate == 0U,
           "factorized join selected the wrong entity token");
    expect(joined.relation.factor_label == 8U,
           "factorized join selected the wrong relation");
    expect(joined.relation.address_candidate == 1U,
           "factorized join selected the wrong relation token");
    expect(joined.tuple_found && joined.applied,
           "reviewed factor tuple did not apply");
    expect(joined.action_accepted && joined.action_variant == 0U &&
               joined.action_distance == 0.0F,
           "factorized join selected the wrong contextual action");
    expect(hidden == std::vector<float>({4.0F, 6.0F}),
           "factorized tuple applied the wrong residual");

    std::vector<float> unfamiliar_hidden{4.0F, 4.0F};
    const auto unfamiliar = hook.apply_nearest(
        {{1.0F, 0.0F}}, {{0.0F, 10.0F}}, unfamiliar_hidden);
    expect(unfamiliar.tuple_found && !unfamiliar.action_accepted &&
               !unfamiliar.applied,
           "unfamiliar action context did not abstain");
    expect(unfamiliar_hidden == std::vector<float>({4.0F, 4.0F}),
           "rejected action context changed the hidden state");

    std::vector<float> missing_hidden{3.0F, 4.0F};
    const auto missing = hook.apply_nearest(
        {{10.0F, 0.0F}}, {{0.0F, 10.0F}}, missing_hidden);
    expect(missing.entity.accepted && missing.relation.accepted,
           "known tuple factors were not independently accepted");
    expect(!missing.tuple_found && !missing.applied,
           "unreviewed factor tuple did not abstain");
    expect(missing_hidden == std::vector<float>({3.0F, 4.0F}),
           "unreviewed factor tuple changed the action state");

    std::vector<float> distant_hidden{3.0F, 4.0F};
    const auto distant = hook.apply_nearest(
        {{50.0F, 50.0F}}, {{0.0F, 10.0F}}, distant_hidden);
    expect(!distant.entity.accepted && distant.relation.accepted,
           "factor distance gates reported the wrong evidence state");
    expect(!distant.applied &&
               distant_hidden == std::vector<float>({3.0F, 4.0F}),
           "rejected factor evidence changed the action state");

    gx1::TupleTargetStateLedger target_states;
    target_states.insert(100U, 8U, {9.0F, 8.0F});
    expect(target_states.size() == 1U &&
               target_states.find(100U, 8U) != nullptr &&
               *target_states.find(100U, 8U) ==
                   std::vector<float>({9.0F, 8.0F}) &&
               target_states.find(200U, 8U) == nullptr,
           "tuple target-state ledger returned the wrong payload");

    bool duplicate_rejected = false;
    try {
        target_states.insert(100U, 8U, {1.0F, 2.0F});
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected,
           "tuple target-state ledger accepted a duplicate tuple");
}

void test_factorized_authorization_conjoins_compatibility_and_intent() {
    gx1::GlaminRuntime runtime(1);
    gx1::GlaminGenerationStore generations(runtime);
    const auto entity_generation = generations.mount_flat(
        "conjunctive-entities", 1, {1.0F});
    const auto relation_generation = generations.mount_flat(
        "conjunctive-relations", 1, {1.0F});
    generations.activate(entity_generation);
    auto entity_pin = generations.pin_active();
    generations.activate(relation_generation);
    auto relation_pin = generations.pin_active();

    const gx1::FactorSearchConfig factor_config{
        2,
        1,
        {1.0F, 0.0F},
        gx1::ProjectionNormalization::none,
        0.1F,
    };
    auto payloads = std::make_shared<gx1::TupleResidualLedger>();
    payloads->insert_variant(10U, 20U, {3.0F}, {1.0F, -1.0F});
    gx1::FactorizedLayerMemoryHook hook(
        std::move(entity_pin),
        factor_config,
        {10U},
        std::move(relation_pin),
        factor_config,
        {20U},
        1.0F,
        payloads,
        0.1F,
        gx1::FactorSearchConfig{
            2,
            1,
            {1.0F, 0.0F},
            gx1::ProjectionNormalization::none,
            0.1F,
        },
        gx1::RetrievalIntentGateConfig{
            gx1::FactorSearchConfig{
                2,
                1,
                {0.0F, 1.0F},
                gx1::ProjectionNormalization::none,
                0.1F,
            },
            {4.0F},
        },
        true);

    const std::vector<std::vector<float>> factors{{1.0F, 0.0F}};
    const std::vector<std::vector<float>> scannable_factors{
        {1.0F, 0.0F}, {3.0F, 0.0F}};
    const auto accepted = hook.authorize_nearest(
        scannable_factors,
        scannable_factors,
        std::vector<float>{9.0F, 4.0F});
    expect(accepted.tuple_found && accepted.compatibility_accepted &&
               accepted.intent_accepted && accepted.action_accepted,
           "conjunctive authorization rejected a compatible retrieval");
    expect(accepted.compatibility_distance == 0.0F &&
               accepted.intent_distance == 0.0F,
           "conjunctive authorization recorded the wrong distances");

    const auto denied = hook.authorize_nearest(
        scannable_factors,
        scannable_factors,
        std::vector<float>{9.0F, 9.0F});
    expect(denied.tuple_found && denied.compatibility_accepted &&
               !denied.intent_accepted && !denied.action_accepted,
           "retrieval-intent rejection did not veto compatibility");
    expect(denied.compatibility_distance == 0.0F &&
               denied.intent_distance == 25.0F,
           "intent veto did not retain independent diagnostics");

    const auto incompatible = hook.authorize_nearest(
        factors, factors, std::vector<float>{9.0F, 4.0F});
    expect(incompatible.tuple_found &&
               !incompatible.compatibility_accepted &&
               incompatible.intent_accepted && !incompatible.action_accepted,
           "tuple compatibility rejection hid the independent intent result");
}

void test_persistent_hook_artifact_atomic_activation_and_corruption() {
    expect(
        gx1::sha256_text("abc") ==
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 implementation failed its standard vector");

    HookArtifactFixture fixture_a("gx1_hook_artifact_a", 1.0F);
    HookArtifactFixture fixture_b("gx1_hook_artifact_b", -1.0F);
    HookArtifactFixture fixture_prefix(
        "gx1_hook_artifact_prefix",
        1.0F,
        2U,
        gx1::AddressSelectionPolicy::prefix_mean_rows);
    const auto expected_model = HookArtifactFixture::model_contract();
    const auto loaded = gx1::load_hook_artifact(fixture_a.path(), expected_model);
    expect(loaded.model.target_tensor == "l_out-1", "loaded target tensor mismatch");
    expect(loaded.hook_config.query_dimension == 2, "loaded query dimension mismatch");
    expect(loaded.hook_config.maximum_distance == 0.75F,
           "loaded memory abstention distance mismatch");
    expect(loaded.hook_config.address_selection ==
               gx1::AddressSelectionPolicy::all_token_rows,
           "loaded address-selection policy mismatch");
    const auto loaded_prefix = gx1::load_hook_artifact(
        fixture_prefix.path(), expected_model);
    expect(loaded_prefix.hook_config.address_selection ==
               gx1::AddressSelectionPolicy::prefix_mean_rows,
           "prefix-mean address-selection policy did not round trip");
    expect(loaded.residual_labels == std::vector<std::uint64_t>({0, 1}),
           "loaded residual labels mismatch");

    gx1::GlaminRuntime runtime(2);
    gx1::PersistentHookGenerationStore generations(runtime);
    const auto generation_a = generations.mount("persistent-hook-a", fixture_a.path(), expected_model);
    const auto generation_b = generations.mount("persistent-hook-b", fixture_b.path(), expected_model);
    expect(generations.mounted_generation_count() == 2, "hook bundle mount count mismatch");

    generations.activate(generation_a);
    auto pin_a = generations.pin_active();
    std::vector<float> hidden_a{0.1F, 0.1F};
    const auto result_a = pin_a.memory_hook.apply(hidden_a);
    expect(result_a.generation == generation_a, "persistent hook A pin mismatch");
    expect(hidden_a == std::vector<float>({1.1F, 0.1F}),
           "persistent hook A residual mismatch");

    generations.activate(generation_b);
    auto pin_b = generations.pin_active();
    expect(pin_a.contract_sha256 != pin_b.contract_sha256,
           "different hook payloads produced the same artifact identity");
    std::vector<float> hidden_b{0.1F, 0.1F};
    static_cast<void>(pin_b.memory_hook.apply(hidden_b));
    expect(hidden_b == std::vector<float>({-0.9F, 0.1F}),
           "persistent hook B residual mismatch");
    std::vector<float> stable_a{0.1F, 0.1F};
    static_cast<void>(pin_a.memory_hook.apply(stable_a));
    expect(stable_a == std::vector<float>({1.1F, 0.1F}),
           "activation changed an existing persistent hook pin");

    const auto expect_corruption_rejected = [&generations, &expected_model](
                                                const std::string& fixture_name,
                                                const std::filesystem::path& file) {
        HookArtifactFixture corrupted(fixture_name, 1.0F);
        corrupted.corrupt(file);
        try {
            static_cast<void>(generations.mount("corrupt", corrupted.path(), expected_model));
            throw std::runtime_error("corrupt hook artifact was mounted: " + file.string());
        } catch (const std::invalid_argument&) {
            expect(generations.mounted_generation_count() == 2,
                   "corrupt artifact partially published a hook generation");
            expect(generations.active_generation() != 0,
                   "corrupt artifact changed the active generation");
        }
    };
    expect_corruption_rejected("gx1_hook_artifact_bad_manifest", "hook_manifest.txt");
    expect_corruption_rejected("gx1_hook_artifact_bad_projection", "projection.f32");
    expect_corruption_rejected("gx1_hook_artifact_bad_labels", "residual_labels.u64");
    expect_corruption_rejected("gx1_hook_artifact_bad_residuals", "residuals.f32");
    expect_corruption_rejected("gx1_hook_artifact_bad_vectors", "glamin/vectors.bin");

    auto wrong_model = expected_model;
    wrong_model.target_tensor = "l_out-2";
    try {
        static_cast<void>(generations.mount("wrong-model", fixture_a.path(), wrong_model));
        throw std::runtime_error("model-incompatible hook artifact was mounted");
    } catch (const std::invalid_argument&) {
        expect(generations.active_generation() == generation_b,
               "failed mount changed the active hook generation");
    }

    HookArtifactFixture wrong_row_count("gx1_hook_artifact_wrong_rows", 1.0F, 1);
    try {
        static_cast<void>(
            generations.mount("wrong-rows", wrong_row_count.path(), expected_model));
        throw std::runtime_error("geometry with an incompatible row ledger was mounted");
    } catch (const std::invalid_argument&) {
        expect(generations.mounted_generation_count() == 2,
               "row-ledger mismatch partially published a generation");
        expect(generations.active_generation() == generation_b,
               "row-ledger mismatch changed the active generation");
    }

    generations.retire(generation_a);
    expect(generations.mounted_generation_count() == 1,
           "retired hook generation remained mounted");
    std::vector<float> retired_pin_hidden{0.1F, 0.1F};
    static_cast<void>(pin_a.memory_hook.apply(retired_pin_hidden));
    expect(retired_pin_hidden == std::vector<float>({1.1F, 0.1F}),
           "retirement invalidated an existing inference pin");
}

} // namespace

int main() {
    try {
        test_generation_qualified_residual_and_stable_pin();
        test_l2_projection_and_missing_payload_fail_closed();
        test_memory_distance_abstention();
        test_automatic_nearest_address_candidate();
        test_prefix_mean_address_candidates();
        test_factorized_tuple_join_and_abstention();
        test_factorized_authorization_conjoins_compatibility_and_intent();
        test_persistent_hook_artifact_atomic_activation_and_corruption();
        std::cout << "hidden-state Glamin hook tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "hidden-state Glamin hook test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
