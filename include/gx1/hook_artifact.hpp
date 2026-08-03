#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gx1/hidden_state_hook.hpp"

namespace gx1 {

struct ModelHookContract {
    std::string model_sha256;
    std::string llama_revision;
    std::string architecture;
    std::string target_tensor;
    std::uint32_t hidden_dimension{0};
};

struct HookArtifactSpec {
    ModelHookContract model;
    std::uint32_t query_dimension{0};
    GlaminMetric metric{GlaminMetric::l2};
    ProjectionNormalization query_normalization{ProjectionNormalization::none};
    float gate{0.0F};
    float maximum_distance{std::numeric_limits<float>::max()};
    AddressSelectionPolicy address_selection{AddressSelectionPolicy::last_token};
    std::vector<float> input_projection;
    std::vector<std::uint64_t> residual_labels;
    std::vector<float> residuals;
    std::string glamin_directory{"glamin"};
    std::string glamin_space_id;
};

struct LoadedHookArtifact {
    std::string contract_sha256;
    ModelHookContract model;
    HiddenStateHookConfig hook_config;
    GlaminMetric metric{GlaminMetric::l2};
    std::vector<std::uint64_t> residual_labels;
    std::vector<float> residuals;
    std::filesystem::path glamin_directory;
    std::string glamin_space_id;
};

[[nodiscard]] std::string sha256_text(const std::string& text);
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

void write_hook_artifact(
    const std::filesystem::path& artifact_directory,
    const HookArtifactSpec& spec);

[[nodiscard]] LoadedHookArtifact load_hook_artifact(
    const std::filesystem::path& artifact_directory,
    const ModelHookContract& expected_model);

struct PinnedHookGeneration {
    FixedLayerMemoryHook memory_hook;
    std::string target_tensor;
    std::string contract_sha256;
};

class PersistentHookGenerationStore final {
public:
    explicit PersistentHookGenerationStore(GlaminRuntime& runtime);

    PersistentHookGenerationStore(const PersistentHookGenerationStore&) = delete;
    PersistentHookGenerationStore& operator=(const PersistentHookGenerationStore&) = delete;
    PersistentHookGenerationStore(PersistentHookGenerationStore&&) = delete;
    PersistentHookGenerationStore& operator=(PersistentHookGenerationStore&&) = delete;

    [[nodiscard]] GlaminGenerationId mount(
        std::string label,
        const std::filesystem::path& artifact_directory,
        const ModelHookContract& expected_model);
    void activate(GlaminGenerationId generation);
    void deactivate();
    [[nodiscard]] PinnedHookGeneration pin_active();
    void retire(GlaminGenerationId generation);

    [[nodiscard]] GlaminGenerationId active_generation() const noexcept;
    [[nodiscard]] std::size_t mounted_generation_count() const noexcept;

private:
    struct Resource {
        HiddenStateHookConfig hook_config;
        std::shared_ptr<const ResidualPayloadLedger> payloads;
        std::string target_tensor;
        std::string contract_sha256;
    };

    GlaminGenerationStore generations_;
    std::map<GlaminGenerationId, Resource> resources_;
};

} // namespace gx1
