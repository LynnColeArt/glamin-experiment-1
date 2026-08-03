#include "gx1/glamin_runtime.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class PersistentGenerationFixture final {
public:
    PersistentGenerationFixture()
        : directory_(std::filesystem::current_path() /
                     "gx1_persistent_generation_fixture") {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        if (!std::filesystem::create_directory(directory_)) {
            throw std::runtime_error("failed to create persistent generation fixture");
        }

        std::ofstream layout(directory_ / "vector_layout.json");
        layout << "{\"dtype\": \"float32\",\"endianness\": \"little\","
                  "\"total_vectors\": 2,\"total_bytes\": 16,\"spaces\":[{"
                  "\"space_id\": \"fixture.space\",\"dim\": 2,\"count\": 2,"
                  "\"byte_stride\": 8,\"offset_bytes\": 0}]}\n";
        layout.close();

        const std::vector<float> values{0.0F, 0.0F, 2.0F, 2.0F};
        std::ofstream vectors(directory_ / "vectors.bin", std::ios::binary);
        vectors.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        vectors.close();
        write_contracts(true);
    }

    ~PersistentGenerationFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    PersistentGenerationFixture(const PersistentGenerationFixture&) = delete;
    PersistentGenerationFixture& operator=(const PersistentGenerationFixture&) = delete;

    [[nodiscard]] std::string path() const {
        return directory_.string();
    }

    void write_contracts(const bool valid_hash) const {
        const auto hash = valid_hash
            ? "sha256:c761b2699931aba1583df083fa99155e82b3ad08545bcb2e2ce3521034f80f88"
            : "sha256:0761b2699931aba1583df083fa99155e82b3ad08545bcb2e2ce3521034f80f88";
        std::ofstream contracts(directory_ / "contracts.json");
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
               "\"contract_hash\": \""
            << hash << "\",\"signature\": \"\"}]}\n";
    }

private:
    std::filesystem::path directory_;
};

void test_abi_version() {
    expect(
        gx1::GlaminRuntime::abi_version() == GLAMIN_ABI_VERSION,
        "unexpected Glamin ABI version");
}

void test_invalid_worker_count_reports_diagnostic() {
    try {
        gx1::GlaminRuntime invalid_runtime(0);
        static_cast<void>(invalid_runtime);
    } catch (const gx1::GlaminError& error) {
        expect(
            error.status() == GLAMIN_STATUS_INVALID_ARGUMENT,
            "zero worker count returned the wrong status");
        expect(
            std::string(error.what()).find("worker_count") != std::string::npos,
            "zero worker count did not produce an actionable diagnostic");
        return;
    }
    throw std::runtime_error("zero worker count unexpectedly created a runtime");
}

void test_runtime_lifecycle_and_move() {
    gx1::GlaminRuntime runtime(2);
    expect(runtime.is_open(), "created runtime must be open");
    expect(runtime.native_handle() != 0, "created runtime must have a nonzero handle");

    const auto original_handle = runtime.native_handle();
    gx1::GlaminRuntime moved(std::move(runtime));
    expect(!runtime.is_open(), "moved-from runtime must be closed");
    expect(moved.native_handle() == original_handle, "move must preserve the native handle");

    moved.close();
    expect(!moved.is_open(), "closed runtime must report closed");
    moved.close();
}

void test_runtime_rejects_live_index() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminFlatIndex index(runtime, 2);

    try {
        runtime.close();
    } catch (const gx1::GlaminError& error) {
        expect(
            error.status() == GLAMIN_STATUS_NOT_READY,
            "live index returned the wrong runtime-close status");
        expect(
            std::string(error.what()).find("indexes") != std::string::npos,
            "runtime-close diagnostic did not mention its live indexes");
        index.close();
        runtime.close();
        return;
    }
    throw std::runtime_error("runtime closed while it still owned an index");
}

void test_flat_recall_and_shared_runtime_lifetime() {
    std::unique_ptr<gx1::GlaminFlatIndex> index;
    {
        gx1::GlaminRuntime runtime(2);
        index = std::make_unique<gx1::GlaminFlatIndex>(runtime, 2);
        index->add({0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F});
        expect(index->vector_count() == 3, "flat index row count mismatch");
    }

    const auto result = index->search({0.1F, 0.1F, 1.9F, 2.1F}, 2);
    const std::vector<std::uint64_t> expected_labels{0, 1, 2, 1};
    const std::vector<float> expected_distances{0.02F, 1.62F, 0.02F, 2.02F};
    expect(result.query_count == 2, "flat search query count mismatch");
    expect(result.k == 2, "flat search k mismatch");
    expect(result.labels == expected_labels, "flat search label mismatch");
    expect(result.distances.size() == expected_distances.size(), "distance count mismatch");
    for (std::size_t result_index = 0; result_index < result.distances.size(); ++result_index) {
        expect(
            std::fabs(result.distances[result_index] - expected_distances[result_index]) <
                1.0e-5F,
            "flat search distance mismatch");
    }
}

void test_real_generation_activation_pin_rollback_and_retirement() {
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation_a = generations.mount_flat(
        "behavior-a", 2, {0.0F, 0.0F, 10.0F, 10.0F});
    const auto generation_b = generations.mount_flat(
        "behavior-b", 2, {100.0F, 100.0F, 2.0F, 2.0F});
    expect(generations.mounted_generation_count() == 2, "generation mount count mismatch");

    try {
        static_cast<void>(generations.pin_active());
        throw std::runtime_error("pinning without an active generation succeeded");
    } catch (const gx1::GlaminError& error) {
        expect(
            error.status() == GLAMIN_STATUS_NOT_READY,
            "pinning without an active generation returned the wrong status");
    }

    generations.activate(generation_a);
    auto pin_a = generations.pin_active();
    const auto original_a = pin_a.search({1.9F, 2.1F}, 1);
    expect(pin_a.id() == generation_a, "generation A pin resolved incorrectly");
    expect(pin_a.label() == "behavior-a", "generation A label mismatch");
    expect(original_a.labels == std::vector<std::uint64_t>{0}, "generation A result mismatch");

    try {
        generations.retire(generation_a);
        throw std::runtime_error("active generation retirement succeeded");
    } catch (const gx1::GlaminError& error) {
        expect(
            error.status() == GLAMIN_STATUS_NOT_READY,
            "active generation retirement returned the wrong status");
    }

    generations.activate(generation_b);
    auto pin_b = generations.pin_active();
    const auto result_b = pin_b.search({1.9F, 2.1F}, 1);
    expect(pin_b.id() == generation_b, "generation B pin resolved incorrectly");
    expect(result_b.labels == std::vector<std::uint64_t>{1}, "generation B result mismatch");
    expect(pin_a.search({1.9F, 2.1F}, 1).labels == original_a.labels,
           "old generation A pin changed after activating B");

    generations.activate(generation_a);
    auto rollback_pin = generations.pin_active();
    const auto rollback_result = rollback_pin.search({1.9F, 2.1F}, 1);
    expect(rollback_result.labels == original_a.labels, "generation rollback was not stable");
    expect(rollback_result.distances == original_a.distances,
           "generation rollback distance changed");

    generations.deactivate();
    generations.retire(generation_a);
    expect(generations.mounted_generation_count() == 1,
           "retired generation A remained mounted");
    expect(pin_a.search({1.9F, 2.1F}, 1).labels == original_a.labels,
           "retirement invalidated the existing generation A pin");
    rollback_pin.release();
    pin_a.release();

    generations.retire(generation_b);
    expect(generations.mounted_generation_count() == 0,
           "retired generation B remained mounted");
    expect(pin_b.search({1.9F, 2.1F}, 1).labels == result_b.labels,
           "retirement invalidated the existing generation B pin");
    pin_b.release();
    runtime.close();
}

void test_generation_pin_keeps_retired_resource_alive() {
    gx1::GlaminRuntime runtime(2);
    std::unique_ptr<gx1::GlaminGenerationPin> surviving_pin;
    {
        gx1::GlaminGenerationStore generations(runtime);
        const auto generation = generations.mount_flat(
            "surviving-pin", 2, {0.0F, 0.0F, 3.0F, 3.0F});
        generations.activate(generation);
        surviving_pin = std::make_unique<gx1::GlaminGenerationPin>(
            generations.pin_active());
        generations.deactivate();
        generations.retire(generation);
    }

    const auto result = surviving_pin->search({2.9F, 3.1F}, 1);
    expect(result.labels == std::vector<std::uint64_t>{1},
           "pin did not keep its retired generation resource alive");
    surviving_pin.reset();
    runtime.close();
}

void test_persistent_generation_load_and_contract_rejection() {
    PersistentGenerationFixture fixture;
    gx1::GlaminRuntime runtime(2);
    gx1::GlaminGenerationStore generations(runtime);
    const auto generation = generations.mount_flat_artifact(
        "persistent-a", fixture.path(), "fixture.space");
    generations.activate(generation);
    auto pin = generations.pin_active();
    const auto original = pin.search({1.9F, 2.1F}, 1);
    expect(pin.dimension() == 2, "persistent generation dimension mismatch");
    expect(pin.vector_count() == 2, "persistent generation vector count mismatch");
    expect(original.labels == std::vector<std::uint64_t>{1},
           "persistent generation search mismatch");

    fixture.write_contracts(false);
    try {
        static_cast<void>(generations.mount_flat_artifact(
            "invalid-contract", fixture.path(), "fixture.space"));
        throw std::runtime_error("invalid persistent contract was accepted");
    } catch (const gx1::GlaminError& error) {
        expect(error.status() == GLAMIN_STATUS_INVALID_ARGUMENT,
               "invalid persistent contract returned the wrong status");
    }
    expect(generations.mounted_generation_count() == 1,
           "failed persistent mount changed the generation store");
    expect(pin.search({1.9F, 2.1F}, 1).labels == original.labels,
           "failed persistent mount changed the active pin");

    generations.deactivate();
    generations.retire(generation);
    pin.release();
    runtime.close();
}

} // namespace

int main() {
    try {
        test_abi_version();
        test_invalid_worker_count_reports_diagnostic();
        test_runtime_lifecycle_and_move();
        test_runtime_rejects_live_index();
        test_flat_recall_and_shared_runtime_lifetime();
        test_real_generation_activation_pin_rollback_and_retirement();
        test_generation_pin_keeps_retired_resource_alive();
        test_persistent_generation_load_and_contract_rejection();
    } catch (const std::exception& error) {
        std::cerr << "Glamin C++ lifecycle test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Glamin C++ lifecycle tests passed\n";
    return EXIT_SUCCESS;
}
