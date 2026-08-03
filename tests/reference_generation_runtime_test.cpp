#include "gx1/reference_generation_runtime.hpp"

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gx1::reference::Generation;
using gx1::reference::GenerationRuntime;

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception>
void expect_throws(const std::function<void()>& operation, const std::string& message) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::shared_ptr<const Generation> make_generation(
    const std::uint64_t id,
    const std::string& label,
    const float bias) {
    return std::make_shared<const Generation>(
        id,
        label,
        std::vector<float>{1.0F, 1.0F},
        std::vector<float>{bias, bias});
}

void test_generation_validation() {
    expect_throws<std::invalid_argument>(
        [] {
            Generation(
                0,
                "invalid",
                std::vector<float>{1.0F},
                std::vector<float>{0.0F});
        },
        "generation id zero must be rejected");

    expect_throws<std::invalid_argument>(
        [] {
            Generation(
                1,
                "mismatched",
                std::vector<float>{1.0F},
                std::vector<float>{0.0F, 1.0F});
        },
        "mismatched affine dimensions must be rejected");
}

void test_activation_pin_and_rollback() {
    GenerationRuntime runtime;
    const auto generation_a = make_generation(1, "a", 1.0F);
    const auto generation_b = make_generation(2, "b", 10.0F);

    expect_throws<std::logic_error>(
        [&runtime] { static_cast<void>(runtime.pin_active()); },
        "pinning without an active generation must fail");

    runtime.install(generation_a);
    runtime.install(generation_b);
    expect(runtime.mounted_generation_count() == 2, "two generations must be mounted");

    runtime.activate(generation_a->id());
    const auto pinned_a = runtime.pin_active();
    const std::vector<float> input{2.0F, 3.0F};
    const auto result_a = pinned_a.traverse(input);

    runtime.activate(generation_b->id());
    const auto pinned_b = runtime.pin_active();
    const auto result_b = pinned_b.traverse(input);

    expect(pinned_a.id() == generation_a->id(), "old pin must retain generation A");
    expect(pinned_b.id() == generation_b->id(), "new pin must resolve generation B");
    expect(result_a == pinned_a.traverse(input), "old pin result must remain stable");
    expect(result_a != result_b, "generation swap must change reference behavior");

    runtime.activate(generation_a->id());
    expect(
        runtime.pin_active().traverse(input) == result_a,
        "rollback must restore the original deterministic result");
}

void test_retirement_preserves_existing_lease() {
    GenerationRuntime runtime;
    const auto generation_a = make_generation(1, "a", 1.0F);
    const auto generation_b = make_generation(2, "b", 2.0F);

    runtime.install(generation_a);
    runtime.install(generation_b);
    runtime.activate(generation_a->id());
    const auto pinned_a = runtime.pin_active();

    runtime.activate(generation_b->id());
    runtime.retire(generation_a->id());

    expect(runtime.mounted_generation_count() == 1, "retired generation must be unmounted");
    expect(
        pinned_a.traverse(std::vector<float>{1.0F, 1.0F}) ==
            std::vector<float>({2.0F, 2.0F}),
        "existing lease must keep retired generation alive");

    expect_throws<std::invalid_argument>(
        [&runtime, generation_a] { runtime.activate(generation_a->id()); },
        "retired generation must not be activated");
    expect_throws<std::logic_error>(
        [&runtime, generation_b] { runtime.retire(generation_b->id()); },
        "active generation must not be retired");
}

void test_invalid_operations() {
    GenerationRuntime runtime;
    const auto generation = make_generation(1, "one", 0.0F);
    runtime.install(generation);

    expect_throws<std::invalid_argument>(
        [&runtime, generation] { runtime.install(generation); },
        "duplicate generation id must fail");
    expect_throws<std::invalid_argument>(
        [&runtime] { runtime.activate(99); },
        "unknown generation activation must fail");

    runtime.activate(generation->id());
    const auto lease = runtime.pin_active();
    expect_throws<std::invalid_argument>(
        [&lease] { static_cast<void>(lease.traverse(std::vector<float>{1.0F})); },
        "dimension mismatch must fail");
}

} // namespace

int main() {
    try {
        test_generation_validation();
        test_activation_pin_and_rollback();
        test_retirement_preserves_existing_lease();
        test_invalid_operations();
    } catch (const std::exception& error) {
        std::cerr << "reference runtime test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "reference runtime tests passed\n";
    return EXIT_SUCCESS;
}
