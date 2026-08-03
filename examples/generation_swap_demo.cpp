#include "gx1/reference_generation_runtime.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using gx1::reference::Generation;
using gx1::reference::GenerationLease;
using gx1::reference::GenerationRuntime;

void print_result(
    const std::string& stage,
    const GenerationLease& lease,
    const std::vector<float>& result) {
    std::cout << stage << " generation=" << lease.id()
              << " label=" << lease.label() << " result=[";
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(2) << result[index];
    }
    std::cout << "]\n";
}

} // namespace

int main() {
    const std::vector<float> input{2.0F, 3.0F, 4.0F};

    auto generation_a = std::make_shared<const Generation>(
        1,
        "behavior-a",
        std::vector<float>{1.0F, 1.0F, 1.0F},
        std::vector<float>{1.0F, 0.0F, -1.0F});
    auto generation_b = std::make_shared<const Generation>(
        2,
        "behavior-b",
        std::vector<float>{-1.0F, 2.0F, 0.5F},
        std::vector<float>{0.0F, 1.0F, 2.0F});

    GenerationRuntime runtime;
    runtime.install(generation_a);
    runtime.install(generation_b);

    runtime.activate(generation_a->id());
    const auto pinned_a = runtime.pin_active();
    const auto original_a_result = pinned_a.traverse(input);
    print_result("initial", pinned_a, original_a_result);

    runtime.activate(generation_b->id());
    const auto pinned_b = runtime.pin_active();
    const auto b_result = pinned_b.traverse(input);
    print_result("after-swap", pinned_b, b_result);

    const auto still_a_result = pinned_a.traverse(input);
    print_result("old-pin", pinned_a, still_a_result);

    runtime.activate(generation_a->id());
    const auto restored_a = runtime.pin_active();
    const auto restored_a_result = restored_a.traverse(input);
    print_result("rollback", restored_a, restored_a_result);

    const bool swap_changed_behavior = original_a_result != b_result;
    const bool old_pin_stayed_consistent = original_a_result == still_a_result;
    const bool rollback_was_deterministic = original_a_result == restored_a_result;

    if (!swap_changed_behavior || !old_pin_stayed_consistent ||
        !rollback_was_deterministic) {
        std::cerr << "reference generation contract failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "reference generation contract passed\n";
    return EXIT_SUCCESS;
}
