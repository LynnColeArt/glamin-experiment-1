#include "gx1/activation_memory_builder.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gx1::ActivationStateSequence candidates(
    const std::vector<float>& identity,
    const float variation) {
    return {
        {1.0F, 1.0F, 1.0F},
        {identity[0] + variation, identity[1], identity[2]},
        {-1.0F, 1.0F, -1.0F},
    };
}

void test_multi_view_key_construction_and_gate() {
    const std::vector<std::vector<std::vector<float>>> identities{
        {{8.0F, 0.2F, 0.1F}, {0.1F, 8.0F, 0.2F}, {0.2F, 0.1F, 8.0F}},
        {{-8.0F, 0.2F, 0.1F}, {0.1F, -8.0F, 0.2F}, {0.2F, 0.1F, -8.0F}},
        {{8.0F, 8.0F, 0.1F}, {0.1F, 8.0F, 8.0F}, {8.0F, 0.1F, 8.0F}},
    };
    std::vector<gx1::ActivationMemoryConstructionView> views;
    for (std::size_t association = 0;
         association < identities.size();
         ++association) {
        for (std::size_t view = 0; view < 3U; ++view) {
            views.push_back(gx1::ActivationMemoryConstructionView{
                association,
                candidates(
                    identities[association][view],
                    static_cast<float>(view) * 0.05F),
                {0.0F, 0.0F, 0.0F},
                {static_cast<float>(association + 1U),
                 static_cast<float>(view + 1U),
                 0.0F},
            });
        }
    }
    const std::vector<gx1::ActivationStateSequence> negatives{
        {{1.0F, -2.0F, 1.0F}, {-2.0F, -1.0F, 1.0F}},
        {{-1.0F, -2.0F, -1.0F}, {2.0F, -1.0F, -2.0F}},
    };
    std::vector<gx1::ActivationMemoryValidationView> validation;
    for (std::size_t association = 0;
         association < identities.size();
         ++association) {
        validation.push_back(gx1::ActivationMemoryValidationView{
            association,
            candidates(identities[association][0], 0.08F),
        });
    }

    const auto result = gx1::ActivationMemoryBuilder::build(
        views, negatives, validation, gx1::ActivationMemoryBuildConfig{3, 0.75F});
    expect(result.keys.size() == 9U, "builder returned the wrong key count");
    expect(result.association_margins.size() == 3U,
           "builder returned the wrong association count");
    expect(result.validation_selections.size() == 3U,
           "builder returned the wrong validation diagnostics");
    expect(result.maximum_validation_distance < result.maximum_distance,
           "builder gate rejects its validation set");
    expect(result.maximum_distance < result.minimum_negative_distance,
           "builder gate accepts its nearest negative");
    const auto expected_gate = result.maximum_validation_distance +
                               0.75F *
                                   (result.minimum_negative_distance -
                                    result.maximum_validation_distance);
    expect(std::abs(result.maximum_distance - expected_gate) < 1.0e-6F,
           "builder returned the wrong calibrated gate");
    for (const auto selected : result.selected_candidates) {
        expect(selected == 1U, "builder did not discover the identity candidate");
    }
    for (std::size_t index = 0; index < result.residuals.size(); ++index) {
        const auto association = views[index].association;
        expect(result.residuals[index][0] == static_cast<float>(association + 1U),
               "builder returned the wrong association action");
        expect(result.residuals[index][1] ==
                   static_cast<float>((index % 3U) + 1U),
               "builder did not preserve the view-conditioned action");
    }
}

} // namespace

int main() {
    try {
        test_multi_view_key_construction_and_gate();
        std::cout << "activation-memory builder tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "activation-memory builder test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
