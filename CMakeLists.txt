cmake_minimum_required(VERSION 3.24)

project(
    glamin_experiment_1
    VERSION 0.1.0
    DESCRIPTION "Live executable geometry inference experiment"
    LANGUAGES C CXX Fortran
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GX1_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
option(GX1_BUILD_GLAMIN_BACKEND "Build the real Glamin runtime adapter" ON)
option(GX1_BUILD_LLAMA_HOOK "Build the pinned llama.cpp hidden-state hook" ON)

function(gx1_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(GX1_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(
            ${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
        )
        if(GX1_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(gx1_reference_runtime STATIC
    src/reference_generation_runtime.cpp
)

target_include_directories(
    gx1_reference_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

gx1_enable_warnings(gx1_reference_runtime)

add_executable(
    gx1_generation_swap_demo
    examples/generation_swap_demo.cpp
)

target_link_libraries(
    gx1_generation_swap_demo
    PRIVATE
        gx1_reference_runtime
)

gx1_enable_warnings(gx1_generation_swap_demo)

include(CTest)

if(BUILD_TESTING)
    add_executable(
        gx1_reference_runtime_tests
        tests/reference_generation_runtime_test.cpp
    )

    target_link_libraries(
        gx1_reference_runtime_tests
        PRIVATE
            gx1_reference_runtime
    )

    gx1_enable_warnings(gx1_reference_runtime_tests)

    add_test(
        NAME reference_generation_runtime
        COMMAND gx1_reference_runtime_tests
    )
endif()

if(GX1_BUILD_GLAMIN_BACKEND)
    set(GLAMIN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glamin")
    set(GLAMIN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/glamin")
    set(GLAMIN_LIBRARY "${GLAMIN_BUILD_DIR}/libglamin.a")

    if(NOT EXISTS "${GLAMIN_SOURCE_DIR}/include/glamin_runtime.h")
        message(FATAL_ERROR "Glamin submodule is missing; run git submodule update --init")
    endif()

    find_program(GX1_GNU_MAKE NAMES gmake make REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS Fortran)
    find_package(Threads REQUIRED)

    add_custom_target(
        glamin_native_build
        COMMAND
            "${GX1_GNU_MAKE}"
            -C "${GLAMIN_SOURCE_DIR}"
            BUILD_DIR="${GLAMIN_BUILD_DIR}"
            all
        BYPRODUCTS "${GLAMIN_LIBRARY}"
        COMMENT "Building the Glamin Fortran/C runtime"
        VERBATIM
    )

    add_library(glamin_native STATIC IMPORTED GLOBAL)
    set_target_properties(
        glamin_native
        PROPERTIES
            IMPORTED_LOCATION "${GLAMIN_LIBRARY}"
    )
    add_dependencies(glamin_native glamin_native_build)

    add_library(
        gx1_glamin_runtime
        STATIC
        src/glamin_runtime.cpp
    )
    target_include_directories(
        gx1_glamin_runtime
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${GLAMIN_SOURCE_DIR}/include
    )
    target_link_libraries(
        gx1_glamin_runtime
        PUBLIC
            glamin_native
            OpenMP::OpenMP_Fortran
            Threads::Threads
            ${CMAKE_DL_LIBS}
            ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
    )
    gx1_enable_warnings(gx1_glamin_runtime)

    add_library(
        gx1_hidden_state_hook
        STATIC
        src/activation_memory_builder.cpp
        src/factorized_memory_hook.cpp
        src/hidden_state_hook.cpp
        src/hook_artifact.cpp
    )
    target_include_directories(
        gx1_hidden_state_hook
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(
        gx1_hidden_state_hook
        PUBLIC
            gx1_glamin_runtime
    )
    gx1_enable_warnings(gx1_hidden_state_hook)

    add_executable(
        gx1_glamin_runtime_smoke
        examples/glamin_runtime_smoke.cpp
    )
    target_link_libraries(
        gx1_glamin_runtime_smoke
        PRIVATE
            gx1_glamin_runtime
    )
    gx1_enable_warnings(gx1_glamin_runtime_smoke)

    add_executable(
        gx1_glamin_generation_swap_demo
        examples/glamin_generation_swap_demo.cpp
    )
    target_link_libraries(
        gx1_glamin_generation_swap_demo
        PRIVATE
            gx1_glamin_runtime
    )
    gx1_enable_warnings(gx1_glamin_generation_swap_demo)

    add_executable(
        gx1_glamin_persistent_generation_demo
        examples/glamin_persistent_generation_demo.cpp
    )
    target_link_libraries(
        gx1_glamin_persistent_generation_demo
        PRIVATE
            gx1_glamin_runtime
    )
    gx1_enable_warnings(gx1_glamin_persistent_generation_demo)

    if(BUILD_TESTING)
        add_executable(
            gx1_glamin_runtime_tests
            tests/glamin_runtime_test.cpp
        )
        target_link_libraries(
            gx1_glamin_runtime_tests
            PRIVATE
                gx1_glamin_runtime
        )
        gx1_enable_warnings(gx1_glamin_runtime_tests)

        add_test(
            NAME glamin_runtime_flat_and_generations
            COMMAND gx1_glamin_runtime_tests
        )

        add_executable(
            gx1_hidden_state_hook_tests
            tests/hidden_state_hook_test.cpp
        )
        target_link_libraries(
            gx1_hidden_state_hook_tests
            PRIVATE
                gx1_hidden_state_hook
        )
        gx1_enable_warnings(gx1_hidden_state_hook_tests)

        add_test(
            NAME hidden_state_glamin_hook
            COMMAND gx1_hidden_state_hook_tests
        )

        add_executable(
            gx1_activation_memory_builder_tests
            tests/activation_memory_builder_test.cpp
        )
        target_link_libraries(
            gx1_activation_memory_builder_tests
            PRIVATE
                gx1_hidden_state_hook
        )
        gx1_enable_warnings(gx1_activation_memory_builder_tests)

        add_test(
            NAME activation_memory_builder
            COMMAND gx1_activation_memory_builder_tests
        )
    endif()

    if(GX1_BUILD_LLAMA_HOOK)
        set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
        if(NOT EXISTS "${LLAMA_SOURCE_DIR}/include/llama.h")
            message(FATAL_ERROR "llama.cpp submodule is missing; run git submodule update --init")
        endif()

        set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory("${LLAMA_SOURCE_DIR}" EXCLUDE_FROM_ALL)

        add_library(
            gx1_llama_glamin_hook
            STATIC
            src/llama_glamin_hook.cpp
        )
        target_include_directories(
            gx1_llama_glamin_hook
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${LLAMA_SOURCE_DIR}/include
                ${LLAMA_SOURCE_DIR}/ggml/include
        )
        target_link_libraries(
            gx1_llama_glamin_hook
            PUBLIC
                gx1_hidden_state_hook
                llama
        )
        gx1_enable_warnings(gx1_llama_glamin_hook)

        add_executable(
            gx1_llama_hidden_state_demo
            examples/llama_hidden_state_demo.cpp
        )
        target_link_libraries(
            gx1_llama_hidden_state_demo
            PRIVATE
                gx1_llama_glamin_hook
        )
        gx1_enable_warnings(gx1_llama_hidden_state_demo)

        add_executable(
            gx1_llama_activation_memory_experiment
            examples/llama_activation_memory_experiment.cpp
        )
        target_link_libraries(
            gx1_llama_activation_memory_experiment
            PRIVATE
                gx1_llama_glamin_hook
        )
        gx1_enable_warnings(gx1_llama_activation_memory_experiment)

        add_executable(
            gx1_llama_factorized_memory_experiment
            examples/llama_factorized_memory_experiment.cpp
        )
        target_link_libraries(
            gx1_llama_factorized_memory_experiment
            PRIVATE
                gx1_llama_glamin_hook
        )
        gx1_enable_warnings(gx1_llama_factorized_memory_experiment)

        if(BUILD_TESTING)
            add_executable(
                gx1_llama_glamin_hook_tests
                tests/llama_glamin_hook_test.cpp
            )
            target_link_libraries(
                gx1_llama_glamin_hook_tests
                PRIVATE
                    gx1_llama_glamin_hook
            )
            gx1_enable_warnings(gx1_llama_glamin_hook_tests)

            add_test(
                NAME llama_glamin_tensor_callback
                COMMAND gx1_llama_glamin_hook_tests
            )
        endif()
    endif()
endif()
