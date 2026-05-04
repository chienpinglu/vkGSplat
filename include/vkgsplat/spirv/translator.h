// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "module.h"

#include <string>

namespace vkgsplat::spirv {

struct TranslationOptions {
    std::string module_name = "shader";
};

struct KernelParameter {
    std::uint32_t variable_id = 0;
    std::string name;
    std::string c_type;
    StorageClass storage_class = StorageClass::Private;
    bool has_descriptor_set = false;
    std::uint32_t descriptor_set = 0;
    bool has_binding = false;
    std::uint32_t binding = 0;
};

struct KernelInterface {
    std::string entry_point;
    ExecutionModel execution_model = ExecutionModel::GLCompute;
    bool uses_global_invocation_id = false;
    std::vector<KernelParameter> parameters;
};

KernelInterface describe_kernel_interface(const Module& module,
                                          const TranslationOptions& options = {});
std::string format_kernel_interface_json(const KernelInterface& interface);

std::string translate_to_c(const Module& module, const TranslationOptions& options = {});
std::string translate_to_cuda(const Module& module, const TranslationOptions& options = {});

} // namespace vkgsplat::spirv
