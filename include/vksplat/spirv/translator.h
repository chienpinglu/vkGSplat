// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "module.h"

#include <string>

namespace vksplat::spirv {

struct TranslationOptions {
    std::string module_name = "shader";
};

std::string translate_to_c(const Module& module, const TranslationOptions& options = {});
std::string translate_to_cuda(const Module& module, const TranslationOptions& options = {});

} // namespace vksplat::spirv
