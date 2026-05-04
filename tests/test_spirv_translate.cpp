// SPDX-License-Identifier: Apache-2.0
//
// Tests the first vkSplat shader-translation milestones: parse a minimal
// SPIR-V module, inventory its IDs, and lower a tiny compute memory subset.

#include <vksplat/spirv/module.h>
#include <vksplat/spirv/translator.h>

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

void push_inst(std::vector<std::uint32_t>& words,
               std::uint16_t opcode,
               std::initializer_list<std::uint32_t> operands) {
    const auto wc = static_cast<std::uint32_t>(operands.size() + 1);
    words.push_back((wc << 16) | opcode);
    words.insert(words.end(), operands.begin(), operands.end());
}

std::uint32_t str_word(const char* chars) {
    std::uint32_t word = 0;
    for (int i = 0; i < 4 && chars[i] != '\0'; ++i) {
        word |= static_cast<std::uint32_t>(static_cast<unsigned char>(chars[i])) << (i * 8);
    }
    return word;
}

std::vector<std::uint32_t> minimal_compute_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        8u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main") }); // OpEntryPoint GLCompute %1 "main"
    push_inst(words, 16, { 1, 17, 1, 1, 1 }); // OpExecutionMode %1 LocalSize 1 1 1
    push_inst(words, 5, { 1, str_word("main") }); // OpName %1 "main"
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 4 }); // OpLabel %4
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> store_constant_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        9u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main") }); // OpEntryPoint GLCompute %1 "main"
    push_inst(words, 16, { 1, 17, 1, 1, 1 }); // OpExecutionMode %1 LocalSize 1 1 1
    push_inst(words, 5, { 5, str_word("out") }); // OpName %5 "out"
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 21, { 4, 32, 0 }); // OpTypeInt %4 32 0
    push_inst(words, 32, { 6, 5, 4 }); // OpTypePointer %6 CrossWorkgroup %4
    push_inst(words, 43, { 4, 7, 42 }); // OpConstant %4 %7 42
    push_inst(words, 59, { 6, 5, 5 }); // OpVariable %6 %5 CrossWorkgroup
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 8 }); // OpLabel %8
    push_inst(words, 62, { 5, 7 }); // OpStore %5 %7
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> storage_buffer_access_chain_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        19u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main") }); // OpEntryPoint GLCompute %1 "main"
    push_inst(words, 16, { 1, 17, 1, 1, 1 }); // OpExecutionMode %1 LocalSize 1 1 1
    push_inst(words, 5, { 12, str_word("src") }); // OpName %12 "src"
    push_inst(words, 5, { 13, str_word("dst") }); // OpName %13 "dst"
    push_inst(words, 71, { 12, 34, 0 }); // OpDecorate %12 DescriptorSet 0
    push_inst(words, 71, { 12, 33, 0 }); // OpDecorate %12 Binding 0
    push_inst(words, 71, { 13, 34, 0 }); // OpDecorate %13 DescriptorSet 0
    push_inst(words, 71, { 13, 33, 1 }); // OpDecorate %13 Binding 1
    push_inst(words, 71, { 9, 2 }); // OpDecorate %9 Block
    push_inst(words, 72, { 9, 0, 35, 0 }); // OpMemberDecorate %9 0 Offset 0
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 21, { 4, 32, 0 }); // OpTypeInt %4 32 0
    push_inst(words, 43, { 4, 5, 0 }); // OpConstant %4 %5 0
    push_inst(words, 43, { 4, 6, 1 }); // OpConstant %4 %6 1
    push_inst(words, 43, { 4, 7, 2 }); // OpConstant %4 %7 2
    push_inst(words, 29, { 8, 4 }); // OpTypeRuntimeArray %8 %4
    push_inst(words, 30, { 9, 8 }); // OpTypeStruct %9 %8
    push_inst(words, 32, { 10, 12, 4 }); // OpTypePointer %10 StorageBuffer %4
    push_inst(words, 32, { 11, 12, 9 }); // OpTypePointer %11 StorageBuffer %9
    push_inst(words, 59, { 11, 12, 12 }); // OpVariable %11 %12 StorageBuffer
    push_inst(words, 59, { 11, 13, 12 }); // OpVariable %11 %13 StorageBuffer
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 18 }); // OpLabel %18
    push_inst(words, 65, { 10, 14, 12, 5, 5 }); // OpAccessChain %10 %14 %12 %5 %5
    push_inst(words, 61, { 4, 15, 14 }); // OpLoad %4 %15 %14
    push_inst(words, 128, { 4, 16, 15, 7 }); // OpIAdd %4 %16 %15 %7
    push_inst(words, 65, { 10, 17, 13, 5, 6 }); // OpAccessChain %10 %17 %13 %5 %6
    push_inst(words, 62, { 17, 16 }); // OpStore %17 %16
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> global_invocation_id_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        18u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main"), 0, 12 }); // OpEntryPoint GLCompute %1 "main" %12
    push_inst(words, 16, { 1, 17, 1, 1, 1 }); // OpExecutionMode %1 LocalSize 1 1 1
    push_inst(words, 5, { 12, str_word("gid") }); // OpName %12 "gid"
    push_inst(words, 5, { 13, str_word("dst") }); // OpName %13 "dst"
    push_inst(words, 71, { 12, 11, 28 }); // OpDecorate %12 BuiltIn GlobalInvocationId
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 21, { 4, 32, 0 }); // OpTypeInt %4 32 0
    push_inst(words, 23, { 5, 4, 3 }); // OpTypeVector %5 %4 3
    push_inst(words, 32, { 6, 1, 5 }); // OpTypePointer %6 Input %5
    push_inst(words, 32, { 7, 5, 4 }); // OpTypePointer %7 CrossWorkgroup %4
    push_inst(words, 59, { 6, 12, 1 }); // OpVariable %6 %12 Input
    push_inst(words, 59, { 7, 13, 5 }); // OpVariable %7 %13 CrossWorkgroup
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 14 }); // OpLabel %14
    push_inst(words, 61, { 5, 15, 12 }); // OpLoad %5 %15 %12
    push_inst(words, 81, { 4, 16, 15, 0 }); // OpCompositeExtract %4 %16 %15 0
    push_inst(words, 62, { 13, 16 }); // OpStore %13 %16
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> push_constant_interface_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        12u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main") }); // OpEntryPoint GLCompute %1 "main"
    push_inst(words, 5, { 9, str_word("pc") }); // OpName %9 "pc"
    push_inst(words, 71, { 7, 2 }); // OpDecorate %7 Block
    push_inst(words, 72, { 7, 0, 35, 0 }); // OpMemberDecorate %7 0 Offset 0
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 21, { 4, 32, 0 }); // OpTypeInt %4 32 0
    push_inst(words, 30, { 7, 4 }); // OpTypeStruct %7 %4
    push_inst(words, 32, { 8, 9, 7 }); // OpTypePointer %8 PushConstant %7
    push_inst(words, 59, { 8, 9, 9 }); // OpVariable %8 %9 PushConstant
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 10 }); // OpLabel %10
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> ray_query_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        40u,
        0u,
    };

    push_inst(words, 17, { 1 }); // OpCapability Shader
    push_inst(words, 17, { 4472 }); // OpCapability RayQueryKHR
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5, 1, str_word("main") }); // OpEntryPoint GLCompute %1 "main"
    push_inst(words, 16, { 1, 17, 1, 1, 1 }); // OpExecutionMode %1 LocalSize 1 1 1
    push_inst(words, 5, { 13, str_word("tlas") }); // OpName %13 "tlas"
    push_inst(words, 5, { 14, str_word("rq") }); // OpName %14 "rq"
    push_inst(words, 5, { 15, str_word("hit") }); // OpName %15 "hit"
    push_inst(words, 71, { 13, 34, 0 }); // OpDecorate %13 DescriptorSet 0
    push_inst(words, 71, { 13, 33, 0 }); // OpDecorate %13 Binding 0
    push_inst(words, 71, { 15, 34, 0 }); // OpDecorate %15 DescriptorSet 0
    push_inst(words, 71, { 15, 33, 1 }); // OpDecorate %15 Binding 1
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 20, { 4 }); // OpTypeBool %4
    push_inst(words, 21, { 5, 32, 0 }); // OpTypeInt %5 32 0
    push_inst(words, 22, { 6, 32 }); // OpTypeFloat %6 32
    push_inst(words, 23, { 7, 6, 3 }); // OpTypeVector %7 %6 3
    push_inst(words, 5341, { 8 }); // OpTypeAccelerationStructureKHR %8
    push_inst(words, 4472, { 9 }); // OpTypeRayQueryKHR %9
    push_inst(words, 32, { 10, 0, 8 }); // OpTypePointer %10 UniformConstant %8
    push_inst(words, 32, { 11, 6, 9 }); // OpTypePointer %11 Private %9
    push_inst(words, 32, { 12, 5, 5 }); // OpTypePointer %12 CrossWorkgroup %5
    push_inst(words, 59, { 10, 13, 0 }); // OpVariable %10 %13 UniformConstant
    push_inst(words, 59, { 11, 14, 6 }); // OpVariable %11 %14 Private
    push_inst(words, 59, { 12, 15, 5 }); // OpVariable %12 %15 CrossWorkgroup
    push_inst(words, 43, { 5, 16, 0 }); // OpConstant %5 %16 0
    push_inst(words, 43, { 5, 17, 255 }); // OpConstant %5 %17 255
    push_inst(words, 43, { 5, 18, 1 }); // OpConstant %5 %18 1
    push_inst(words, 43, { 6, 19, 0x00000000 }); // OpConstant %6 %19 0.0
    push_inst(words, 43, { 6, 20, 0x3f800000 }); // OpConstant %6 %20 1.0
    push_inst(words, 80, { 7, 21, 19, 19, 19 }); // OpCompositeConstruct %7 %21
    push_inst(words, 80, { 7, 22, 19, 19, 20 }); // OpCompositeConstruct %7 %22
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 23 }); // OpLabel %23
    push_inst(words, 61, { 8, 24, 13 }); // OpLoad %8 %24 %13
    push_inst(words, 4473, { 14, 24, 16, 17, 21, 19, 22, 20 }); // OpRayQueryInitializeKHR
    push_inst(words, 4477, { 4, 25, 14 }); // OpRayQueryProceedKHR %4 %25 %14
    push_inst(words, 4479, { 5, 26, 14, 18 }); // OpRayQueryGetIntersectionTypeKHR %5 %26 %14 %18
    push_inst(words, 62, { 15, 26 }); // OpStore %15 %26
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

std::vector<std::uint32_t> ray_generation_trace_ray_spirv() {
    std::vector<std::uint32_t> words = {
        vksplat::spirv::kMagic,
        0x00010500u,
        0u,
        40u,
        0u,
    };

    push_inst(words, 17, { 4479 }); // OpCapability RayTracingKHR
    push_inst(words, 14, { 0, 1 }); // OpMemoryModel Logical GLSL450
    push_inst(words, 15, { 5313, 1, str_word("main") }); // OpEntryPoint RayGenerationKHR %1 "main"
    push_inst(words, 5, { 13, str_word("tlas") }); // OpName %13 "tlas"
    push_inst(words, 5, { 14, str_word("payl") }); // OpName %14 "payl"
    push_inst(words, 5, { 15, str_word("out") }); // OpName %15 "out"
    push_inst(words, 71, { 13, 34, 0 }); // OpDecorate %13 DescriptorSet 0
    push_inst(words, 71, { 13, 33, 0 }); // OpDecorate %13 Binding 0
    push_inst(words, 71, { 15, 34, 0 }); // OpDecorate %15 DescriptorSet 0
    push_inst(words, 71, { 15, 33, 1 }); // OpDecorate %15 Binding 1
    push_inst(words, 19, { 2 }); // OpTypeVoid %2
    push_inst(words, 33, { 3, 2 }); // OpTypeFunction %3 %2
    push_inst(words, 21, { 4, 32, 0 }); // OpTypeInt %4 32 0
    push_inst(words, 22, { 5, 32 }); // OpTypeFloat %5 32
    push_inst(words, 23, { 6, 5, 3 }); // OpTypeVector %6 %5 3
    push_inst(words, 5341, { 7 }); // OpTypeAccelerationStructureKHR %7
    push_inst(words, 32, { 8, 0, 7 }); // OpTypePointer %8 UniformConstant %7
    push_inst(words, 32, { 9, 5338, 4 }); // OpTypePointer %9 RayPayloadKHR %4
    push_inst(words, 32, { 10, 5, 4 }); // OpTypePointer %10 CrossWorkgroup %4
    push_inst(words, 59, { 8, 13, 0 }); // OpVariable %8 %13 UniformConstant
    push_inst(words, 59, { 9, 14, 5338 }); // OpVariable %9 %14 RayPayloadKHR
    push_inst(words, 59, { 10, 15, 5 }); // OpVariable %10 %15 CrossWorkgroup
    push_inst(words, 43, { 4, 16, 0 }); // OpConstant %4 %16 0
    push_inst(words, 43, { 4, 17, 255 }); // OpConstant %4 %17 255
    push_inst(words, 43, { 4, 18, 1 }); // OpConstant %4 %18 1
    push_inst(words, 43, { 5, 19, 0x00000000 }); // OpConstant %5 %19 0.0
    push_inst(words, 43, { 5, 20, 0x3f800000 }); // OpConstant %5 %20 1.0
    push_inst(words, 54, { 2, 1, 0, 3 }); // OpFunction %2 %1 None %3
    push_inst(words, 248, { 23 }); // OpLabel %23
    push_inst(words, 80, { 6, 21, 19, 19, 19 }); // OpCompositeConstruct %6 %21
    push_inst(words, 80, { 6, 22, 19, 19, 20 }); // OpCompositeConstruct %6 %22
    push_inst(words, 61, { 7, 24, 13 }); // OpLoad %7 %24 %13
    push_inst(words, 4445, { 24, 16, 17, 16, 18, 16, 21, 19, 22, 20, 14 }); // OpTraceRayKHR
    push_inst(words, 61, { 4, 25, 14 }); // OpLoad %4 %25 %14
    push_inst(words, 62, { 15, 25 }); // OpStore %15 %25
    push_inst(words, 253, {}); // OpReturn
    push_inst(words, 56, {}); // OpFunctionEnd
    return words;
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

int main() {
    using namespace vksplat::spirv;

    const auto words = minimal_compute_spirv();
    const Module module = parse_module(words);
    if (module.entry_points.size() != 1 || module.entry_points[0].name != "main") {
        std::fprintf(stderr, "entry point parse failed\n");
        return 1;
    }
    if (module.instructions.size() != 11) {
        std::fprintf(stderr, "instruction count mismatch: %zu\n", module.instructions.size());
        return 1;
    }

    const std::string c = translate_to_c(module);
    const std::string cuda = translate_to_cuda(module);
    if (!contains(c, "void main(void)") || !contains(c, "execution_model: compute")) {
        std::fprintf(stderr, "C translation missing expected entry shell:\n%s\n", c.c_str());
        return 1;
    }
    if (!contains(cuda, "extern \"C\" __global__ void main_kernel(void)") ||
        !contains(cuda, "execution_model: compute")) {
        std::fprintf(stderr, "CUDA translation missing expected entry shell:\n%s\n", cuda.c_str());
        return 1;
    }

    const Module store_module = parse_module(store_constant_spirv());
    const Analysis analysis = analyze_module(store_module);
    const auto variable = analysis.variables.find(5);
    if (variable == analysis.variables.end() || variable->second.name != "out" ||
        variable->second.storage_class != StorageClass::CrossWorkgroup) {
        std::fprintf(stderr, "SPIR-V analysis did not inventory the CrossWorkgroup output\n");
        return 1;
    }

    const std::string store_c = translate_to_c(store_module);
    const std::string store_cuda = translate_to_cuda(store_module);
    if (!contains(store_c, "void main(uint32_t* out)") || !contains(store_c, "out[0] = 42u;")) {
        std::fprintf(stderr, "C translation did not lower the constant store:\n%s\n", store_c.c_str());
        return 1;
    }
    if (!contains(store_cuda, "main_kernel(uint32_t* out)") ||
        !contains(store_cuda, "out[0] = 42u;")) {
        std::fprintf(stderr, "CUDA translation did not lower the constant store:\n%s\n",
                     store_cuda.c_str());
        return 1;
    }

    const Module access_module = parse_module(storage_buffer_access_chain_spirv());
    const Analysis access_analysis = analyze_module(access_module);
    const auto src = access_analysis.variables.find(12);
    const auto dst = access_analysis.variables.find(13);
    if (src == access_analysis.variables.end() || dst == access_analysis.variables.end() ||
        !src->second.has_descriptor_set || src->second.descriptor_set != 0 ||
        !src->second.has_binding || src->second.binding != 0 ||
        !dst->second.has_binding || dst->second.binding != 1 ||
        access_analysis.member_decorations.empty()) {
        std::fprintf(stderr, "SPIR-V analysis did not capture descriptor/member decorations\n");
        return 1;
    }

    const std::string access_c = translate_to_c(access_module);
    const std::string access_cuda = translate_to_cuda(access_module);
    const KernelInterface access_interface = describe_kernel_interface(access_module);
    const std::string access_manifest = format_kernel_interface_json(access_interface);
    if (access_interface.entry_point != "main" ||
        access_interface.execution_model != ExecutionModel::GLCompute ||
        access_interface.parameters.size() != 2 ||
        access_interface.parameters[0].name != "src" ||
        access_interface.parameters[0].c_type != "uint32_t*" ||
        !access_interface.parameters[0].has_descriptor_set ||
        access_interface.parameters[0].descriptor_set != 0 ||
        !access_interface.parameters[1].has_binding ||
        access_interface.parameters[1].binding != 1 ||
        !contains(access_manifest, "\"storage_class\": \"storage_buffer\"") ||
        !contains(access_manifest, "\"binding\": 1")) {
        std::fprintf(stderr, "SPIR-V kernel interface manifest is wrong:\n%s\n",
                     access_manifest.c_str());
        return 1;
    }
    if (!contains(access_c, "void main(uint32_t* src, uint32_t* dst)") ||
        !contains(access_c, "dst[1u] = (src[0u] + 2u);")) {
        std::fprintf(stderr, "C translation did not lower access-chain load/add/store:\n%s\n",
                     access_c.c_str());
        return 1;
    }
    if (!contains(access_cuda, "main_kernel(uint32_t* src, uint32_t* dst)") ||
        !contains(access_cuda, "dst[1u] = (src[0u] + 2u);")) {
        std::fprintf(stderr, "CUDA translation did not lower access-chain load/add/store:\n%s\n",
                     access_cuda.c_str());
        return 1;
    }

    const Module gid_module = parse_module(global_invocation_id_spirv());
    const Analysis gid_analysis = analyze_module(gid_module);
    const auto gid = gid_analysis.variables.find(12);
    if (gid == gid_analysis.variables.end() || !gid->second.has_builtin || gid->second.builtin != 28) {
        std::fprintf(stderr, "SPIR-V analysis did not capture GlobalInvocationId\n");
        return 1;
    }

    const std::string gid_c = translate_to_c(gid_module);
    const std::string gid_cuda = translate_to_cuda(gid_module);
    const KernelInterface gid_interface = describe_kernel_interface(gid_module);
    if (!gid_interface.uses_global_invocation_id ||
        gid_interface.parameters.size() != 1 ||
        gid_interface.parameters[0].name != "dst") {
        std::fprintf(stderr, "SPIR-V kernel interface did not expose builtin dispatch state\n");
        return 1;
    }
    if (!contains(gid_c, "uint32_t global_invocation_id_x") ||
        !contains(gid_c, "dst[0] = global_invocation_id_x;")) {
        std::fprintf(stderr, "C translation did not lower GlobalInvocationId:\n%s\n", gid_c.c_str());
        return 1;
    }
    if (contains(gid_cuda, "global_invocation_id_x") ||
        !contains(gid_cuda, "dst[0] = ((uint32_t)(blockIdx.x * blockDim.x + threadIdx.x));")) {
        std::fprintf(stderr, "CUDA translation did not lower GlobalInvocationId:\n%s\n",
                     gid_cuda.c_str());
        return 1;
    }

    const Module pc_module = parse_module(push_constant_interface_spirv());
    const KernelInterface pc_interface = describe_kernel_interface(pc_module);
    const std::string pc_manifest = format_kernel_interface_json(pc_interface);
    if (pc_interface.parameters.size() != 1 ||
        pc_interface.parameters[0].name != "pc" ||
        pc_interface.parameters[0].storage_class != StorageClass::PushConstant ||
        !contains(pc_manifest, "\"storage_class\": \"push_constant\"")) {
        std::fprintf(stderr, "SPIR-V kernel interface did not expose push constants:\n%s\n",
                     pc_manifest.c_str());
        return 1;
    }

    const Module ray_module = parse_module(ray_query_spirv());
    const Analysis ray_analysis = analyze_module(ray_module);
    if (ray_analysis.types.find(8) == ray_analysis.types.end() ||
        ray_analysis.types.find(8)->second.kind != TypeKind::AccelerationStructure ||
        ray_analysis.types.find(9) == ray_analysis.types.end() ||
        ray_analysis.types.find(9)->second.kind != TypeKind::RayQuery) {
        std::fprintf(stderr, "SPIR-V analysis did not capture ray-query opaque types\n");
        return 1;
    }

    const std::string ray_c = translate_to_c(ray_module);
    const std::string ray_cuda = translate_to_cuda(ray_module);
    const KernelInterface ray_interface = describe_kernel_interface(ray_module);
    if (ray_interface.parameters.size() != 2 ||
        ray_interface.parameters[0].name != "tlas" ||
        ray_interface.parameters[0].c_type != "const void*" ||
        ray_interface.parameters[0].storage_class != StorageClass::UniformConstant) {
        std::fprintf(stderr, "SPIR-V kernel interface did not expose TLAS resource\n");
        return 1;
    }
    if (!contains(ray_c, "void main(const void* tlas, uint32_t* hit)") ||
        !contains(ray_c, "vksplat_ray_query rq = { 0u, 0u };") ||
        !contains(ray_c, "vksplat_ray_query_initialize(&rq, tlas, 0u, 255u") ||
        !contains(ray_c, "hit[0] = vksplat_ray_query_intersection_type(&rq, 1u);")) {
        std::fprintf(stderr, "C translation did not lower ray query operations:\n%s\n", ray_c.c_str());
        return 1;
    }
    if (!contains(ray_cuda, "main_kernel(const void* tlas, uint32_t* hit)") ||
        !contains(ray_cuda, "vksplat_ray_query_initialize(&rq, tlas, 0u, 255u") ||
        !contains(ray_cuda, "hit[0] = vksplat_ray_query_intersection_type(&rq, 1u);")) {
        std::fprintf(stderr, "CUDA translation did not lower ray query operations:\n%s\n",
                     ray_cuda.c_str());
        return 1;
    }

    const Module trace_module = parse_module(ray_generation_trace_ray_spirv());
    const Analysis trace_analysis = analyze_module(trace_module);
    const auto payload = trace_analysis.variables.find(14);
    if (trace_module.entry_points.empty() ||
        trace_module.entry_points[0].execution_model != ExecutionModel::RayGenerationKHR ||
        payload == trace_analysis.variables.end() ||
        payload->second.storage_class != StorageClass::RayPayloadKHR) {
        std::fprintf(stderr, "SPIR-V analysis did not capture ray-generation payload metadata\n");
        return 1;
    }

    const std::string trace_c = translate_to_c(trace_module);
    const std::string trace_cuda = translate_to_cuda(trace_module);
    if (!contains(trace_c, "execution_model: ray_generation") ||
        !contains(trace_c, "uint32_t payl = 0;") ||
        !contains(trace_c, "vksplat_trace_ray(tlas, 0u, 255u, 0u, 1u, 0u") ||
        !contains(trace_c, "&payl);") ||
        !contains(trace_c, "out[0] = payl;")) {
        std::fprintf(stderr, "C translation did not lower OpTraceRayKHR:\n%s\n", trace_c.c_str());
        return 1;
    }
    if (!contains(trace_cuda, "execution_model: ray_generation") ||
        !contains(trace_cuda, "vksplat_trace_ray(tlas, 0u, 255u, 0u, 1u, 0u") ||
        !contains(trace_cuda, "out[0] = payl;")) {
        std::fprintf(stderr, "CUDA translation did not lower OpTraceRayKHR:\n%s\n",
                     trace_cuda.c_str());
        return 1;
    }

    return 0;
}
