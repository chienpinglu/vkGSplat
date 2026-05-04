// SPDX-License-Identifier: Apache-2.0
//
// Small SPIR-V module parser used by vkSplat's shader translation
// layer. This is not a validator; it extracts the instruction stream
// and entry-point metadata that the C/CUDA translators consume.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <span>
#include <string>
#include <vector>

namespace vksplat::spirv {

constexpr std::uint32_t kMagic = 0x07230203u;

enum class ExecutionModel : std::uint32_t {
    Vertex = 0,
    Fragment = 4,
    GLCompute = 5,
    RayGenerationKHR = 5313,
    IntersectionKHR = 5314,
    AnyHitKHR = 5315,
    ClosestHitKHR = 5316,
    MissKHR = 5317,
    CallableKHR = 5318,
    TaskEXT = 5267,
    MeshEXT = 5268,
};

struct Instruction {
    std::uint16_t opcode = 0;
    std::vector<std::uint32_t> operands;
};

struct EntryPoint {
    ExecutionModel execution_model = ExecutionModel::GLCompute;
    std::uint32_t id = 0;
    std::string name;
};

struct Module {
    std::uint32_t version = 0;
    std::uint32_t generator = 0;
    std::uint32_t bound = 0;
    std::vector<Instruction> instructions;
    std::vector<EntryPoint> entry_points;
};

enum class StorageClass : std::uint32_t {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    PushConstant = 9,
    StorageBuffer = 12,
    CallableDataKHR = 5328,
    IncomingCallableDataKHR = 5329,
    RayPayloadKHR = 5338,
    HitAttributeKHR = 5339,
    IncomingRayPayloadKHR = 5342,
    ShaderRecordBufferKHR = 5343,
};

enum class TypeKind {
    Unknown,
    Void,
    Bool,
    Int,
    Float,
    Vector,
    RuntimeArray,
    Struct,
    Pointer,
    Function,
    AccelerationStructure,
    RayQuery,
};

struct Type {
    TypeKind kind = TypeKind::Unknown;
    std::uint32_t id = 0;
    std::uint32_t width = 0;
    bool signedness = false;
    std::uint32_t element_type_id = 0;
    std::uint32_t return_type_id = 0;
    std::uint32_t component_count = 0;
    StorageClass storage_class = StorageClass::Private;
    std::vector<std::uint32_t> member_type_ids;
};

struct Constant {
    std::uint32_t id = 0;
    std::uint32_t type_id = 0;
    std::uint64_t value = 0;
};

struct Variable {
    std::uint32_t id = 0;
    std::uint32_t result_type_id = 0;
    std::uint32_t pointee_type_id = 0;
    StorageClass storage_class = StorageClass::Private;
    std::string name;
    bool has_descriptor_set = false;
    std::uint32_t descriptor_set = 0;
    bool has_binding = false;
    std::uint32_t binding = 0;
    bool has_builtin = false;
    std::uint32_t builtin = 0;
};

struct Decoration {
    std::uint32_t target_id = 0;
    std::uint32_t decoration = 0;
    std::vector<std::uint32_t> literals;
};

struct MemberDecoration {
    std::uint32_t target_id = 0;
    std::uint32_t member = 0;
    std::uint32_t decoration = 0;
    std::vector<std::uint32_t> literals;
};

struct Analysis {
    std::unordered_map<std::uint32_t, std::string> names;
    std::unordered_map<std::uint32_t, Type> types;
    std::unordered_map<std::uint32_t, Constant> constants;
    std::unordered_map<std::uint32_t, Variable> variables;
    std::vector<Decoration> decorations;
    std::vector<MemberDecoration> member_decorations;
};

Module parse_module(std::span<const std::uint32_t> words);
Analysis analyze_module(const Module& module);

std::string execution_model_name(ExecutionModel model);
std::string storage_class_name(StorageClass storage_class);

} // namespace vksplat::spirv
