// SPDX-License-Identifier: Apache-2.0

#include "vksplat/spirv/module.h"

#include <cstddef>
#include <stdexcept>

namespace vksplat::spirv {
namespace {

std::string decode_literal_string(const std::vector<std::uint32_t>& operands,
                                  std::size_t first_word) {
    std::string out;
    for (std::size_t i = first_word; i < operands.size(); ++i) {
        const std::uint32_t word = operands[i];
        for (int byte = 0; byte < 4; ++byte) {
            const char c = static_cast<char>((word >> (byte * 8)) & 0xffu);
            if (c == '\0') return out;
            out.push_back(c);
        }
    }
    return out;
}

} // namespace

Module parse_module(std::span<const std::uint32_t> words) {
    if (words.size() < 5) {
        throw std::runtime_error("SPIR-V module is shorter than the 5-word header");
    }
    if (words[0] != kMagic) {
        throw std::runtime_error("SPIR-V module has invalid magic");
    }

    Module module;
    module.version = words[1];
    module.generator = words[2];
    module.bound = words[3];

    std::size_t offset = 5;
    while (offset < words.size()) {
        const std::uint32_t first = words[offset];
        const auto word_count = static_cast<std::uint16_t>(first >> 16);
        const auto opcode = static_cast<std::uint16_t>(first & 0xffffu);
        if (word_count == 0) {
            throw std::runtime_error("SPIR-V instruction has zero word count");
        }
        if (offset + word_count > words.size()) {
            throw std::runtime_error("SPIR-V instruction overruns module");
        }

        Instruction inst;
        inst.opcode = opcode;
        inst.operands.assign(words.begin() + static_cast<std::ptrdiff_t>(offset + 1),
                             words.begin() + static_cast<std::ptrdiff_t>(offset + word_count));
        module.instructions.push_back(inst);

        if (opcode == 15 && inst.operands.size() >= 2) { // OpEntryPoint
            EntryPoint ep;
            ep.execution_model = static_cast<ExecutionModel>(inst.operands[0]);
            ep.id = inst.operands[1];
            ep.name = decode_literal_string(inst.operands, 2);
            module.entry_points.push_back(ep);
        }

        offset += word_count;
    }

    return module;
}

Analysis analyze_module(const Module& module) {
    Analysis analysis;

    for (const Instruction& inst : module.instructions) {
        const std::vector<std::uint32_t>& operands = inst.operands;
        switch (inst.opcode) {
            case 5: { // OpName
                if (operands.size() >= 2) {
                    analysis.names[operands[0]] = decode_literal_string(operands, 1);
                }
                break;
            }
            case 19: { // OpTypeVoid
                if (!operands.empty()) {
                    Type type;
                    type.kind = TypeKind::Void;
                    type.id = operands[0];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 20: { // OpTypeBool
                if (!operands.empty()) {
                    Type type;
                    type.kind = TypeKind::Bool;
                    type.id = operands[0];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 21: { // OpTypeInt
                if (operands.size() >= 3) {
                    Type type;
                    type.kind = TypeKind::Int;
                    type.id = operands[0];
                    type.width = operands[1];
                    type.signedness = operands[2] != 0;
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 22: { // OpTypeFloat
                if (operands.size() >= 2) {
                    Type type;
                    type.kind = TypeKind::Float;
                    type.id = operands[0];
                    type.width = operands[1];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 23: { // OpTypeVector
                if (operands.size() >= 3) {
                    Type type;
                    type.kind = TypeKind::Vector;
                    type.id = operands[0];
                    type.element_type_id = operands[1];
                    type.component_count = operands[2];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 29: { // OpTypeRuntimeArray
                if (operands.size() >= 2) {
                    Type type;
                    type.kind = TypeKind::RuntimeArray;
                    type.id = operands[0];
                    type.element_type_id = operands[1];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 30: { // OpTypeStruct
                if (!operands.empty()) {
                    Type type;
                    type.kind = TypeKind::Struct;
                    type.id = operands[0];
                    type.member_type_ids.assign(operands.begin() + 1, operands.end());
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 32: { // OpTypePointer
                if (operands.size() >= 3) {
                    Type type;
                    type.kind = TypeKind::Pointer;
                    type.id = operands[0];
                    type.storage_class = static_cast<StorageClass>(operands[1]);
                    type.element_type_id = operands[2];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 33: { // OpTypeFunction
                if (operands.size() >= 2) {
                    Type type;
                    type.kind = TypeKind::Function;
                    type.id = operands[0];
                    type.return_type_id = operands[1];
                    type.member_type_ids.assign(operands.begin() + 2, operands.end());
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 4472: { // OpTypeRayQueryKHR
                if (!operands.empty()) {
                    Type type;
                    type.kind = TypeKind::RayQuery;
                    type.id = operands[0];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 5341: { // OpTypeAccelerationStructureKHR
                if (!operands.empty()) {
                    Type type;
                    type.kind = TypeKind::AccelerationStructure;
                    type.id = operands[0];
                    analysis.types[type.id] = type;
                }
                break;
            }
            case 43: { // OpConstant
                if (operands.size() >= 3) {
                    Constant constant;
                    constant.type_id = operands[0];
                    constant.id = operands[1];
                    constant.value = operands[2];
                    if (operands.size() >= 4) {
                        constant.value |= static_cast<std::uint64_t>(operands[3]) << 32;
                    }
                    analysis.constants[constant.id] = constant;
                }
                break;
            }
            case 59: { // OpVariable
                if (operands.size() >= 3) {
                    Variable variable;
                    variable.result_type_id = operands[0];
                    variable.id = operands[1];
                    variable.storage_class = static_cast<StorageClass>(operands[2]);

                    const auto type_it = analysis.types.find(variable.result_type_id);
                    if (type_it != analysis.types.end() && type_it->second.kind == TypeKind::Pointer) {
                        variable.pointee_type_id = type_it->second.element_type_id;
                    }

                    const auto name_it = analysis.names.find(variable.id);
                    variable.name = name_it == analysis.names.end() ? "" : name_it->second;
                    analysis.variables[variable.id] = variable;
                }
                break;
            }
            case 71: { // OpDecorate
                if (operands.size() >= 2) {
                    Decoration decoration;
                    decoration.target_id = operands[0];
                    decoration.decoration = operands[1];
                    decoration.literals.assign(operands.begin() + 2, operands.end());
                    analysis.decorations.push_back(std::move(decoration));
                }
                break;
            }
            case 72: { // OpMemberDecorate
                if (operands.size() >= 3) {
                    MemberDecoration decoration;
                    decoration.target_id = operands[0];
                    decoration.member = operands[1];
                    decoration.decoration = operands[2];
                    decoration.literals.assign(operands.begin() + 3, operands.end());
                    analysis.member_decorations.push_back(std::move(decoration));
                }
                break;
            }
            default:
                break;
        }
    }

    for (auto& [id, variable] : analysis.variables) {
        if (variable.name.empty()) {
            const auto name_it = analysis.names.find(id);
            if (name_it != analysis.names.end()) variable.name = name_it->second;
        }
    }

    for (const Decoration& decoration : analysis.decorations) {
        auto variable_it = analysis.variables.find(decoration.target_id);
        if (variable_it == analysis.variables.end() || decoration.literals.empty()) continue;

        if (decoration.decoration == 33) { // Binding
            variable_it->second.has_binding = true;
            variable_it->second.binding = decoration.literals[0];
        } else if (decoration.decoration == 34) { // DescriptorSet
            variable_it->second.has_descriptor_set = true;
            variable_it->second.descriptor_set = decoration.literals[0];
        } else if (decoration.decoration == 11) { // BuiltIn
            variable_it->second.has_builtin = true;
            variable_it->second.builtin = decoration.literals[0];
        }
    }

    return analysis;
}

std::string execution_model_name(ExecutionModel model) {
    switch (model) {
        case ExecutionModel::Vertex: return "vertex";
        case ExecutionModel::Fragment: return "fragment";
        case ExecutionModel::GLCompute: return "compute";
        case ExecutionModel::RayGenerationKHR: return "ray_generation";
        case ExecutionModel::IntersectionKHR: return "intersection";
        case ExecutionModel::AnyHitKHR: return "any_hit";
        case ExecutionModel::ClosestHitKHR: return "closest_hit";
        case ExecutionModel::MissKHR: return "miss";
        case ExecutionModel::CallableKHR: return "callable";
        case ExecutionModel::TaskEXT: return "task";
        case ExecutionModel::MeshEXT: return "mesh";
        default: return "unknown";
    }
}

std::string storage_class_name(StorageClass storage_class) {
    switch (storage_class) {
        case StorageClass::UniformConstant: return "uniform_constant";
        case StorageClass::Input: return "input";
        case StorageClass::Uniform: return "uniform";
        case StorageClass::Output: return "output";
        case StorageClass::Workgroup: return "workgroup";
        case StorageClass::CrossWorkgroup: return "cross_workgroup";
        case StorageClass::Private: return "private";
        case StorageClass::Function: return "function";
        case StorageClass::PushConstant: return "push_constant";
        case StorageClass::StorageBuffer: return "storage_buffer";
        case StorageClass::CallableDataKHR: return "callable_data";
        case StorageClass::IncomingCallableDataKHR: return "incoming_callable_data";
        case StorageClass::RayPayloadKHR: return "ray_payload";
        case StorageClass::HitAttributeKHR: return "hit_attribute";
        case StorageClass::IncomingRayPayloadKHR: return "incoming_ray_payload";
        case StorageClass::ShaderRecordBufferKHR: return "shader_record_buffer";
        default: return "unknown";
    }
}

} // namespace vksplat::spirv
