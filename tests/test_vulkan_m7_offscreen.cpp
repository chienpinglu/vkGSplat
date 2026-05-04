// SPDX-License-Identifier: Apache-2.0
//
// M7 gate for the Vulkan mesh-shader path. On a Vulkan SDK + mesh shader
// machine this verifies that we can create a device with the required
// feature set. The actual offscreen draw target is compiled in only once
// shader compilation artifacts are supplied by the build.

#include <vkgsplat/vulkan/mesh_shader_3dgs.h>

#include <cstdio>
#include <vector>

namespace {

bool check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "%s failed: %d\n", where, static_cast<int>(r));
        return false;
    }
    return true;
}

} // namespace

int main() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkGSplat M7 offscreen gate";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    if (!check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance")) return 77;

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        std::fprintf(stderr, "No Vulkan physical devices available\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice picked = VK_NULL_HANDLE;
    for (VkPhysicalDevice pd : devices) {
        if (vkgsplat::vk::physical_device_supports_mesh_shader_3dgs(pd)) {
            picked = pd;
            break;
        }
    }
    if (picked == VK_NULL_HANDLE) {
        std::fprintf(stderr, "No Vulkan device supports vkGSplat mesh-shader requirements\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    std::uint32_t q_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(picked, &q_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(q_count);
    vkGetPhysicalDeviceQueueFamilyProperties(picked, &q_count, queues.data());

    std::uint32_t queue_family = UINT32_MAX;
    for (std::uint32_t i = 0; i < q_count; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family = i;
            break;
        }
    }
    if (queue_family == UINT32_MAX) {
        std::fprintf(stderr, "No graphics queue found\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{};
    mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    mesh.meshShader = VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bda.pNext = &mesh;
    bda.bufferDeviceAddress = VK_TRUE;

    const char* exts[] = {
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &bda;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = exts;

    VkDevice device = VK_NULL_HANDLE;
    if (!check(vkCreateDevice(picked, &dci, nullptr, &device), "vkCreateDevice")) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
