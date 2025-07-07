// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

using ILGPU.Runtime.Vulkan.Native;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.Vulkan
{
    /// <summary>
    /// Represents a Vulkan compute device.
    /// </summary>
    [DebuggerDisplay("Vulkan Device: {Name}")]
    public sealed class VulkanDevice : Device
    {
        #region Instance

        /// <summary>
        /// The underlying Vulkan physical device properties.
        /// </summary>
        internal VkPhysicalDeviceProperties Properties { get; }

        /// <summary>
        /// The physical device handle.
        /// </summary>
        internal VkPhysicalDevice PhysicalDevice { get; }

        /// <summary>
        /// Gets the device vendor ID.
        /// </summary>
        public uint VendorID => Properties.vendorID;

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public uint DeviceID => Properties.deviceID;

        /// <summary>
        /// Gets the device type.
        /// </summary>
        public VkPhysicalDeviceType DeviceType => Properties.deviceType;

        /// <summary>
        /// Gets the driver version.
        /// </summary>
        public uint DriverVersion => Properties.driverVersion;

        /// <summary>
        /// Gets the API version.
        /// </summary>
        public uint APIVersion => Properties.apiVersion;

        /// <summary>
        /// Gets the maximum compute work group count.
        /// </summary>
        public Index3D MaxWorkGroupCount => new(
            (int)Properties.limits.maxComputeWorkGroupCount0,
            (int)Properties.limits.maxComputeWorkGroupCount1,
            (int)Properties.limits.maxComputeWorkGroupCount2);

        /// <summary>
        /// Gets the maximum compute work group size.
        /// </summary>
        public Index3D MaxWorkGroupSize => new(
            (int)Properties.limits.maxComputeWorkGroupSize0,
            (int)Properties.limits.maxComputeWorkGroupSize1,
            (int)Properties.limits.maxComputeWorkGroupSize2);

        /// <summary>
        /// Gets the maximum compute work group invocations.
        /// </summary>
        public uint MaxWorkGroupInvocations => Properties.limits.maxComputeWorkGroupInvocations;

        /// <summary>
        /// Gets whether this device supports unified memory.
        /// </summary>
        public new bool SupportsUnifiedMemory => DeviceType == VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

        internal VulkanDevice(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties properties)
        {
            PhysicalDevice = physicalDevice;
            Properties = properties;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public new string Name => GetDeviceNameFromProperties() ?? $"Vulkan Device {DeviceID}";

        /// <summary>
        /// Gets the total device memory in bytes.
        /// </summary>
        public new long MemorySize =>
                // Estimate memory size based on device type
                DeviceType switch
                {
                    VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 8L * 1024 * 1024 * 1024, // 8GB
                    VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 4L * 1024 * 1024 * 1024, // 4GB
                    _ => 2L * 1024 * 1024 * 1024 // 2GB default
                };

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public new static AcceleratorType AcceleratorType => AcceleratorType.Vulkan;

        /// <summary>
        /// Gets the maximum grid size.
        /// </summary>
        public new Index3D MaxGridSize => MaxWorkGroupCount;

        /// <summary>
        /// Gets the maximum group size.
        /// </summary>
        public new Index3D MaxGroupSize => MaxWorkGroupSize;

        /// <summary>
        /// Gets the maximum number of threads per group.
        /// </summary>
        public new int MaxNumThreadsPerGroup => (int)MaxWorkGroupInvocations;

        /// <summary>
        /// Gets the maximum shared memory per group in bytes.
        /// </summary>
        public new static long MaxSharedMemoryPerGroup => 32 * 1024; // 32KB typical

        /// <summary>
        /// Gets the maximum constant memory in bytes.
        /// </summary>
        public new static long MaxConstantMemory => 64 * 1024; // 64KB typical

        /// <summary>
        /// Gets the warp/subgroup size.
        /// </summary>
        public new int WarpSize =>
                // Typical subgroup sizes by vendor
                GetVendorName() switch
                {
                    "NVIDIA" => 32,    // NVIDIA warp size
                    "AMD" => 64,       // AMD wavefront size
                    "Intel" => 16,     // Intel SIMD width (varies)
                    _ => 32            // Default
                };

        /// <summary>
        /// Gets the number of multiprocessors/compute units.
        /// </summary>
        public new int NumMultiprocessors =>
                // Estimate based on device type and vendor
                DeviceType switch
                {
                    VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => GetVendorName() switch
                    {
                        "NVIDIA" => 36,    // Typical RTX GPU
                        "AMD" => 32,       // Typical RDNA GPU
                        "Intel" => 16,     // Intel Arc GPU
                        _ => 24
                    },
                    VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => GetVendorName() switch
                    {
                        "Intel" => 8,     // Intel integrated graphics
                        "AMD" => 6,       // AMD APU
                        _ => 4
                    },
                    _ => 8
                };

        /// <summary>
        /// Gets the device vendor.
        /// </summary>
        public string Vendor => GetVendorName();

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public override DeviceId DeviceId => new(DeviceID, AcceleratorType.Vulkan);

        /// <summary>
        /// Creates an accelerator for this device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created accelerator.</returns>
        public override Accelerator CreateAccelerator(Context context) =>
            new VulkanAccelerator(context, this);

        #endregion

        #region Device Detection

        /// <summary>
        /// Gets all available Vulkan devices.
        /// </summary>
        /// <returns>Array of Vulkan devices.</returns>
        public static VulkanDevice[] GetDevices()
        {
            if (!VulkanNative.IsVulkanSupported())
                return [];

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Allocate unmanaged memory for strings
                var appNamePtr = Marshal.StringToHGlobalAnsi("ILGPU Device Enumeration");
                var engineNamePtr = Marshal.StringToHGlobalAnsi("ILGPU");
                
                try
                {
                    // Create temporary instance to enumerate devices
                    var appInfo = new VkApplicationInfo
                    {
                        sType = VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                        pApplicationName = appNamePtr,
                        applicationVersion = 1,
                        pEngineName = engineNamePtr,
                        engineVersion = 1,
                        apiVersion = VulkanNative.VK_API_VERSION_1_1
                    };

                    // Allocate and pin the app info struct
                    var appInfoPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkApplicationInfo>());
                    try
                    {
                        Marshal.StructureToPtr(appInfo, appInfoPtr, false);
                        
                        var createInfo = new VkInstanceCreateInfo
                        {
                            sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                            pApplicationInfo = appInfoPtr
                        };

                        var result = VulkanNative.CreateInstance(ref createInfo, IntPtr.Zero, out var instance);
                        if (result != VkResult.VK_SUCCESS)
                            return [];

                        try
                        {
                            // Enumerate physical devices
                            uint deviceCount = 0;
                            result = VulkanNative.EnumeratePhysicalDevices(instance, ref deviceCount, null!);
                            if (result != VkResult.VK_SUCCESS || deviceCount == 0)
                                return [];

                            var physicalDevices = new VkPhysicalDevice[deviceCount];
                            result = VulkanNative.EnumeratePhysicalDevices(instance, ref deviceCount, physicalDevices);
                            if (result != VkResult.VK_SUCCESS)
                                return [];

                            // Create device objects
                            var devices = new VulkanDevice[deviceCount];
                            for (int i = 0; i < deviceCount; i++)
                            {
                                VulkanNative.GetPhysicalDeviceProperties(physicalDevices[i], out var properties);
                                devices[i] = new VulkanDevice(physicalDevices[i], properties);
                            }

                            return devices;
                        }
                        finally
                        {
                            VulkanNative.DestroyInstance(instance, IntPtr.Zero);
                        }
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(appInfoPtr);
                    }
                }
                finally
                {
                    // Free allocated strings
                    Marshal.FreeHGlobal(appNamePtr);
                    Marshal.FreeHGlobal(engineNamePtr);
                }
            }
            catch (DllNotFoundException)
            {
                return [];
            }
            catch (EntryPointNotFoundException)
            {
                return [];
            }
            catch
            {
                return [];
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Gets the default Vulkan device.
        /// </summary>
        /// <returns>The default Vulkan device or null if none available.</returns>
        public static VulkanDevice? GetDefaultDevice()
        {
            var devices = GetDevices();
            
            // Prefer discrete GPUs over integrated
            foreach (var device in devices)
            {
                if (device.DeviceType == VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                    return device;
            }

            // Fallback to any available device
            return devices.Length > 0 ? devices[0] : null;
        }

        /// <summary>
        /// Checks if Vulkan is supported on this system.
        /// </summary>
        /// <returns>True if Vulkan is supported; otherwise, false.</returns>
        public static bool IsSupported() => VulkanNative.IsVulkanSupported();

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets the vendor name from vendor ID.
        /// </summary>
        /// <returns>Vendor name.</returns>
        private string GetVendorName() => VendorID switch
        {
            0x10DE => "NVIDIA",
            0x1002 => "AMD",
            0x8086 => "Intel",
            0x5143 => "Qualcomm",
            0x1010 => "ImgTec",
            0x13B5 => "ARM",
            _ => $"Unknown (0x{VendorID:X4})"
        };

        /// <summary>
        /// Gets the device type description.
        /// </summary>
        /// <returns>Device type description.</returns>
        public string GetDeviceTypeDescription() => DeviceType switch
        {
            VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "Discrete GPU",
            VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "Integrated GPU",
            VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => "Virtual GPU",
            VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_CPU => "CPU",
            _ => "Other"
        };

        /// <summary>
        /// Gets a human-readable string of device capabilities.
        /// </summary>
        /// <returns>Device capabilities summary.</returns>
        public string GetCapabilitiesString() => $"API {APIVersion >> 22}.{(APIVersion >> 12) & 0x3FF}, " +
                   $"Driver 0x{DriverVersion:X8}, " +
                   $"{NumMultiprocessors} CUs, " +
                   $"Subgroup {WarpSize}, " +
                   $"{MemorySize / (1024 * 1024)} MB, " +
                   $"{GetDeviceTypeDescription()}";

        /// <summary>
        /// Extracts device name from properties structure.
        /// </summary>
        /// <returns>Device name string.</returns>
        private unsafe string? GetDeviceNameFromProperties()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Use stackalloc to create buffer on stack
                Span<byte> deviceNameBytes = stackalloc byte[256];
                var localProperties = Properties; // Create local copy to access fixed buffer
                
                // Copy bytes from fixed buffer to span
                for (int i = 0; i < 256; i++)
                {
                    deviceNameBytes[i] = localProperties.deviceName[i];
                    if (localProperties.deviceName[i] == 0)
                        break;
                }
                
                // Find the null terminator
                int length = 0;
                for (int i = 0; i < 256 && deviceNameBytes[i] != 0; i++)
                    length++;
                    
                return System.Text.Encoding.UTF8.GetString(deviceNameBytes.Slice(0, length));
            }
            catch
            {
                return null;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region Object

        /// <summary>
        /// Returns a string representation of this device.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString() => 
            $"Vulkan Device: {Name} ({GetDeviceTypeDescription()}, {MemorySize / (1024 * 1024)} MB)";

        /// <summary>
        /// Returns the hash code of this device.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode() => PhysicalDevice.Handle.GetHashCode();

        /// <summary>
        /// Checks for equality with another object.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public override bool Equals(object? obj) =>
            obj is VulkanDevice other && PhysicalDevice.Handle == other.PhysicalDevice.Handle;

        #endregion
    }

    // Required struct for VkDeviceQueueCreateInfo
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    internal struct VkDeviceQueueCreateInfo
    {
        public VkStructureType sType;
        public IntPtr pNext;
        public uint flags;
        public uint queueFamilyIndex;
        public uint queueCount;
        public IntPtr pQueuePriorities;
    }
}