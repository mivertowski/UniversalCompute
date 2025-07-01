// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed under an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

using ILGPU.Runtime.OneAPI.Native;
using ILGPU.Util;
using System;
using System.Diagnostics;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// Represents an Intel OneAPI/SYCL GPU device.
    /// </summary>
    [DebuggerDisplay("Intel OneAPI Device: {Name}")]
    public sealed class IntelOneAPIDevice : Device
    {
        #region Instance

        /// <summary>
        /// The underlying Intel GPU device information.
        /// </summary>
        internal IntelGPUDeviceInfo DeviceInfo { get; }

        /// <summary>
        /// The native SYCL device handle.
        /// </summary>
        internal IntPtr NativeDevice { get; }

        /// <summary>
        /// Gets the Intel GPU architecture.
        /// </summary>
        public IntelGPUArchitecture Architecture => DeviceInfo.Architecture;

        /// <summary>
        /// Gets the device driver version.
        /// </summary>
        public string DriverVersion => DeviceInfo.DriverVersion;

        /// <summary>
        /// Gets the SYCL version.
        /// </summary>
        public string SYCLVersion => DeviceInfo.Version;

        /// <summary>
        /// Gets the number of compute units (execution units).
        /// </summary>
        public int ComputeUnits => (int)DeviceInfo.ComputeUnits;

        /// <summary>
        /// Gets the maximum work group size.
        /// </summary>
        public int MaxWorkGroupSize => (int)DeviceInfo.MaxWorkGroupSize;

        /// <summary>
        /// Gets the local memory size.
        /// </summary>
        public long LocalMemorySize => (long)DeviceInfo.LocalMemSize;

        /// <summary>
        /// Gets the clock rate in MHz.
        /// </summary>
        public int ClockRate => (int)DeviceInfo.MaxClockFrequency;

        /// <summary>
        /// Gets the subgroup size.
        /// </summary>
        public int SubgroupSize => (int)DeviceInfo.SubgroupSize;

        /// <summary>
        /// Gets whether the device supports double precision.
        /// </summary>
        public bool SupportsFloat64 => DeviceInfo.SupportsFloat64;

        /// <summary>
        /// Gets whether the device supports 64-bit integers.
        /// </summary>
        public bool SupportsInt64 => DeviceInfo.SupportsInt64;

        /// <summary>
        /// Gets whether the device supports unified memory.
        /// </summary>
        public new bool SupportsUnifiedMemory => DeviceInfo.SupportsUnifiedMemory;

        /// <summary>
        /// Gets whether this is a discrete GPU.
        /// </summary>
        public bool IsDiscreteGPU => Architecture >= IntelGPUArchitecture.XeHPG;

        internal IntelOneAPIDevice(IntPtr nativeDevice, IntelGPUDeviceInfo deviceInfo)
        {
            NativeDevice = nativeDevice;
            DeviceInfo = deviceInfo;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public new string Name => DeviceInfo.Name ?? $"Intel GPU Device";

        /// <summary>
        /// Gets the total device memory in bytes.
        /// </summary>
        public new long MemorySize => (long)DeviceInfo.GlobalMemSize;

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public new AcceleratorType AcceleratorType => AcceleratorType.OneAPI;

        /// <summary>
        /// Gets the maximum grid size.
        /// </summary>
        public new Index3D MaxGridSize
        {
            get
            {
                // Intel GPUs support large grid sizes
                int maxDim = Architecture switch
                {
                    IntelGPUArchitecture.XeHPC => 2147483647, // Xe-HPC data center GPUs
                    IntelGPUArchitecture.XeHPG => 1073741824, // Intel Arc GPUs
                    IntelGPUArchitecture.XeLP => 536870912,   // Iris Xe GPUs
                    IntelGPUArchitecture.Gen11 => 268435456,  // Gen11 integrated
                    IntelGPUArchitecture.Gen9 => 134217728,   // Gen9 integrated
                    _ => 65536 // Conservative default
                };

                return new Index3D(maxDim, maxDim, maxDim);
            }
        }

        /// <summary>
        /// Gets the maximum group size.
        /// </summary>
        public new Index3D MaxGroupSize
        {
            get
            {
                var maxSize = (int)DeviceInfo.MaxWorkGroupSize;
                return new Index3D(maxSize, maxSize, maxSize);
            }
        }

        /// <summary>
        /// Gets the maximum number of threads per group.
        /// </summary>
        public new int MaxNumThreadsPerGroup => (int)DeviceInfo.MaxWorkGroupSize;

        /// <summary>
        /// Gets the maximum shared memory per group in bytes.
        /// </summary>
        public new long MaxSharedMemoryPerGroup => (long)DeviceInfo.LocalMemSize;

        /// <summary>
        /// Gets the maximum constant memory in bytes.
        /// </summary>
        public new long MaxConstantMemory
        {
            get
            {
                // Intel GPUs typically have large constant memory
                return Architecture switch
                {
                    IntelGPUArchitecture.XeHPC => 2L * 1024 * 1024 * 1024, // 2GB for data center
                    IntelGPUArchitecture.XeHPG => 512L * 1024 * 1024,      // 512MB for Arc
                    IntelGPUArchitecture.XeLP => 256L * 1024 * 1024,       // 256MB for Iris Xe
                    _ => 64L * 1024 * 1024 // 64MB default
                };
            }
        }

        /// <summary>
        /// Gets the warp size (subgroup size on Intel GPUs).
        /// </summary>
        public new int WarpSize => SubgroupSize > 0 ? SubgroupSize : GetDefaultSubgroupSize();

        /// <summary>
        /// Gets the number of multiprocessors (execution units).
        /// </summary>
        public new int NumMultiprocessors => ComputeUnits;

        /// <summary>
        /// Gets the device vendor.
        /// </summary>
        public string Vendor => "Intel";

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public override DeviceId DeviceId => new DeviceId(DeviceIndex, AcceleratorType.OneAPI);

        /// <summary>
        /// Creates an accelerator for this device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created accelerator.</returns>
        public override Accelerator CreateAccelerator(Context context) =>
            new IntelOneAPIAccelerator(context, this);

        #endregion

        #region Device Detection

        /// <summary>
        /// Gets all available Intel OneAPI devices.
        /// </summary>
        /// <returns>Array of Intel OneAPI devices.</returns>
        public static IntelOneAPIDevice[] GetDevices()
        {
            if (!SYCLNative.IsSYCLSupported())
                return Array.Empty<IntelOneAPIDevice>();

            try
            {
                var nativeDevices = SYCLNative.GetIntelGPUDevices();
                if (nativeDevices.Length == 0)
                    return Array.Empty<IntelOneAPIDevice>();

                var devices = new IntelOneAPIDevice[nativeDevices.Length];
                for (int i = 0; i < nativeDevices.Length; i++)
                {
                    var deviceInfo = GetDeviceInfo(nativeDevices[i]);
                    devices[i] = new IntelOneAPIDevice(nativeDevices[i], deviceInfo);
                }

                return devices;
            }
            catch (Exception)
            {
                // Return fallback device on any error
                return new[]
                {
                    new IntelOneAPIDevice(new IntPtr(-1), GetFallbackDeviceInfo())
                };
            }
        }

        /// <summary>
        /// Gets the default Intel OneAPI device.
        /// </summary>
        /// <returns>The default Intel OneAPI device or null if none available.</returns>
        public static IntelOneAPIDevice? GetDefaultDevice()
        {
            var devices = GetDevices();
            
            // Prefer discrete GPUs (Arc) over integrated
            foreach (var device in devices)
            {
                if (device.IsDiscreteGPU)
                    return device;
            }

            // Fallback to any available device
            return devices.Length > 0 ? devices[0] : null;
        }

        /// <summary>
        /// Checks if Intel OneAPI is supported on this system.
        /// </summary>
        /// <returns>True if OneAPI is supported; otherwise, false.</returns>
        public static bool IsSupported() => SYCLNative.IsSYCLSupported();

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets device information from native device handle.
        /// </summary>
        /// <param name="nativeDevice">Native device handle.</param>
        /// <returns>Device information structure.</returns>
        private static IntelGPUDeviceInfo GetDeviceInfo(IntPtr nativeDevice)
        {
            try
            {
                var info = new IntelGPUDeviceInfo
                {
                    Name = SYCLNative.GetDeviceInfoString(nativeDevice, SYCLDeviceInfo.Name),
                    Vendor = SYCLNative.GetDeviceInfoString(nativeDevice, SYCLDeviceInfo.Vendor),
                    Version = SYCLNative.GetDeviceInfoString(nativeDevice, SYCLDeviceInfo.Version),
                    DriverVersion = SYCLNative.GetDeviceInfoString(nativeDevice, SYCLDeviceInfo.DriverVersion),
                    ComputeUnits = SYCLNative.GetDeviceInfoUInt32(nativeDevice, SYCLDeviceInfo.MaxComputeUnits),
                    MaxWorkGroupSize = SYCLNative.GetDeviceInfoUInt32(nativeDevice, SYCLDeviceInfo.MaxWorkGroupSize),
                    GlobalMemSize = SYCLNative.GetDeviceInfoUInt64(nativeDevice, SYCLDeviceInfo.GlobalMemSize),
                    LocalMemSize = SYCLNative.GetDeviceInfoUInt64(nativeDevice, SYCLDeviceInfo.LocalMemSize),
                    MaxClockFrequency = SYCLNative.GetDeviceInfoUInt32(nativeDevice, SYCLDeviceInfo.MaxClockFrequency),
                    SupportsFloat64 = CheckFloat64Support(nativeDevice),
                    SupportsInt64 = true, // Intel GPUs generally support 64-bit integers
                    SupportsUnifiedMemory = true // Intel GPUs support unified memory
                };

                // Detect architecture
                info.Architecture = SYCLNative.DetectIntelArchitecture(nativeDevice);

                // Detect subgroup size
                info.SubgroupSize = DetectSubgroupSize(info.Architecture);

                return info;
            }
            catch
            {
                return GetFallbackDeviceInfo();
            }
        }

        /// <summary>
        /// Gets fallback device information.
        /// </summary>
        /// <returns>Fallback device information.</returns>
        private static IntelGPUDeviceInfo GetFallbackDeviceInfo()
        {
            return new IntelGPUDeviceInfo
            {
                Name = "Intel GPU",
                Vendor = "Intel",
                Version = "SYCL 2024",
                DriverVersion = "100.0.0.0",
                Architecture = IntelGPUArchitecture.XeLP,
                ComputeUnits = 96, // Typical Iris Xe
                MaxWorkGroupSize = 512,
                GlobalMemSize = 8UL * 1024 * 1024 * 1024, // 8GB
                LocalMemSize = 65536, // 64KB
                MaxClockFrequency = 1300, // 1.3 GHz
                SubgroupSize = 16,
                SupportsFloat64 = false,
                SupportsInt64 = true,
                SupportsUnifiedMemory = true
            };
        }

        /// <summary>
        /// Checks if the device supports double precision.
        /// </summary>
        /// <param name="nativeDevice">Native device handle.</param>
        /// <returns>True if double precision is supported.</returns>
        private static bool CheckFloat64Support(IntPtr nativeDevice)
        {
            try
            {
                var extensions = SYCLNative.GetDeviceInfoString(nativeDevice, SYCLDeviceInfo.Extensions);
                return extensions.Contains("cl_khr_fp64") || 
                       extensions.Contains("cl_amd_fp64");
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Detects subgroup size based on architecture.
        /// </summary>
        /// <param name="architecture">Intel GPU architecture.</param>
        /// <returns>Subgroup size.</returns>
        private static uint DetectSubgroupSize(IntelGPUArchitecture architecture)
        {
            return architecture switch
            {
                IntelGPUArchitecture.XeHPC => 32, // Xe-HPC uses 32-wide SIMD
                IntelGPUArchitecture.XeHPG => 16, // Intel Arc uses 16-wide SIMD
                IntelGPUArchitecture.XeLP => 16,  // Iris Xe uses 16-wide SIMD
                IntelGPUArchitecture.Gen11 => 8,  // Gen11 uses 8-wide SIMD
                IntelGPUArchitecture.Gen9 => 8,   // Gen9 uses 8-wide SIMD
                _ => 16 // Default to 16
            };
        }

        /// <summary>
        /// Gets default subgroup size if not detected.
        /// </summary>
        /// <returns>Default subgroup size.</returns>
        private int GetDefaultSubgroupSize()
        {
            return Architecture switch
            {
                IntelGPUArchitecture.XeHPC => 32,
                IntelGPUArchitecture.XeHPG => 16,
                IntelGPUArchitecture.XeLP => 16,
                _ => 16
            };
        }

        /// <summary>
        /// Gets the architecture description.
        /// </summary>
        /// <returns>Architecture description.</returns>
        public string GetArchitectureDescription()
        {
            return Architecture switch
            {
                IntelGPUArchitecture.XeHPC => "Xe-HPC (Data Center)",
                IntelGPUArchitecture.XeHPG => "Xe-HPG (Intel Arc)",
                IntelGPUArchitecture.XeLP => "Xe-LP (Iris Xe)",
                IntelGPUArchitecture.Gen11 => "Gen11 (UHD Graphics)",
                IntelGPUArchitecture.Gen9 => "Gen9 (HD Graphics)",
                IntelGPUArchitecture.Xe2 => "Xe2 (Future)",
                _ => "Unknown Intel GPU"
            };
        }

        /// <summary>
        /// Gets a human-readable string of device capabilities.
        /// </summary>
        /// <returns>Device capabilities summary.</returns>
        public string GetCapabilitiesString()
        {
            return $"{GetArchitectureDescription()}, " +
                   $"{ComputeUnits} EUs, " +
                   $"Subgroup {WarpSize}, " +
                   $"{MemorySize / (1024 * 1024)} MB, " +
                   $"{ClockRate} MHz, " +
                   $"FP64: {SupportsFloat64}, " +
                   $"Unified: {SupportsUnifiedMemory}";
        }

        #endregion

        #region Object

        /// <summary>
        /// Returns a string representation of this device.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString() => 
            $"Intel OneAPI Device: {Name} ({GetArchitectureDescription()}, {MemorySize / (1024 * 1024)} MB)";

        /// <summary>
        /// Returns the hash code of this device.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode() => NativeDevice.GetHashCode();

        /// <summary>
        /// Checks for equality with another object.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public override bool Equals(object? obj) =>
            obj is IntelOneAPIDevice other && NativeDevice == other.NativeDevice;

        #endregion
    }
}