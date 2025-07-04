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

using ILGPU.Runtime.ROCm.Native;
using ILGPU.Util;
using System;
using System.Diagnostics;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// Represents an AMD ROCm GPU device.
    /// </summary>
    [DebuggerDisplay("ROCm Device {ROCmDeviceId}: {Name}")]
    public sealed class ROCmDevice : Device
    {
        #region Instance

        /// <summary>
        /// The underlying HIP device properties.
        /// </summary>
        internal HipDeviceProperties Properties { get; }

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public int ROCmDeviceId { get; }

        /// <summary>
        /// Gets the compute capability major version.
        /// </summary>
        public int ComputeCapabilityMajor => Properties.Major;

        /// <summary>
        /// Gets the compute capability minor version.
        /// </summary>
        public int ComputeCapabilityMinor => Properties.Minor;

        /// <summary>
        /// Gets the compute capability as a combined version.
        /// </summary>
        public Version ComputeCapability => new(ComputeCapabilityMajor, ComputeCapabilityMinor);

        /// <summary>
        /// Gets the GPU architecture name.
        /// </summary>
        public string Architecture => GetArchitectureName(ComputeCapabilityMajor, ComputeCapabilityMinor);

        /// <summary>
        /// Gets the number of multiprocessors (CUs - Compute Units).
        /// </summary>
        public int MultiprocessorCount => Properties.MultiProcessorCount;

        /// <summary>
        /// Gets the maximum number of threads per multiprocessor.
        /// </summary>
        public int MaxThreadsPerMultiprocessor => Properties.MaxThreadsPerMultiProcessor;

        /// <summary>
        /// Gets the warp size (wavefront size on AMD).
        /// </summary>
        public new int WarpSize => Properties.WarpSize;

        /// <summary>
        /// Gets the maximum shared memory per block.
        /// </summary>
        public new long MaxSharedMemoryPerGroup => (long)Properties.SharedMemPerBlock;

        /// <summary>
        /// Gets the maximum constant memory size.
        /// </summary>
        public new long MaxConstantMemory => (long)Properties.TotalConstMem;

        /// <summary>
        /// Gets the clock rate in KHz.
        /// </summary>
        public int ClockRate => Properties.ClockRate;

        /// <summary>
        /// Gets the memory clock rate in KHz.
        /// </summary>
        public int MemoryClockRate => Properties.MemoryClockRate;

        /// <summary>
        /// Gets the memory bus width in bits.
        /// </summary>
        public int MemoryBusWidth => Properties.MemoryBusWidth;

        /// <summary>
        /// Gets the L2 cache size in bytes.
        /// </summary>
        public int L2CacheSize => Properties.L2CacheSize;

        /// <summary>
        /// Gets whether the device supports unified addressing.
        /// </summary>
        public bool SupportsUnifiedAddressing => Properties.UnifiedAddressing != 0;

        /// <summary>
        /// Gets whether the device supports managed memory.
        /// </summary>
        public bool SupportsManagedMemory => Properties.ManagedMemory != 0;

        /// <summary>
        /// Gets whether the device supports cooperative launch.
        /// </summary>
        public bool SupportsCooperativeLaunch => Properties.CooperativeLaunch != 0;

        /// <summary>
        /// Gets whether the device supports concurrent kernels.
        /// </summary>
        public bool SupportsConcurrentKernels => Properties.ConcurrentKernels != 0;

        /// <summary>
        /// Gets the PCI bus ID.
        /// </summary>
        public int PciBusId => Properties.PciBusId;

        /// <summary>
        /// Gets the PCI device ID.
        /// </summary>
        public int PciDeviceId => Properties.PciDeviceId;

        /// <summary>
        /// Gets the PCI domain ID.
        /// </summary>
        public int PciDomainId => Properties.PciDomainId;

        internal ROCmDevice(int deviceId, HipDeviceProperties properties)
        {
            ROCmDeviceId = deviceId;
            Properties = properties;
            
            // Set Device base class properties
            Name = Properties.Name ?? $"ROCm Device {ROCmDeviceId}";
            MemorySize = (long)Properties.TotalGlobalMem;
            MaxGridSize = new Index3D(Properties.MaxGridSize[0], Properties.MaxGridSize[1], Properties.MaxGridSize[2]);
            MaxGroupSize = new Index3D(Properties.MaxThreadsDim[0], Properties.MaxThreadsDim[1], Properties.MaxThreadsDim[2]);
            MaxNumThreadsPerGroup = Properties.MaxThreadsPerBlock;
            MaxSharedMemoryPerGroup = Properties.SharedMemPerBlock;
            MaxConstantMemory = Properties.TotalConstMem;
            WarpSize = Properties.WarpSize;
            NumMultiprocessors = Properties.MultiProcessorCount;
            MaxNumThreadsPerMultiprocessor = Properties.MaxThreadsPerMultiProcessor;
            
            InitializeMemoryInfo();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public override DeviceId DeviceId => new(ROCmDeviceId, AcceleratorType.ROCm);


        /// <summary>
        /// Gets the device vendor (AMD).
        /// </summary>
        public string Vendor => "AMD";

        /// <summary>
        /// Creates an accelerator for this device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created accelerator.</returns>
        public override Accelerator CreateAccelerator(Context context)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            return new ROCmAccelerator(context, this);
        }

        #endregion

        #region Device Detection

        /// <summary>
        /// Gets all available ROCm devices.
        /// </summary>
        /// <returns>Array of ROCm devices.</returns>
        public static ROCmDevice[] GetDevices()
        {
            if (!ROCmNative.IsROCmSupported())
                return Array.Empty<ROCmDevice>();

            int deviceCount = ROCmNative.GetDeviceCountSafe();
            if (deviceCount == 0)
                return Array.Empty<ROCmDevice>();

            var devices = new ROCmDevice[deviceCount];
            for (int i = 0; i < deviceCount; i++)
            {
                var properties = ROCmNative.GetDevicePropertiesSafe(i);
                if (properties.HasValue)
                {
                    devices[i] = new ROCmDevice(i, properties.Value);
                }
                else
                {
                    // Create a fallback device with minimal properties
                    var fallbackProps = new HipDeviceProperties
                    {
                        Name = $"ROCm Device {i}",
                        TotalGlobalMem = 8UL * 1024 * 1024 * 1024, // 8GB default
                        MultiProcessorCount = 36, // Typical for mid-range AMD GPU
                        WarpSize = 64, // AMD wavefront size
                        MaxThreadsPerBlock = 1024,
                        MaxThreadsPerMultiProcessor = 2560,
                        SharedMemPerBlock = 65536, // 64KB
                        Major = 9, // GCN 5.0 equivalent
                        Minor = 0
                    };
                    devices[i] = new ROCmDevice(i, fallbackProps);
                }
            }

            return devices;
        }

        /// <summary>
        /// Gets the default ROCm device (device 0).
        /// </summary>
        /// <returns>The default ROCm device or null if none available.</returns>
        public static ROCmDevice? GetDefaultDevice()
        {
            var devices = GetDevices();
            return devices.Length > 0 ? devices[0] : null;
        }

        /// <summary>
        /// Checks if ROCm is supported on this system.
        /// </summary>
        /// <returns>True if ROCm is supported; otherwise, false.</returns>
        public static bool IsSupported() => ROCmNative.IsROCmSupported();

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets the architecture name based on compute capability.
        /// </summary>
        /// <param name="major">Major version.</param>
        /// <param name="minor">Minor version.</param>
        /// <returns>Architecture name.</returns>
        private static string GetArchitectureName(int major, int minor) => major switch
        {
            6 => "GCN 3.0 (Fiji/Polaris)",
            7 => "GCN 4.0 (Vega)",
            8 => "GCN 5.0 (Vega II)",
            9 => minor switch
            {
                0 => "RDNA 1.0 (Navi 10)",
                _ => "RDNA 1.x"
            },
            10 => "RDNA 2.0 (Navi 2x)",
            11 => "RDNA 3.0 (Navi 3x)",
            12 => "RDNA 4.0 (Navi 4x)",
            _ => $"Unknown ({major}.{minor})"
        };

        /// <summary>
        /// Determines if this is a discrete GPU.
        /// </summary>
        /// <returns>True if discrete GPU; otherwise, false.</returns>
        public bool IsDiscreteGpu() => Properties.Integrated == 0;

        /// <summary>
        /// Gets a human-readable string of device capabilities.
        /// </summary>
        /// <returns>Device capabilities summary.</returns>
        public string GetCapabilitiesString() => $"Compute {ComputeCapability}, {MultiprocessorCount} CUs, " +
                   $"{MaxThreadsPerMultiprocessor} threads/CU, " +
                   $"Warp {WarpSize}, {MemorySize / (1024 * 1024)} MB, " +
                   $"{ClockRate / 1000} MHz, {Architecture}";

        #endregion

        #region Object

        /// <summary>
        /// Returns a string representation of this device.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString() => 
            $"ROCm Device {ROCmDeviceId}: {Name} ({Architecture}, {MemorySize / (1024 * 1024)} MB)";

        /// <summary>
        /// Returns the hash code of this device.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode() => ROCmDeviceId.GetHashCode();

        /// <summary>
        /// Checks for equality with another object.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public override bool Equals(object? obj) =>
            obj is ROCmDevice other && ROCmDeviceId == other.ROCmDeviceId;

        #endregion
    }
}