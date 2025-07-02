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

#if ENABLE_ONEAPI_ACCELERATOR
namespace ILGPU.Backends.OneAPI
{
    /// <summary>
    /// Represents an Intel OneAPI device.
    /// </summary>
    public sealed class OneAPIDevice
    {
        /// <summary>
        /// Initializes a new instance of the OneAPIDevice class.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="platformId">The platform identifier.</param>
        public OneAPIDevice(IntPtr deviceId, IntPtr platformId)
        {
            DeviceId = deviceId;
            PlatformId = platformId;

            // Query device information
            Name = OneAPINative.GetDeviceInfo<string>(deviceId, OneAPIDeviceInfo.Name);
            Vendor = OneAPINative.GetDeviceInfo<string>(deviceId, OneAPIDeviceInfo.Vendor);
            DeviceType = OneAPINative.GetDeviceInfo<OneAPIDeviceType>(deviceId, OneAPIDeviceInfo.Type);
            DriverVersion = OneAPINative.GetDeviceInfo<string>(deviceId, OneAPIDeviceInfo.DriverVersion);
            
            // Create ILGPU device representation
            Device = CreateDevice();
        }

        /// <summary>
        /// Gets the native device ID.
        /// </summary>
        public IntPtr DeviceId { get; }

        /// <summary>
        /// Gets the native platform ID.
        /// </summary>
        public IntPtr PlatformId { get; }

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the device vendor.
        /// </summary>
        public string Vendor { get; }

        /// <summary>
        /// Gets the device type.
        /// </summary>
        public OneAPIDeviceType DeviceType { get; }

        /// <summary>
        /// Gets the driver version.
        /// </summary>
        public string DriverVersion { get; }

        /// <summary>
        /// Gets the ILGPU device representation.
        /// </summary>
        public Device Device { get; }

        /// <summary>
        /// Checks if this is an Intel GPU device.
        /// </summary>
        public bool IsIntelGPU => Vendor.Contains("Intel", StringComparison.OrdinalIgnoreCase) && 
                                  DeviceType == OneAPIDeviceType.GPU;

        /// <summary>
        /// Checks if this is an Intel CPU device.
        /// </summary>
        public bool IsIntelCPU => Vendor.Contains("Intel", StringComparison.OrdinalIgnoreCase) && 
                                  DeviceType == OneAPIDeviceType.CPU;

        /// <summary>
        /// Checks if this is an Intel FPGA device.
        /// </summary>
        public bool IsIntelFPGA => Vendor.Contains("Intel", StringComparison.OrdinalIgnoreCase) && 
                                   DeviceType == OneAPIDeviceType.FPGA;

        /// <summary>
        /// Gets the device architecture.
        /// </summary>
        public IntelArchitecture GetArchitecture()
        {
            if (!IsIntelGPU)
                return IntelArchitecture.Unknown;

            // Detect Intel GPU architecture based on device name
            var nameLower = Name.ToLowerInvariant();
            
            if (nameLower.Contains("xe-lp") || nameLower.Contains("gen12"))
                return IntelArchitecture.XeLP;
            else if (nameLower.Contains("xe-hpg") || nameLower.Contains("arc"))
                return IntelArchitecture.XeHPG;
            else if (nameLower.Contains("xe-hp"))
                return IntelArchitecture.XeHP;
            else if (nameLower.Contains("xe-hpc") || nameLower.Contains("ponte vecchio"))
                return IntelArchitecture.XeHPC;
            else if (nameLower.Contains("xe2") || nameLower.Contains("battlemage"))
                return IntelArchitecture.Xe2;
            else if (nameLower.Contains("gen11"))
                return IntelArchitecture.Gen11;
            else if (nameLower.Contains("gen9"))
                return IntelArchitecture.Gen9;
            
            return IntelArchitecture.Unknown;
        }

        private Device CreateDevice()
        {
            var acceleratorType = DeviceType switch
            {
                OneAPIDeviceType.GPU => AcceleratorType.OneAPI,
                OneAPIDeviceType.CPU => AcceleratorType.CPU,
                OneAPIDeviceType.FPGA => AcceleratorType.OneAPI,
                _ => AcceleratorType.OneAPI
            };

            return new Device(
                name: Name,
                acceleratorId: 0, // Will be set by Context
                acceleratorType: acceleratorType);
        }

        /// <summary>
        /// Returns a string representation of the device.
        /// </summary>
        /// <returns>A string describing the device.</returns>
        public override string ToString() => $"{Name} ({DeviceType}, {Vendor})";

        #region Static Device Discovery

        /// <summary>
        /// Gets all available OneAPI devices.
        /// </summary>
        /// <returns>An array of OneAPI devices.</returns>
        public static OneAPIDevice[] GetDevices()
        {
            try
            {
                var devices = new List<OneAPIDevice>();
                var platforms = Native.OneAPINative.GetPlatforms();

                foreach (var platform in platforms)
                {
                    var platformDevices = Native.OneAPINative.GetDevices(platform);
                    foreach (var device in platformDevices)
                    {
                        try
                        {
                            devices.Add(new OneAPIDevice(device, platform));
                        }
                        catch
                        {
                            // Skip devices that cannot be initialized
                        }
                    }
                }

                return devices.ToArray();
            }
            catch
            {
                return Array.Empty<OneAPIDevice>();
            }
        }

        /// <summary>
        /// Gets OneAPI devices of a specific type.
        /// </summary>
        /// <param name="deviceType">The device type to filter by.</param>
        /// <returns>An array of OneAPI devices of the specified type.</returns>
        public static OneAPIDevice[] GetDevices(OneAPIDeviceType deviceType)
        {
            return GetDevices().Where(d => d.DeviceType == deviceType).ToArray();
        }

        /// <summary>
        /// Gets the default OneAPI device (typically the first GPU, then CPU).
        /// </summary>
        /// <returns>The default OneAPI device, or null if none available.</returns>
        public static OneAPIDevice? GetDefaultDevice()
        {
            var devices = GetDevices();
            
            // Prefer Intel GPUs first
            var intelGPU = devices.FirstOrDefault(d => d.IsIntelGPU);
            if (intelGPU != null) return intelGPU;
            
            // Then any GPU
            var anyGPU = devices.FirstOrDefault(d => d.DeviceType == OneAPIDeviceType.GPU);
            if (anyGPU != null) return anyGPU;
            
            // Then Intel CPUs
            var intelCPU = devices.FirstOrDefault(d => d.IsIntelCPU);
            if (intelCPU != null) return intelCPU;
            
            // Finally any device
            return devices.FirstOrDefault();
        }

        /// <summary>
        /// Gets the best OneAPI device for compute workloads.
        /// </summary>
        /// <returns>The best OneAPI device, or null if none available.</returns>
        public static OneAPIDevice? GetBestDevice()
        {
            var devices = GetDevices();
            
            // Rank devices by computational capability
            return devices
                .Where(d => d.DeviceType == OneAPIDeviceType.GPU || d.DeviceType == OneAPIDeviceType.CPU)
                .OrderByDescending(d => GetDeviceScore(d))
                .FirstOrDefault();
        }

        /// <summary>
        /// Gets Intel GPU devices only.
        /// </summary>
        /// <returns>An array of Intel GPU devices.</returns>
        public static OneAPIDevice[] GetIntelGPUs()
        {
            return GetDevices().Where(d => d.IsIntelGPU).ToArray();
        }

        /// <summary>
        /// Calculates a score for device ranking.
        /// </summary>
        private static int GetDeviceScore(OneAPIDevice device)
        {
            var score = 0;
            
            // Prefer Intel devices
            if (device.Vendor.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                score += 1000;
            
            // GPU gets higher score than CPU
            if (device.DeviceType == OneAPIDeviceType.GPU)
                score += 500;
            else if (device.DeviceType == OneAPIDeviceType.CPU)
                score += 100;
            
            // Architecture-based scoring
            var architecture = device.GetArchitecture();
            score += architecture switch
            {
                IntelArchitecture.Xe2 => 50,
                IntelArchitecture.XeHPC => 45,
                IntelArchitecture.XeHPG => 40,
                IntelArchitecture.XeHP => 35,
                IntelArchitecture.XeLP => 30,
                IntelArchitecture.Gen11 => 20,
                IntelArchitecture.Gen9 => 10,
                _ => 0
            };
            
            return score;
        }

        #endregion
    }

    /// <summary>
    /// OneAPI device types.
    /// </summary>
    public enum OneAPIDeviceType
    {
        /// <summary>
        /// CPU device.
        /// </summary>
        CPU = 1,

        /// <summary>
        /// GPU device.
        /// </summary>
        GPU = 2,

        /// <summary>
        /// FPGA device.
        /// </summary>
        FPGA = 3,

        /// <summary>
        /// Custom accelerator device.
        /// </summary>
        Accelerator = 4,

        /// <summary>
        /// Default device type.
        /// </summary>
        Default = 5,

        /// <summary>
        /// Host device.
        /// </summary>
        Host = 6
    }

    /// <summary>
    /// Intel GPU architecture generations.
    /// </summary>
    public enum IntelArchitecture
    {
        /// <summary>
        /// Unknown architecture.
        /// </summary>
        Unknown,

        /// <summary>
        /// Gen9 architecture (Skylake, Kaby Lake, Coffee Lake).
        /// </summary>
        Gen9,

        /// <summary>
        /// Gen11 architecture (Ice Lake).
        /// </summary>
        Gen11,

        /// <summary>
        /// Xe-LP architecture (Tiger Lake, DG1).
        /// </summary>
        XeLP,

        /// <summary>
        /// Xe-HPG architecture (Arc Alchemist).
        /// </summary>
        XeHPG,

        /// <summary>
        /// Xe-HP architecture (Data Center GPU).
        /// </summary>
        XeHP,

        /// <summary>
        /// Xe-HPC architecture (Ponte Vecchio).
        /// </summary>
        XeHPC,

        /// <summary>
        /// Xe2 architecture (Battlemage and beyond).
        /// </summary>
        Xe2
    }

    /// <summary>
    /// OneAPI device capabilities.
    /// </summary>
    public sealed class OneAPICapabilities
    {
        /// <summary>
        /// Gets or sets the device type.
        /// </summary>
        public OneAPIDeviceType DeviceType { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of compute units.
        /// </summary>
        public int MaxComputeUnits { get; set; }

        /// <summary>
        /// Gets or sets the maximum work group size.
        /// </summary>
        public int MaxWorkGroupSize { get; set; }

        /// <summary>
        /// Gets or sets the maximum work item sizes per dimension.
        /// </summary>
        public long[] MaxWorkItemSizes { get; set; }

        /// <summary>
        /// Gets or sets the global memory size in bytes.
        /// </summary>
        public long GlobalMemorySize { get; set; }

        /// <summary>
        /// Gets or sets the local memory size in bytes.
        /// </summary>
        public long LocalMemorySize { get; set; }

        /// <summary>
        /// Gets or sets the maximum constant buffer size in bytes.
        /// </summary>
        public long MaxConstantBufferSize { get; set; }

        /// <summary>
        /// Gets or sets the global memory bandwidth in bytes per second.
        /// </summary>
        public long GlobalMemoryBandwidth { get; set; }

        /// <summary>
        /// Gets or sets the amount of global memory currently used.
        /// </summary>
        public long GlobalMemoryUsed { get; set; }

        /// <summary>
        /// Gets or sets whether Unified Shared Memory (USM) is supported.
        /// </summary>
        public bool SupportsUSM { get; set; }

        /// <summary>
        /// Gets or sets whether FP16 operations are supported.
        /// </summary>
        public bool SupportsFP16 { get; set; }

        /// <summary>
        /// Gets or sets whether FP64 operations are supported.
        /// </summary>
        public bool SupportsFP64 { get; set; }

        /// <summary>
        /// Gets or sets whether subgroups are supported.
        /// </summary>
        public bool SupportsSubgroups { get; set; }

        /// <summary>
        /// Gets or sets the subgroup size (warp size).
        /// </summary>
        public int SubGroupSize { get; set; }

        /// <summary>
        /// Gets or sets the number of execution units (Intel-specific).
        /// </summary>
        public int NumExecutionUnits { get; set; }

        /// <summary>
        /// Gets or sets the maximum threads per execution unit (Intel-specific).
        /// </summary>
        public int MaxThreadsPerEU { get; set; }

        /// <summary>
        /// Gets or sets the preferred work group size multiple.
        /// </summary>
        public int PreferredWorkGroupSizeMultiple { get; set; }

        /// <summary>
        /// Gets the estimated peak performance in GFLOPS.
        /// </summary>
        /// <param name="dataType">The data type.</param>
        /// <returns>Estimated GFLOPS.</returns>
        public double GetPeakPerformance(OneAPIDataType dataType)
        {
            // Estimate based on execution units and frequency
            var baseGFLOPS = NumExecutionUnits * MaxThreadsPerEU * 2.0; // Assuming 2 ops per cycle
            
            return dataType switch
            {
                OneAPIDataType.Float32 => baseGFLOPS,
                OneAPIDataType.Float16 => baseGFLOPS * 2, // FP16 typically has 2x throughput
                OneAPIDataType.BFloat16 => baseGFLOPS * 2,
                OneAPIDataType.Int8 => baseGFLOPS * 4, // INT8 typically has 4x throughput
                _ => baseGFLOPS
            };
        }

        /// <summary>
        /// Checks if the device supports a specific feature.
        /// </summary>
        /// <param name="feature">The feature to check.</param>
        /// <returns>True if supported; otherwise, false.</returns>
        public bool SupportsFeature(OneAPIFeature feature)
        {
            return feature switch
            {
                OneAPIFeature.UnifiedSharedMemory => SupportsUSM,
                OneAPIFeature.SubGroups => SupportsSubgroups,
                OneAPIFeature.HalfPrecision => SupportsFP16,
                OneAPIFeature.DoublePrecision => SupportsFP64,
                _ => false
            };
        }
    }

    /// <summary>
    /// OneAPI features.
    /// </summary>
    public enum OneAPIFeature
    {
        /// <summary>
        /// Unified Shared Memory support.
        /// </summary>
        UnifiedSharedMemory,

        /// <summary>
        /// Sub-group (warp) operations.
        /// </summary>
        SubGroups,

        /// <summary>
        /// Half precision (FP16) support.
        /// </summary>
        HalfPrecision,

        /// <summary>
        /// Double precision (FP64) support.
        /// </summary>
        DoublePrecision,

        /// <summary>
        /// Atomic operations support.
        /// </summary>
        Atomics,

        /// <summary>
        /// Image support.
        /// </summary>
        Images
    }

    /// <summary>
    /// OneAPI data types.
    /// </summary>
    public enum OneAPIDataType
    {
        /// <summary>
        /// 32-bit floating point.
        /// </summary>
        Float32,

        /// <summary>
        /// 16-bit floating point.
        /// </summary>
        Float16,

        /// <summary>
        /// 16-bit brain floating point.
        /// </summary>
        BFloat16,

        /// <summary>
        /// 8-bit integer.
        /// </summary>
        Int8,

        /// <summary>
        /// 32-bit integer.
        /// </summary>
        Int32,

        /// <summary>
        /// 64-bit floating point.
        /// </summary>
        Float64
    }

    /// <summary>
    /// OneAPI compilation options.
    /// </summary>
    public sealed class OneAPICompilationOptions
    {
        /// <summary>
        /// Gets the default compilation options.
        /// </summary>
        public static OneAPICompilationOptions Default => new OneAPICompilationOptions();

        /// <summary>
        /// Gets or sets the build options string.
        /// </summary>
        public string BuildOptions { get; set; } = "-cl-std=CL3.0 -cl-mad-enable";

        /// <summary>
        /// Gets or sets whether to enable fast math.
        /// </summary>
        public bool EnableFastMath { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to enable kernel profiling.
        /// </summary>
        public bool EnableProfiling { get; set; }

        /// <summary>
        /// Gets or sets the optimization level.
        /// </summary>
        public OneAPIOptimizationLevel OptimizationLevel { get; set; } = OneAPIOptimizationLevel.O2;

        /// <summary>
        /// Gets or sets target-specific options.
        /// </summary>
        public string TargetOptions { get; set; }

        /// <summary>
        /// Builds the complete options string.
        /// </summary>
        /// <returns>The complete build options.</returns>
        public string GetBuildOptions()
        {
            var options = BuildOptions;
            
            if (EnableFastMath)
                options += " -cl-fast-relaxed-math";
            
            if (EnableProfiling)
                options += " -cl-enable-profiling";
            
            options += OptimizationLevel switch
            {
                OneAPIOptimizationLevel.O0 => " -O0",
                OneAPIOptimizationLevel.O1 => " -O1",
                OneAPIOptimizationLevel.O2 => " -O2",
                OneAPIOptimizationLevel.O3 => " -O3",
                _ => " -O2"
            };
            
            if (!string.IsNullOrEmpty(TargetOptions))
                options += " " + TargetOptions;
            
            return options;
        }
    }

    /// <summary>
    /// OneAPI optimization levels.
    /// </summary>
    public enum OneAPIOptimizationLevel
    {
        /// <summary>
        /// No optimization.
        /// </summary>
        O0,

        /// <summary>
        /// Basic optimization.
        /// </summary>
        O1,

        /// <summary>
        /// Standard optimization.
        /// </summary>
        O2,

        /// <summary>
        /// Aggressive optimization.
        /// </summary>
        O3
    }
}
#endif