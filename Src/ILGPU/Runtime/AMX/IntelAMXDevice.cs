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

using ILGPU.Runtime.AMX.Native;
using ILGPU.Util;
using System;
using System.Diagnostics;
#if WINDOWS
using System.Management;
#endif

namespace ILGPU.Runtime.AMX
{
    /// <summary>
    /// Represents an Intel AMX (Advanced Matrix Extensions) device.
    /// </summary>
    [DebuggerDisplay("Intel AMX Device: {Name}")]
    public sealed class IntelAMXDevice : Device
    {
        #region Instance

        /// <summary>
        /// Gets the processor name that supports AMX.
        /// </summary>
        public string ProcessorName { get; }

        /// <summary>
        /// Gets whether this processor supports BF16 operations.
        /// </summary>
        public bool SupportsBF16 { get; }

        /// <summary>
        /// Gets whether this processor supports INT8 operations.
        /// </summary>
        public bool SupportsINT8 { get; }

        /// <summary>
        /// Gets whether this processor supports mixed precision operations.
        /// </summary>
        public bool SupportsMixedPrecision { get; }

        /// <summary>
        /// Gets the maximum tile size (16x16 for current AMX implementations).
        /// </summary>
        public int MaxTileSize { get; }

        /// <summary>
        /// Gets the number of available tiles (8 for current AMX implementations).
        /// </summary>
        public int TileCount { get; }

        /// <summary>
        /// Gets the number of CPU cores with AMX support.
        /// </summary>
        public int NumCores { get; }

        /// <summary>
        /// Gets the cache size in bytes.
        /// </summary>
        public long CacheSize { get; }

        /// <summary>
        /// Gets the base clock frequency in MHz.
        /// </summary>
        public int BaseClockFrequency { get; }

        /// <summary>
        /// Gets the maximum turbo frequency in MHz.
        /// </summary>
        public int MaxTurboFrequency { get; }

        internal IntelAMXDevice(
            string processorName,
            bool supportsBF16,
            bool supportsINT8,
            bool supportsMixedPrecision,
            int numCores,
            long cacheSize,
            int baseClockFrequency,
            int maxTurboFrequency)
        {
            ProcessorName = processorName ?? "Intel Processor with AMX";
            SupportsBF16 = supportsBF16;
            SupportsINT8 = supportsINT8;
            SupportsMixedPrecision = supportsMixedPrecision;
            MaxTileSize = 16; // Fixed for current AMX implementation
            TileCount = 8;    // Fixed for current AMX implementation
            NumCores = numCores;
            CacheSize = cacheSize;
            BaseClockFrequency = baseClockFrequency;
            MaxTurboFrequency = maxTurboFrequency;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string Name => $"Intel AMX ({ProcessorName})";

        /// <summary>
        /// Gets the total device memory in bytes (system RAM).
        /// </summary>
        public long MemorySize => GetSystemMemorySize();

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public AcceleratorType AcceleratorType => AcceleratorType.CPU; // AMX is CPU-based

        /// <summary>
        /// Gets the maximum grid size.
        /// </summary>
        public Index3D MaxGridSize
        {
            get
            {
                // For CPU-based accelerator, grid size is limited by available cores
                int maxDim = Math.Max(NumCores * 1024, 65536);
                return new Index3D(maxDim, maxDim, maxDim);
            }
        }

        /// <summary>
        /// Gets the maximum group size.
        /// </summary>
        public Index3D MaxGroupSize
        {
            get
            {
                // For AMX, group size is typically limited by tile operations
                return new Index3D(MaxTileSize, MaxTileSize, 1);
            }
        }

        /// <summary>
        /// Gets the maximum number of threads per group.
        /// </summary>
        public int MaxNumThreadsPerGroup => MaxTileSize * MaxTileSize;

        /// <summary>
        /// Gets the maximum shared memory per group in bytes.
        /// </summary>
        public long MaxSharedMemoryPerGroup => CacheSize / NumCores; // L3 cache per core

        /// <summary>
        /// Gets the maximum constant memory in bytes.
        /// </summary>
        public long MaxConstantMemory => CacheSize; // Use cache as constant memory

        /// <summary>
        /// Gets the device vendor.
        /// </summary>
        public string Vendor => "Intel";

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public override DeviceId DeviceId => new(0, AcceleratorType.CPU);

        /// <summary>
        /// Creates an accelerator for this device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created accelerator.</returns>
        public override Accelerator CreateAccelerator(Context context) =>
            new IntelAMXAccelerator(context, this);

        #endregion

        #region Device Detection

        /// <summary>
        /// Gets all available Intel AMX devices.
        /// </summary>
        /// <returns>Array of Intel AMX devices.</returns>
        public static IntelAMXDevice[] GetDevices()
        {
            try
            {
                // Check if AMX is supported
                if (!AMXNative.IsAMXSupported())
                    return Array.Empty<IntelAMXDevice>();

                var deviceInfo = DetectProcessorInfo();
                if (deviceInfo == null)
                    return Array.Empty<IntelAMXDevice>();

                return new[] { deviceInfo };
            }
            catch (Exception)
            {
                // Return fallback device on any error
                return new[]
                {
                    CreateFallbackDevice()
                };
            }
        }

        /// <summary>
        /// Gets the default Intel AMX device.
        /// </summary>
        /// <returns>The default Intel AMX device or null if none available.</returns>
        public static IntelAMXDevice? GetDefaultDevice()
        {
            var devices = GetDevices();
            return devices.Length > 0 ? devices[0] : null;
        }

        /// <summary>
        /// Checks if Intel AMX is supported on this system.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        public static bool IsSupported() => AMXNative.IsAMXSupported();

        #endregion

        #region Helper Methods

        /// <summary>
        /// Detects processor information and AMX capabilities.
        /// </summary>
        /// <returns>Intel AMX device information or null if not supported.</returns>
        private static IntelAMXDevice? DetectProcessorInfo()
        {
            try
            {
                string processorName = "Intel Processor";
                int numCores = Environment.ProcessorCount;
                long cacheSize = 32 * 1024 * 1024; // Default 32MB L3 cache
                int baseClockFrequency = 2400; // Default 2.4 GHz
                int maxTurboFrequency = 3600; // Default 3.6 GHz

                // Try to get detailed processor information on Windows
                if (OperatingSystem.IsWindows())
                {
                    try
                    {
                        using var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_Processor");
                        foreach (ManagementObject obj in searcher.Get())
                        {
                            processorName = obj["Name"]?.ToString() ?? processorName;
                            numCores = Convert.ToInt32(obj["NumberOfCores"] ?? numCores);
                            baseClockFrequency = Convert.ToInt32(obj["MaxClockSpeed"] ?? baseClockFrequency);
                            
                            // Estimate cache size (L3 cache is typically 1-4MB per core)
                            cacheSize = numCores * 2L * 1024 * 1024; // 2MB per core estimate
                            maxTurboFrequency = (int)(baseClockFrequency * 1.5); // Estimate turbo
                            break;
                        }
                    }
                    catch
                    {
                        // Use defaults if WMI fails
                    }
                }
                else
                {
                    // Try to read /proc/cpuinfo on Linux
                    try
                    {
                        var cpuinfo = System.IO.File.ReadAllText("/proc/cpuinfo");
                        if (cpuinfo.Contains("model name"))
                        {
                            var lines = cpuinfo.Split('\n');
                            foreach (var line in lines)
                            {
                                if (line.StartsWith("model name"))
                                {
                                    var parts = line.Split(':');
                                    if (parts.Length > 1)
                                        processorName = parts[1].Trim();
                                    break;
                                }
                            }
                        }
                    }
                    catch
                    {
                        // Use defaults if /proc/cpuinfo fails
                    }
                }

                // Detect AMX capabilities based on processor generation
                bool supportsBF16 = processorName.Contains("Xeon") || 
                                   processorName.Contains("12th Gen") || 
                                   processorName.Contains("13th Gen") ||
                                   processorName.Contains("14th Gen");
                                   
                bool supportsINT8 = true; // All AMX processors support INT8
                bool supportsMixedPrecision = supportsBF16; // Mixed precision available with BF16

                return new IntelAMXDevice(
                    processorName,
                    supportsBF16,
                    supportsINT8,
                    supportsMixedPrecision,
                    numCores,
                    cacheSize,
                    baseClockFrequency,
                    maxTurboFrequency);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Creates a fallback AMX device with conservative settings.
        /// </summary>
        /// <returns>Fallback Intel AMX device.</returns>
        private static IntelAMXDevice CreateFallbackDevice() => new(
                "Intel Processor with AMX",
                false, // Conservative: no BF16
                true,  // INT8 support
                false, // No mixed precision
                Environment.ProcessorCount,
                32 * 1024 * 1024, // 32MB L3 cache
                2400, // 2.4 GHz base
                3600  // 3.6 GHz turbo
            );

        /// <summary>
        /// Gets the system memory size.
        /// </summary>
        /// <returns>System memory size in bytes.</returns>
        private static long GetSystemMemorySize()
        {
            try
            {
                if (OperatingSystem.IsWindows())
                {
                    using var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_ComputerSystem");
                    foreach (ManagementObject obj in searcher.Get())
                    {
                        var totalMemory = obj["TotalPhysicalMemory"];
                        if (totalMemory != null)
                            return Convert.ToInt64(totalMemory);
                    }
                }
                else
                {
                    // Try to read /proc/meminfo on Linux
                    var meminfo = System.IO.File.ReadAllText("/proc/meminfo");
                    var lines = meminfo.Split('\n');
                    foreach (var line in lines)
                    {
                        if (line.StartsWith("MemTotal:"))
                        {
                            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                            if (parts.Length > 1 && long.TryParse(parts[1], out var memKB))
                                return memKB * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
            catch
            {
                // Ignore errors and use default
            }

            // Default to 16GB if detection fails
            return 16L * 1024 * 1024 * 1024;
        }

        /// <summary>
        /// Gets a human-readable string of device capabilities.
        /// </summary>
        /// <returns>Device capabilities summary.</returns>
        public string GetCapabilitiesString() => $"{ProcessorName}, " +
                   $"{NumCores} cores, " +
                   $"Tiles {TileCount}x{MaxTileSize}x{MaxTileSize}, " +
                   $"{MemorySize / (1024 * 1024)} MB, " +
                   $"{BaseClockFrequency} MHz, " +
                   $"BF16: {SupportsBF16}, " +
                   $"INT8: {SupportsINT8}, " +
                   $"Mixed: {SupportsMixedPrecision}";

        #endregion

        #region Object

        /// <summary>
        /// Returns a string representation of this device.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString() => 
            $"Intel AMX Device: {ProcessorName} ({NumCores} cores, {MemorySize / (1024 * 1024)} MB)";

        /// <summary>
        /// Returns the hash code of this device.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode() => ProcessorName.GetHashCode() ^ NumCores.GetHashCode();

        /// <summary>
        /// Checks for equality with another object.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public override bool Equals(object? obj) =>
            obj is IntelAMXDevice other && 
            ProcessorName == other.ProcessorName && 
            NumCores == other.NumCores;

        #endregion
    }
}