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

using System;
using System.Collections.Generic;
using System.Linq;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Specifies the preferred memory placement strategy for universal buffers.
    /// </summary>
    public enum MemoryPlacement
    {
        /// <summary>
        /// Automatically select the optimal memory placement based on usage patterns and hardware.
        /// </summary>
        Auto,

        /// <summary>
        /// Use Apple's unified memory architecture (macOS/iOS with Apple Silicon).
        /// Provides zero-copy access between CPU and GPU.
        /// </summary>
        AppleUnified,

        /// <summary>
        /// Use CUDA managed memory for NVIDIA GPUs.
        /// Allows automatic migration between CPU and GPU.
        /// </summary>
        CudaManaged,

        /// <summary>
        /// Use Intel's shared memory for integrated GPUs.
        /// Optimal for Intel integrated graphics with shared system memory.
        /// </summary>
        IntelShared,

        /// <summary>
        /// Use host pinned memory for fast CPU-GPU transfers.
        /// Provides non-pageable memory for efficient DMA transfers.
        /// </summary>
        HostPinned,

        /// <summary>
        /// Standard device memory allocation.
        /// Traditional GPU memory with explicit transfers.
        /// </summary>
        DeviceLocal,

        /// <summary>
        /// Host memory accessible to accelerators.
        /// CPU memory that can be accessed by GPU with potential performance penalty.
        /// </summary>
        HostAccessible,

        /// <summary>
        /// Memory optimized for read-only data.
        /// Ideal for constants, lookup tables, and read-only inputs.
        /// </summary>
        ReadOnlyOptimized,

        /// <summary>
        /// Memory optimized for write-only operations.
        /// Ideal for output buffers and intermediate results.
        /// </summary>
        WriteOnlyOptimized,

        /// <summary>
        /// Memory placement optimized for streaming operations.
        /// Suitable for large sequential data processing.
        /// </summary>
        StreamingOptimized,

        /// <summary>
        /// Memory placement optimized for random access patterns.
        /// Provides good performance for irregular memory access.
        /// </summary>
        RandomAccessOptimized
    }

    /// <summary>
    /// Describes the characteristics and capabilities of a memory placement.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the MemoryPlacementInfo struct.
    /// </remarks>
    public readonly struct MemoryPlacementInfo(
        MemoryPlacement placement,
        bool supportsZeroCopy,
        bool supportsAutoMigration,
        float relativeBandwidth,
        float relativeLatency,
        bool isAvailable,
        string description)
    {
        /// <summary>
        /// Gets the memory placement type.
        /// </summary>
        public MemoryPlacement Placement { get; } = placement;

        /// <summary>
        /// Gets whether this placement supports zero-copy access.
        /// </summary>
        public bool SupportsZeroCopy { get; } = supportsZeroCopy;

        /// <summary>
        /// Gets whether this placement supports automatic migration.
        /// </summary>
        public bool SupportsAutoMigration { get; } = supportsAutoMigration;

        /// <summary>
        /// Gets the relative bandwidth of this memory placement (1.0 = baseline).
        /// </summary>
        public float RelativeBandwidth { get; } = relativeBandwidth;

        /// <summary>
        /// Gets the relative latency of this memory placement (1.0 = baseline).
        /// </summary>
        public float RelativeLatency { get; } = relativeLatency;

        /// <summary>
        /// Gets whether this placement is available on the current platform.
        /// </summary>
        public bool IsAvailable { get; } = isAvailable;

        /// <summary>
        /// Gets a description of this memory placement.
        /// </summary>
        public string Description { get; } = description;
    }

    /// <summary>
    /// Provides information about memory placement options on the current platform.
    /// </summary>
    public static class MemoryPlacementCapabilities
    {
        /// <summary>
        /// Gets information about all available memory placements on the current platform.
        /// </summary>
        /// <returns>An array of memory placement information.</returns>
        public static MemoryPlacementInfo[] GetAvailablePlacements()
        {
            var placements = new List<MemoryPlacementInfo>
            {
                // Add standard placements that are always available
                new(
                MemoryPlacement.DeviceLocal,
                supportsZeroCopy: false,
                supportsAutoMigration: false,
                relativeBandwidth: 1.0f,
                relativeLatency: 1.0f,
                isAvailable: true,
                description: "Standard device memory"),
                new(
                MemoryPlacement.HostPinned,
                supportsZeroCopy: false,
                supportsAutoMigration: false,
                relativeBandwidth: 0.8f,
                relativeLatency: 2.0f,
                isAvailable: true,
                description: "Host pinned memory")
            };

            // Platform-specific placements
            if (PlatformDetection.IsAppleSilicon)
            {
                placements.Add(new MemoryPlacementInfo(
                    MemoryPlacement.AppleUnified,
                    supportsZeroCopy: true,
                    supportsAutoMigration: true,
                    relativeBandwidth: 1.2f,
                    relativeLatency: 0.5f,
                    isAvailable: true,
                    description: "Apple unified memory architecture"));
            }

            if (PlatformDetection.HasCudaSupport)
            {
                placements.Add(new MemoryPlacementInfo(
                    MemoryPlacement.CudaManaged,
                    supportsZeroCopy: true,
                    supportsAutoMigration: true,
                    relativeBandwidth: 0.9f,
                    relativeLatency: 1.5f,
                    isAvailable: true,
                    description: "CUDA managed memory"));
            }

            if (PlatformDetection.HasIntelIntegratedGpu)
            {
                placements.Add(new MemoryPlacementInfo(
                    MemoryPlacement.IntelShared,
                    supportsZeroCopy: true,
                    supportsAutoMigration: false,
                    relativeBandwidth: 0.7f,
                    relativeLatency: 0.8f,
                    isAvailable: true,
                    description: "Intel integrated GPU shared memory"));
            }

            return placements.ToArray();
        }

        /// <summary>
        /// Gets the optimal memory placement for the specified usage pattern.
        /// </summary>
        /// <param name="accessPattern">The expected memory access pattern.</param>
        /// <param name="dataSize">The size of the data in bytes.</param>
        /// <param name="isReadOnly">Whether the data is read-only.</param>
        /// <returns>The recommended memory placement.</returns>
        public static MemoryPlacement GetOptimalPlacement(
            MemoryAccessPattern accessPattern,
            long dataSize,
            bool isReadOnly = false)
        {
            var availablePlacements = GetAvailablePlacements();

            // For very small data, prefer zero-copy if available
            if (dataSize < 64 * 1024) // 64KB
            {
                var zeroCopyPlacement = availablePlacements
                    .Where(p => p.SupportsZeroCopy && p.IsAvailable)
                    .OrderBy(p => p.RelativeLatency)
                    .FirstOrDefault();

                if (zeroCopyPlacement.IsAvailable)
                    return zeroCopyPlacement.Placement;
            }

            // For large data with sequential access, prefer high bandwidth
            if (accessPattern == MemoryAccessPattern.Sequential && dataSize > 1024 * 1024) // 1MB
            {
                var highBandwidthPlacement = availablePlacements
                    .Where(p => p.IsAvailable)
                    .OrderByDescending(p => p.RelativeBandwidth)
                    .First();

                return highBandwidthPlacement.Placement;
            }

            // For random access, prefer low latency
            if (accessPattern == MemoryAccessPattern.Random)
            {
                var lowLatencyPlacement = availablePlacements
                    .Where(p => p.IsAvailable)
                    .OrderBy(p => p.RelativeLatency)
                    .First();

                return lowLatencyPlacement.Placement;
            }

            // Default to device local memory
            return MemoryPlacement.DeviceLocal;
        }
    }

}

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Provides platform detection capabilities for memory placement optimization.
    /// </summary>
    internal static class PlatformDetection
    {
        /// <summary>
        /// Gets whether the current platform is Apple Silicon (M1/M2/M3).
        /// </summary>
        public static bool IsAppleSilicon
        {
            get
            {
                if (!OperatingSystem.IsMacOS())
                    return false;

                // Check for Apple Silicon architecture
                return System.Runtime.Intrinsics.Arm.ArmBase.Arm64.IsSupported;
            }
        }

        /// <summary>
        /// Gets whether CUDA support is available on the current platform.
        /// </summary>
        public static bool HasCudaSupport
        {
            get
            {
                try
                {
                    // This would check for CUDA runtime availability
                    // Implementation would depend on ILGPU's CUDA detection
                    return Environment.OSVersion.Platform != PlatformID.MacOSX;
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets whether Intel integrated GPU is available.
        /// </summary>
        public static bool HasIntelIntegratedGpu
        {
            get
            {
                // This would check for Intel integrated graphics
                // Implementation would depend on platform-specific detection
                return true; // Simplified for now
            }
        }
    }
}