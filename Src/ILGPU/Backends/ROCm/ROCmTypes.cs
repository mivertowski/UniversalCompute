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

using ILGPU.Runtime;
using System;

namespace ILGPU.Backends.ROCm
{
    /// <summary>
    /// ROCm GPU architecture types.
    /// </summary>
    public enum ROCmArchitecture
    {
        /// <summary>
        /// Unknown or unsupported architecture.
        /// </summary>
        Unknown,

        /// <summary>
        /// Graphics Core Next 3rd generation (Fiji, Polaris).
        /// </summary>
        GCN3,

        /// <summary>
        /// Graphics Core Next 4th generation (Vega).
        /// </summary>
        GCN4,

        /// <summary>
        /// Graphics Core Next 5th generation (Vega 7nm).
        /// </summary>
        GCN5,

        /// <summary>
        /// RDNA 1st generation (Navi 10, RX 5000 series).
        /// </summary>
        RDNA1,

        /// <summary>
        /// RDNA 2nd generation (Navi 21, RX 6000 series).
        /// </summary>
        RDNA2,

        /// <summary>
        /// RDNA 3rd generation (Navi 31, RX 7000 series).
        /// </summary>
        RDNA3,

        /// <summary>
        /// CDNA 1st generation (MI100).
        /// </summary>
        CDNA1,

        /// <summary>
        /// CDNA 2nd generation (MI200 series).
        /// </summary>
        CDNA2,

        /// <summary>
        /// CDNA 3rd generation (MI300 series).
        /// </summary>
        CDNA3
    }

    /// <summary>
    /// ROCm instruction set information.
    /// </summary>
    public readonly struct ROCmInstructionSet
    {
        /// <summary>
        /// The ROCm architecture.
        /// </summary>
        public ROCmArchitecture Architecture { get; }

        /// <summary>
        /// The major version number.
        /// </summary>
        public int Major { get; }

        /// <summary>
        /// The minor version number.
        /// </summary>
        public int Minor { get; }

        /// <summary>
        /// The patch version number.
        /// </summary>
        public int Patch { get; }

        /// <summary>
        /// Initializes a new ROCm instruction set.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="major">The major version.</param>
        /// <param name="minor">The minor version.</param>
        /// <param name="patch">The patch version.</param>
        public ROCmInstructionSet(ROCmArchitecture architecture, int major, int minor, int patch = 0)
        {
            Architecture = architecture;
            Major = major;
            Minor = minor;
            Patch = patch;
        }

        /// <summary>
        /// Gets the instruction set as a version string.
        /// </summary>
        /// <returns>The version string.</returns>
        public override string ToString() => $"{Architecture} {Major}.{Minor}.{Patch}";

        /// <summary>
        /// Creates an instruction set from GFX target string.
        /// </summary>
        /// <param name="gfxTarget">The GFX target (e.g., "gfx900").</param>
        /// <returns>The instruction set.</returns>
        public static ROCmInstructionSet FromGfxTarget(string gfxTarget)
        {
            return gfxTarget switch
            {
                "gfx803" => new ROCmInstructionSet(ROCmArchitecture.GCN3, 8, 0, 3),
                "gfx900" => new ROCmInstructionSet(ROCmArchitecture.GCN4, 9, 0, 0),
                "gfx906" => new ROCmInstructionSet(ROCmArchitecture.GCN5, 9, 0, 6),
                "gfx908" => new ROCmInstructionSet(ROCmArchitecture.CDNA1, 9, 0, 8),
                "gfx90a" => new ROCmInstructionSet(ROCmArchitecture.CDNA2, 9, 0, 10),
                "gfx942" => new ROCmInstructionSet(ROCmArchitecture.CDNA3, 9, 4, 2),
                "gfx1010" => new ROCmInstructionSet(ROCmArchitecture.RDNA1, 10, 1, 0),
                "gfx1030" => new ROCmInstructionSet(ROCmArchitecture.RDNA2, 10, 3, 0),
                "gfx1100" => new ROCmInstructionSet(ROCmArchitecture.RDNA3, 11, 0, 0),
                _ => new ROCmInstructionSet(ROCmArchitecture.Unknown, 0, 0, 0)
            };
        }
    }

    /// <summary>
    /// ROCm accelerator capabilities.
    /// </summary>
    public sealed class ROCmCapabilities
    {
        /// <summary>
        /// Gets the ROCm architecture.
        /// </summary>
        public ROCmArchitecture Architecture { get; }

        /// <summary>
        /// Gets the compute units (multiprocessors).
        /// </summary>
        public int ComputeUnits { get; }

        /// <summary>
        /// Gets the wavefront size (similar to CUDA warp size).
        /// </summary>
        public int WavefrontSize { get; }

        /// <summary>
        /// Gets the maximum wavefronts per compute unit.
        /// </summary>
        public int MaxWavefrontsPerComputeUnit { get; }

        /// <summary>
        /// Gets the local data share (LDS) size per compute unit.
        /// </summary>
        public int LocalDataShareSize { get; }

        /// <summary>
        /// Gets whether the device supports cooperative launch.
        /// </summary>
        public bool SupportsCooperativeLaunch { get; }

        /// <summary>
        /// Gets whether the device supports concurrent kernels.
        /// </summary>
        public bool SupportsConcurrentKernels { get; }

        /// <summary>
        /// Gets whether the device supports unified memory.
        /// </summary>
        public bool SupportsUnifiedMemory { get; }

        /// <summary>
        /// Gets whether the device supports FP16 operations.
        /// </summary>
        public bool SupportsFP16 { get; }

        /// <summary>
        /// Gets whether the device supports packed FP16 operations.
        /// </summary>
        public bool SupportsPackedFP16 { get; }

        /// <summary>
        /// Gets whether the device supports INT8 operations.
        /// </summary>
        public bool SupportsINT8 { get; }

        /// <summary>
        /// Gets whether the device supports matrix/tensor operations.
        /// </summary>
        public bool SupportsMatrixOps { get; }

        /// <summary>
        /// Initializes new ROCm capabilities.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="computeUnits">The number of compute units.</param>
        /// <param name="wavefrontSize">The wavefront size.</param>
        /// <param name="maxWavefrontsPerComputeUnit">Max wavefronts per compute unit.</param>
        /// <param name="localDataShareSize">Local data share size.</param>
        /// <param name="supportsCooperativeLaunch">Supports cooperative launch.</param>
        /// <param name="supportsConcurrentKernels">Supports concurrent kernels.</param>
        /// <param name="supportsUnifiedMemory">Supports unified memory.</param>
        /// <param name="supportsFP16">Supports FP16.</param>
        /// <param name="supportsPackedFP16">Supports packed FP16.</param>
        /// <param name="supportsINT8">Supports INT8.</param>
        /// <param name="supportsMatrixOps">Supports matrix operations.</param>
        public ROCmCapabilities(
            ROCmArchitecture architecture,
            int computeUnits,
            int wavefrontSize = 64,
            int maxWavefrontsPerComputeUnit = 40,
            int localDataShareSize = 65536,
            bool supportsCooperativeLaunch = false,
            bool supportsConcurrentKernels = true,
            bool supportsUnifiedMemory = false,
            bool supportsFP16 = false,
            bool supportsPackedFP16 = false,
            bool supportsINT8 = false,
            bool supportsMatrixOps = false)
        {
            Architecture = architecture;
            ComputeUnits = computeUnits;
            WavefrontSize = wavefrontSize;
            MaxWavefrontsPerComputeUnit = maxWavefrontsPerComputeUnit;
            LocalDataShareSize = localDataShareSize;
            SupportsCooperativeLaunch = supportsCooperativeLaunch;
            SupportsConcurrentKernels = supportsConcurrentKernels;
            SupportsUnifiedMemory = supportsUnifiedMemory;
            SupportsFP16 = supportsFP16;
            SupportsPackedFP16 = supportsPackedFP16;
            SupportsINT8 = supportsINT8;
            SupportsMatrixOps = supportsMatrixOps;
        }

        /// <summary>
        /// Creates capabilities from device properties.
        /// </summary>
        /// <param name="properties">The device properties.</param>
        /// <param name="architecture">The detected architecture.</param>
        /// <returns>The capabilities.</returns>
        internal static ROCmCapabilities FromDeviceProperties(
            Runtime.ROCm.Native.HipDeviceProperties properties,
            ROCmArchitecture architecture)
        {
            // Determine advanced features based on architecture
            var supportsFP16 = architecture >= ROCmArchitecture.GCN4;
            var supportsPackedFP16 = architecture >= ROCmArchitecture.GCN5;
            var supportsINT8 = architecture >= ROCmArchitecture.CDNA1;
            var supportsMatrixOps = architecture >= ROCmArchitecture.CDNA2;
            var supportsUnifiedMemory = properties.UnifiedAddressing != 0;
            var supportsCooperativeLaunch = properties.CooperativeLaunch != 0;
            var supportsConcurrentKernels = properties.ConcurrentKernels != 0;

            // Wavefront size is typically 64 for AMD GPUs, but can be 32 on some RDNA
            var wavefrontSize = architecture >= ROCmArchitecture.RDNA1 ? 32 : 64;

            return new ROCmCapabilities(
                architecture,
                properties.MultiProcessorCount,
                wavefrontSize,
                properties.MaxThreadsPerMultiProcessor / wavefrontSize,
                (int)properties.SharedMemPerMultiprocessor,
                supportsCooperativeLaunch,
                supportsConcurrentKernels,
                supportsUnifiedMemory,
                supportsFP16,
                supportsPackedFP16,
                supportsINT8,
                supportsMatrixOps);
        }

        /// <summary>
        /// Gets the maximum number of threads per compute unit.
        /// </summary>
        public int MaxThreadsPerComputeUnit => MaxWavefrontsPerComputeUnit * WavefrontSize;

        /// <summary>
        /// Gets whether this device supports the specified precision.
        /// </summary>
        /// <param name="precision">The precision to check.</param>
        /// <returns>True if supported; otherwise, false.</returns>
        public bool SupportsPrecision(ROCmPrecision precision)
        {
            return precision switch
            {
                ROCmPrecision.FP32 => true,
                ROCmPrecision.FP64 => true, // Most ROCm devices support FP64
                ROCmPrecision.FP16 => SupportsFP16,
                ROCmPrecision.PackedFP16 => SupportsPackedFP16,
                ROCmPrecision.INT8 => SupportsINT8,
                ROCmPrecision.INT16 => true,
                ROCmPrecision.INT32 => true,
                ROCmPrecision.INT64 => true,
                _ => false
            };
        }
    }

    /// <summary>
    /// ROCm precision types.
    /// </summary>
    public enum ROCmPrecision
    {
        /// <summary>32-bit floating point.</summary>
        FP32,

        /// <summary>64-bit floating point.</summary>
        FP64,

        /// <summary>16-bit floating point.</summary>
        FP16,

        /// <summary>Packed 16-bit floating point (2 values per 32-bit register).</summary>
        PackedFP16,

        /// <summary>8-bit integer.</summary>
        INT8,

        /// <summary>16-bit integer.</summary>
        INT16,

        /// <summary>32-bit integer.</summary>
        INT32,

        /// <summary>64-bit integer.</summary>
        INT64
    }
}