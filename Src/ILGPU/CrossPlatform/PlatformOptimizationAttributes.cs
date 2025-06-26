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

namespace ILGPU.CrossPlatform
{
    /// <summary>
    /// Base attribute for platform-specific optimizations.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
    public abstract class PlatformOptimizationAttribute : Attribute
    {
        /// <summary>
        /// Gets the target platform for this optimization.
        /// </summary>
        public abstract string TargetPlatform { get; }

        /// <summary>
        /// Gets or sets the optimization level (0-3, where 3 is most aggressive).
        /// </summary>
        public int OptimizationLevel { get; set; } = 2;

        /// <summary>
        /// Gets or sets whether this optimization is required or optional.
        /// </summary>
        public bool Required { get; set; }
    }

    /// <summary>
    /// Specifies Apple-specific optimizations for kernels.
    /// </summary>
    public sealed class AppleOptimizationAttribute : PlatformOptimizationAttribute
    {
        /// <inheritdoc/>
        public override string TargetPlatform => "Apple";

        /// <summary>
        /// Gets or sets whether to use Apple Matrix Extension (AMX) instructions.
        /// </summary>
        public bool UseAMX { get; set; }

        /// <summary>
        /// Gets or sets whether to use Apple Neural Engine (ANE) when available.
        /// </summary>
        public bool UseNeuralEngine { get; set; }

        /// <summary>
        /// Gets or sets whether to use Metal Performance Shaders.
        /// </summary>
        public bool UseMetalPerformanceShaders { get; set; } = true;

        /// <summary>
        /// Gets or sets the preferred Metal shader language version.
        /// </summary>
        public string MetalVersion { get; set; } = "2.4";
    }

    /// <summary>
    /// Specifies Intel-specific optimizations for kernels.
    /// </summary>
    public sealed class IntelOptimizationAttribute : PlatformOptimizationAttribute
    {
        /// <inheritdoc/>
        public override string TargetPlatform => "Intel";

        /// <summary>
        /// Gets or sets whether to use Intel Advanced Matrix Extensions (AMX).
        /// </summary>
        public bool UseAMX { get; set; }

        /// <summary>
        /// Gets or sets whether to use Intel AVX-512 instructions.
        /// </summary>
        public bool UseAVX512 { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to use Intel Math Kernel Library (MKL) routines.
        /// </summary>
        public bool UseMKL { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to use Intel Deep Learning Boost (DL Boost).
        /// </summary>
        public bool UseDLBoost { get; set; }

        /// <summary>
        /// Gets or sets whether to use Intel Neural Processing Unit (NPU) when available.
        /// </summary>
        public bool UseNPU { get; set; }
    }

    /// <summary>
    /// Specifies NVIDIA-specific optimizations for kernels.
    /// </summary>
    public sealed class NvidiaOptimizationAttribute : PlatformOptimizationAttribute
    {
        /// <inheritdoc/>
        public override string TargetPlatform => "NVIDIA";

        /// <summary>
        /// Gets or sets whether to use Tensor Cores for mixed precision operations.
        /// </summary>
        public bool UseTensorCores { get; set; }

        /// <summary>
        /// Gets or sets the minimum compute capability required.
        /// </summary>
        public string MinComputeCapability { get; set; } = "7.0";

        /// <summary>
        /// Gets or sets whether to use cuBLAS for BLAS operations.
        /// </summary>
        public bool UseCuBLAS { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to use cuDNN for deep learning operations.
        /// </summary>
        public bool UseCuDNN { get; set; }

        /// <summary>
        /// Gets or sets whether to use PTX assembly optimizations.
        /// </summary>
        public bool UsePTXOptimizations { get; set; } = true;

        /// <summary>
        /// Gets or sets the cooperative groups usage strategy.
        /// </summary>
        public CooperativeGroupsStrategy CooperativeGroups { get; set; } = CooperativeGroupsStrategy.Auto;
    }

    /// <summary>
    /// Specifies AMD-specific optimizations for kernels.
    /// </summary>
    public sealed class AMDOptimizationAttribute : PlatformOptimizationAttribute
    {
        /// <inheritdoc/>
        public override string TargetPlatform => "AMD";

        /// <summary>
        /// Gets or sets whether to use ROCm BLAS operations.
        /// </summary>
        public bool UseROCmBLAS { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to use AMD Matrix Instruction (MFMA).
        /// </summary>
        public bool UseMFMA { get; set; }

        /// <summary>
        /// Gets or sets the minimum GFX architecture version.
        /// </summary>
        public string MinGFXVersion { get; set; } = "gfx900";

        /// <summary>
        /// Gets or sets whether to use wavefront size optimizations.
        /// </summary>
        public bool OptimizeWavefrontSize { get; set; } = true;
    }

    /// <summary>
    /// Specifies the cooperative groups usage strategy for NVIDIA kernels.
    /// </summary>
    public enum CooperativeGroupsStrategy
    {
        /// <summary>
        /// Automatically determine when to use cooperative groups.
        /// </summary>
        Auto,

        /// <summary>
        /// Always use cooperative groups when possible.
        /// </summary>
        Always,

        /// <summary>
        /// Never use cooperative groups.
        /// </summary>
        Never,

        /// <summary>
        /// Use cooperative groups only for large problems.
        /// </summary>
        LargeProblemsOnly
    }
}