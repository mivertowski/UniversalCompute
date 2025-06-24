// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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
    /// Marks a kernel method as universal, meaning it can be compiled and optimized
    /// for all supported ILGPU backends (CPU, CUDA, OpenCL, Metal, etc.).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public sealed class UniversalKernelAttribute : Attribute
    {
        /// <summary>
        /// Gets or sets whether to enable platform-specific optimizations.
        /// </summary>
        public bool EnableOptimizations { get; set; } = true;

        /// <summary>
        /// Gets or sets the preferred execution strategy for this kernel.
        /// </summary>
        public KernelExecutionStrategy PreferredStrategy { get; set; } = KernelExecutionStrategy.Auto;

        /// <summary>
        /// Gets or sets the minimum problem size for efficient execution.
        /// Below this size, the kernel may be scheduled on CPU.
        /// </summary>
        public long MinimumProblemSize { get; set; } = 1024;

        /// <summary>
        /// Gets or sets whether this kernel supports mixed precision execution.
        /// </summary>
        public bool SupportsMixedPrecision { get; set; } = false;

        /// <summary>
        /// Initializes a new instance of the UniversalKernelAttribute class.
        /// </summary>
        public UniversalKernelAttribute()
        {
        }
    }

    /// <summary>
    /// Specifies the execution strategy for universal kernels.
    /// </summary>
    public enum KernelExecutionStrategy
    {
        /// <summary>
        /// Automatically select the best execution strategy based on hardware and problem size.
        /// </summary>
        Auto,

        /// <summary>
        /// Prefer GPU execution for compute-intensive operations.
        /// </summary>
        PreferGpu,

        /// <summary>
        /// Prefer CPU execution for control-heavy or small problems.
        /// </summary>
        PreferCpu,

        /// <summary>
        /// Use hybrid execution across multiple accelerators.
        /// </summary>
        Hybrid,

        /// <summary>
        /// Use specialized AI/ML accelerators when available.
        /// </summary>
        PreferAI
    }
}