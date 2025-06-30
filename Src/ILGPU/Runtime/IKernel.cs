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
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Represents a compiled kernel that can be executed on an accelerator.
    /// </summary>
    public interface IKernel : IDisposable
    {
        /// <summary>
        /// Gets the name of the kernel.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the accelerator this kernel was compiled for.
        /// </summary>
        Accelerator Accelerator { get; }

        /// <summary>
        /// Gets information about the kernel's resource requirements.
        /// </summary>
        KernelInfo Info { get; }

        /// <summary>
        /// Executes the kernel with the specified parameters.
        /// </summary>
        /// <param name="groupSize">The group size for kernel execution.</param>
        /// <param name="gridSize">The grid size for kernel execution.</param>
        /// <param name="parameters">The kernel parameters.</param>
        void Execute(int groupSize, int gridSize, params object[] parameters);

        /// <summary>
        /// Executes the kernel asynchronously with the specified parameters.
        /// </summary>
        /// <param name="groupSize">The group size for kernel execution.</param>
        /// <param name="gridSize">The grid size for kernel execution.</param>
        /// <param name="parameters">The kernel parameters.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task representing the asynchronous execution.</returns>
        Task ExecuteAsync(int groupSize, int gridSize, object[] parameters, 
            CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Represents kernel source code for compilation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the KernelSource struct.
    /// </remarks>
    /// <param name="source">The kernel source code.</param>
    /// <param name="entryPoint">The entry point function name.</param>
    /// <param name="language">The source language.</param>
    public readonly struct KernelSource(string source, string entryPoint, KernelLanguage language)
    {

        /// <summary>
        /// Gets the kernel source code.
        /// </summary>
        public string Source { get; } = source ?? throw new ArgumentNullException(nameof(source));

        /// <summary>
        /// Gets the entry point function name.
        /// </summary>
        public string EntryPoint { get; } = entryPoint ?? throw new ArgumentNullException(nameof(entryPoint));

        /// <summary>
        /// Gets the source language.
        /// </summary>
        public KernelLanguage Language { get; } = language;
    }

    /// <summary>
    /// Represents the source language for kernels.
    /// </summary>
    public enum KernelLanguage
    {
        /// <summary>
        /// ILGPU kernel language.
        /// </summary>
        ILGPU,

        /// <summary>
        /// CUDA C++ kernel language.
        /// </summary>
        CUDA,

        /// <summary>
        /// OpenCL C kernel language.
        /// </summary>
        OpenCL,

        /// <summary>
        /// Metal Shading Language.
        /// </summary>
        Metal,

        /// <summary>
        /// HLSL compute shader language.
        /// </summary>
        HLSL,

        /// <summary>
        /// Vulkan SPIR-V.
        /// </summary>
        SPIRV,

        /// <summary>
        /// Intel OneAPI DPC++.
        /// </summary>
        DPCpp
    }

    /// <summary>
    /// Compilation options for kernels.
    /// </summary>
    public sealed class CompilationOptions
    {
        /// <summary>
        /// Gets or sets whether to enable debug information.
        /// </summary>
        public bool DebugMode { get; set; }

        /// <summary>
        /// Gets or sets the optimization level.
        /// </summary>
        public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.O2;

        /// <summary>
        /// Gets or sets whether to enable fast math optimizations.
        /// </summary>
        public bool FastMath { get; set; }

        /// <summary>
        /// Gets or sets additional compiler flags.
        /// </summary>
        public IReadOnlyList<string>? AdditionalFlags { get; set; }

        /// <summary>
        /// Gets or sets the target architecture.
        /// </summary>
        public string? TargetArchitecture { get; set; }
    }

    /// <summary>
    /// Optimization levels for kernel compilation.
    /// </summary>
    public enum OptimizationLevel
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