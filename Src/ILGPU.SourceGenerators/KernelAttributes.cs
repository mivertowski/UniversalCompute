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
// Change License: Apache License, Version 2.0using System;

namespace ILGPU.SourceGenerators
{
    /// <summary>
    /// Marks a static method as a GPU kernel that should have an AOT-compatible launcher generated.
    /// This attribute replaces the need for runtime IL generation.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public sealed class AOTKernelAttribute : Attribute
    {
        /// <summary>
        /// Optional name for the generated launcher method.
        /// If not specified, defaults to "Launch{MethodName}".
        /// </summary>
        public string? LauncherName { get; set; }

        /// <summary>
        /// Indicates whether this kernel should be optimized for specific GPU architectures.
        /// </summary>
        public bool ArchitectureSpecific { get; set; }

        /// <summary>
        /// Specifies the optimization level for the generated launcher.
        /// </summary>
        public OptimizationLevel Optimization { get; set; } = OptimizationLevel.Default;
    }

    /// <summary>
    /// Marks a class as containing kernel methods that should have launchers generated.
    /// This enables batch generation of launchers for all kernels in the class.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class KernelContainerAttribute : Attribute
    {
        /// <summary>
        /// Optional prefix for all generated launcher methods in this class.
        /// </summary>
        public string? LauncherPrefix { get; set; }

        /// <summary>
        /// Indicates whether to generate a single launcher class for all kernels
        /// or individual launcher methods.
        /// </summary>
        public bool GenerateSingleLauncherClass { get; set; } = true;
    }

    /// <summary>
    /// Optimization levels for AOT kernel generation.
    /// </summary>
    public enum OptimizationLevel
    {
        /// <summary>
        /// Default optimization suitable for most kernels.
        /// </summary>
        Default,

        /// <summary>
        /// Optimized for execution speed, may increase compile time.
        /// </summary>
        Speed,

        /// <summary>
        /// Optimized for memory usage and smaller binary size.
        /// </summary>
        Size,

        /// <summary>
        /// Debug-friendly version with minimal optimizations.
        /// </summary>
        Debug
    }
}
