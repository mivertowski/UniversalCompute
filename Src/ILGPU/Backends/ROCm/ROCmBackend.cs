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

using ILGPU.Backends.EntryPoints;
using ILGPU.IR;
using ILGPU.Runtime;
using ILGPU.Runtime.ROCm;
using ILGPU.Util;
using System;

namespace ILGPU.Backends.ROCm
{
    /// <summary>
    /// ROCm backend for code generation.
    /// </summary>
    public sealed class ROCmBackend : Backend
    {
        #region Instance

        /// <summary>
        /// The ROCm instruction set.
        /// </summary>
        public ROCmInstructionSet InstructionSet { get; }

        /// <summary>
        /// The ROCm capabilities.
        /// </summary>
        public ROCmCapabilities Capabilities { get; }

        /// <summary>
        /// Initializes a new ROCm backend.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="instructionSet">The instruction set.</param>
        /// <param name="capabilities">The capabilities.</param>
        public ROCmBackend(
            Context context,
            ROCmInstructionSet instructionSet,
            ROCmCapabilities capabilities)
            : base(
                context,
                BackendType.OpenCL, // Use OpenCL as base for now
                BackendFlags.None,
                new ArgumentMapper(context))
        {
            InstructionSet = instructionSet;
            Capabilities = capabilities;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Compiles the given entry point into a ROCm kernel.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        /// <param name="backendContext">The backend context.</param>
        /// <param name="specialization">The kernel specialization.</param>
        /// <returns>The compiled kernel.</returns>
        protected override CompiledKernel Compile(
            EntryPoint entryPoint,
            in BackendContext backendContext,
            in KernelSpecialization specialization)
        {
            // For now, create a placeholder compiled kernel
            // In a real implementation, this would:
            // 1. Generate HIP/ROCm code from the IR
            // 2. Compile it using the ROCm compiler toolchain
            // 3. Return the compiled binary

            var kernelInfo = new KernelInfo(
                localMemorySize: 0,
                sharedMemorySize: 0,
                allocaKindInformation: new AllocaKindInformation(),
                functions: System.Collections.Immutable.ImmutableArray<CompiledKernel.FunctionInfo>.Empty);

            // Generate a placeholder binary (in real implementation, this would be actual HIP code)
            var placeholderBinary = new byte[] { 0x48, 0x49, 0x50, 0x00 }; // "HIP\0"

            return new Runtime.ROCm.ROCmCompiledKernel(
                Context,
                entryPoint,
                kernelInfo,
                placeholderBinary);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the backend type.
        /// </summary>
        public string Name => "ROCm";

        #endregion
    }
}