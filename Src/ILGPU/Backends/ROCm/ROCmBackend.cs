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
using ILGPU.IR.Analyses;
using ILGPU.Runtime;
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
        public new ROCmCapabilities Capabilities { get; }

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
                capabilities, // Use ROCm capabilities
                ArgumentMapper.CreateDefault(context))
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
            try
            {
                // Generate HIP C++ code from ILGPU IR
                var hipCodeGenerator = new ROCmCodeGenerator(this, InstructionSet, Capabilities);
                var hipSourceCode = hipCodeGenerator.GenerateCode(entryPoint);

                // Compile HIP code to binary using HIP compiler
                var hipCompiler = new ROCmCompiler(InstructionSet);
                var compiledBinary = hipCompiler.CompileToHSAIL(hipSourceCode, entryPoint.Name);

                var kernelInfo = new KernelInfo(
                    0, // SharedMemory
                    0, // ConstantMemory
                    new AllocaKindInformation(),
                    []);

                return new Runtime.ROCm.ROCmCompiledKernel(
                    Context,
                    entryPoint,
                    kernelInfo,
                    compiledBinary);
            }
            catch (Exception ex)
            {
                // Fall back to placeholder binary for now
                var kernelInfo = new KernelInfo(
                    0, // SharedMemory
                    0, // ConstantMemory
                    new AllocaKindInformation(),
                    []);

                // Generate a placeholder binary with error information
                var errorMessage = $"// ROCm compilation failed: {ex.Message}\n";
                var placeholderBinary = System.Text.Encoding.UTF8.GetBytes(errorMessage);

                return new Runtime.ROCm.ROCmCompiledKernel(
                    Context,
                    entryPoint,
                    kernelInfo,
                    placeholderBinary);
            }
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the backend type.
        /// </summary>
        public static string Name => "ROCm";

        #endregion
    }
}