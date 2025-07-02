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
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ILGPU.Backends.ROCm
{
    /// <summary>
    /// Generates HIP C++ code from ILGPU IR.
    /// </summary>
    internal sealed class ROCmCodeGenerator
    {
        #region Instance

        private readonly ROCmBackend backend;
        private readonly ROCmInstructionSet instructionSet;
        private readonly ROCmCapabilities capabilities;
        private readonly StringBuilder codeBuilder;
        private readonly Dictionary<Value, string> valueMapping;
        private int variableCounter;

        /// <summary>
        /// Initializes a new ROCm code generator.
        /// </summary>
        /// <param name="backend">The ROCm backend.</param>
        /// <param name="instructionSet">The instruction set.</param>
        /// <param name="capabilities">The capabilities.</param>
        internal ROCmCodeGenerator(
            ROCmBackend backend,
            ROCmInstructionSet instructionSet,
            ROCmCapabilities capabilities)
        {
            this.backend = backend;
            this.instructionSet = instructionSet;
            this.capabilities = capabilities;
            codeBuilder = new StringBuilder();
            valueMapping = new Dictionary<Value, string>();
            variableCounter = 0;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Generates HIP C++ code for the given entry point.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        /// <returns>The generated HIP C++ source code.</returns>
        internal string GenerateCode(EntryPoint entryPoint)
        {
            codeBuilder.Clear();
            valueMapping.Clear();
            variableCounter = 0;

            // Generate HIP includes and declarations
            GenerateIncludes();
            GenerateKernelSignature(entryPoint);
            GenerateKernelBody(entryPoint);

            return codeBuilder.ToString();
        }

        /// <summary>
        /// Generates HIP includes and common declarations.
        /// </summary>
        private void GenerateIncludes()
        {
            codeBuilder.AppendLine("// Generated HIP kernel code for ROCm");
            codeBuilder.AppendLine("#include <hip/hip_runtime.h>");
            codeBuilder.AppendLine("#include <hip/hip_runtime_api.h>");
            codeBuilder.AppendLine("#include <stdint.h>");
            codeBuilder.AppendLine("#include <math.h>");
            codeBuilder.AppendLine();

            // Add HIP-specific device functions
            codeBuilder.AppendLine("// HIP device functions");
            codeBuilder.AppendLine("__device__ inline int get_global_id(int dim) {");
            codeBuilder.AppendLine("  switch(dim) {");
            codeBuilder.AppendLine("    case 0: return blockIdx.x * blockDim.x + threadIdx.x;");
            codeBuilder.AppendLine("    case 1: return blockIdx.y * blockDim.y + threadIdx.y;");
            codeBuilder.AppendLine("    case 2: return blockIdx.z * blockDim.z + threadIdx.z;");
            codeBuilder.AppendLine("    default: return 0;");
            codeBuilder.AppendLine("  }");
            codeBuilder.AppendLine("}");
            codeBuilder.AppendLine();

            codeBuilder.AppendLine("__device__ inline int get_local_id(int dim) {");
            codeBuilder.AppendLine("  switch(dim) {");
            codeBuilder.AppendLine("    case 0: return threadIdx.x;");
            codeBuilder.AppendLine("    case 1: return threadIdx.y;");
            codeBuilder.AppendLine("    case 2: return threadIdx.z;");
            codeBuilder.AppendLine("    default: return 0;");
            codeBuilder.AppendLine("  }");
            codeBuilder.AppendLine("}");
            codeBuilder.AppendLine();

            codeBuilder.AppendLine("__device__ inline int get_group_id(int dim) {");
            codeBuilder.AppendLine("  switch(dim) {");
            codeBuilder.AppendLine("    case 0: return blockIdx.x;");
            codeBuilder.AppendLine("    case 1: return blockIdx.y;");
            codeBuilder.AppendLine("    case 2: return blockIdx.z;");
            codeBuilder.AppendLine("    default: return 0;");
            codeBuilder.AppendLine("  }");
            codeBuilder.AppendLine("}");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates the kernel signature.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelSignature(EntryPoint entryPoint)
        {
            codeBuilder.Append("extern \"C\" __global__ void ");
            codeBuilder.Append(entryPoint.Name);
            codeBuilder.Append("(");

            // Generate parameters
            var parameters = entryPoint.Parameters;
            for (int i = 0; i < parameters.Count; i++)
            {
                if (i > 0)
                    codeBuilder.Append(", ");

                var param = parameters[i];
                var paramType = GetHipType(param.Type);
                var paramName = $"param_{i}";
                
                codeBuilder.Append(paramType);
                codeBuilder.Append(" ");
                codeBuilder.Append(paramName);

                // Map parameter to name
                valueMapping[param] = paramName;
            }

            codeBuilder.AppendLine(")");
        }

        /// <summary>
        /// Generates the kernel body.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelBody(EntryPoint entryPoint)
        {
            codeBuilder.AppendLine("{");

            try
            {
                // Generate basic grid/block indexing
                codeBuilder.AppendLine("  // Grid and thread indexing");
                codeBuilder.AppendLine("  int globalIdx = get_global_id(0);");
                codeBuilder.AppendLine("  int globalIdy = get_global_id(1);");
                codeBuilder.AppendLine("  int globalIdz = get_global_id(2);");
                codeBuilder.AppendLine("  int localIdx = get_local_id(0);");
                codeBuilder.AppendLine("  int localIdy = get_local_id(1);");
                codeBuilder.AppendLine("  int localIdz = get_local_id(2);");
                codeBuilder.AppendLine();

                // Generate basic kernel body
                codeBuilder.AppendLine("  // Kernel body (simplified implementation)");
                codeBuilder.AppendLine("  // TODO: Implement full ILGPU IR to HIP translation");
                
                // Add bounds checking
                codeBuilder.AppendLine("  if (globalIdx >= 65536) return; // Basic bounds check");
                codeBuilder.AppendLine();

                // Try to generate some basic operations based on entry point
                GenerateBasicOperations(entryPoint);
            }
            catch (Exception ex)
            {
                // Add error as comment
                codeBuilder.AppendLine($"  // Error during code generation: {ex.Message}");
                codeBuilder.AppendLine("  // Placeholder kernel body");
            }

            codeBuilder.AppendLine("}");
        }

        /// <summary>
        /// Generates basic operations for the kernel.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateBasicOperations(EntryPoint entryPoint)
        {
            // Walk through the IR blocks and generate basic operations
            foreach (var block in entryPoint.Blocks)
            {
                codeBuilder.AppendLine($"  // Block {block.Name}");
                
                // Simplified operation generation for now
                codeBuilder.AppendLine("  // Basic operations placeholder");
                codeBuilder.AppendLine("  // TODO: Implement full IR traversal and code generation");
            }
        }

        /// <summary>
        /// Gets or creates a variable name for an IR value.
        /// </summary>
        private string GetVariableName(Value value)
        {
            if (valueMapping.TryGetValue(value, out string? existing))
                return existing;

            var name = $"var_{variableCounter++}";
            valueMapping[value] = name;
            return name;
        }

        /// <summary>
        /// Converts basic type to HIP C++ type.
        /// </summary>
        private static string GetHipType(string typeName)
        {
            return typeName switch
            {
                "bool" => "bool",
                "int8" => "int8_t",
                "int16" => "int16_t", 
                "int32" => "int32_t",
                "int64" => "int64_t",
                "float16" => "half",
                "float32" => "float",
                "float64" => "double",
                _ => "void*"
            };
        }

        #endregion
    }
}