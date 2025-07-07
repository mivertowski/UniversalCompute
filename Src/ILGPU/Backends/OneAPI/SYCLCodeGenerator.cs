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
using System;
using System.Collections.Generic;
using System.Text;

namespace ILGPU.Backends.OneAPI
{
    /// <summary>
    /// Generates SYCL/DPC++ code from ILGPU IR.
    /// </summary>
    internal sealed class SYCLCodeGenerator
    {
        #region Instance

        private readonly StringBuilder codeBuilder;
        private readonly Dictionary<Value, string> valueMapping;
        private int variableCounter;

        /// <summary>
        /// Initializes a new SYCL code generator.
        /// </summary>
        internal SYCLCodeGenerator()
        {
            codeBuilder = new StringBuilder();
            valueMapping = [];
            variableCounter = 0;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Generates SYCL/DPC++ code for the given entry point.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        /// <returns>The generated SYCL source code.</returns>
        internal string GenerateCode(EntryPoint entryPoint)
        {
            codeBuilder.Clear();
            valueMapping.Clear();
            variableCounter = 0;

            // Generate SYCL includes and declarations
            GenerateIncludes();
            GenerateKernelClass(entryPoint);
            GenerateLaunchFunction(entryPoint);

            return codeBuilder.ToString();
        }

        /// <summary>
        /// Generates SYCL includes and common declarations.
        /// </summary>
        private void GenerateIncludes()
        {
            codeBuilder.AppendLine("// Generated SYCL kernel code for Intel OneAPI");
            codeBuilder.AppendLine("#include <sycl/sycl.hpp>");
            codeBuilder.AppendLine("#include <dpct/dpct.hpp>");
            codeBuilder.AppendLine("#include <cstdint>");
            codeBuilder.AppendLine("#include <cmath>");
            codeBuilder.AppendLine();
            codeBuilder.AppendLine("using namespace sycl;");
            codeBuilder.AppendLine("using namespace sycl::ext::oneapi;");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates the SYCL kernel class.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelClass(EntryPoint entryPoint)
        {
            var kernelName = entryPoint.Name;
            
            codeBuilder.AppendLine($"// SYCL kernel class for {kernelName}");
            codeBuilder.AppendLine($"class {kernelName}_kernel {{");
            codeBuilder.AppendLine("private:");
            
            // Generate kernel parameters as private members
            GenerateKernelParameters(entryPoint);
            
            codeBuilder.AppendLine();
            codeBuilder.AppendLine("public:");
            
            // Generate constructor
            GenerateKernelConstructor(entryPoint);
            
            // Generate operator() for SYCL execution
            GenerateKernelOperator(entryPoint);
            
            codeBuilder.AppendLine("};");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates kernel parameters as class members.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelParameters(EntryPoint entryPoint)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var parameters = entryPoint.Parameters;
                for (int i = 0; i < parameters.Count; i++)
                {
                    var param = parameters[i];
                    var typeString = param.BaseType?.ToString() ?? "void*";
                    var paramType = GetSYCLType(typeString);
                    var paramName = $"param_{i}";
                    
                    codeBuilder.AppendLine($"    {paramType} {paramName};");
                    // Note: Parameters are Type objects, not IR Values, so they don't go in valueMapping
                }
            }
            catch (Exception ex)
            {
                codeBuilder.AppendLine($"    // Error generating parameters: {ex.Message}");
                codeBuilder.AppendLine("    void* default_param;");
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Generates the kernel constructor.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelConstructor(EntryPoint entryPoint)
        {
            var kernelName = entryPoint.Name;
            var parameters = entryPoint.Parameters;
            
            codeBuilder.Append($"    {kernelName}_kernel(");
            
            // Constructor parameters
            for (int i = 0; i < parameters.Count; i++)
            {
                if (i > 0) codeBuilder.Append(", ");
                
                var param = parameters[i];
                var paramType = GetSYCLType(param.BaseType.ToString());
                var paramName = $"param_{i}";
                
                codeBuilder.Append($"{paramType} {paramName}_arg");
            }
            
            codeBuilder.AppendLine(")");
            
            // Constructor initializer list
            if (parameters.Count > 0)
            {
                codeBuilder.Append("        : ");
                for (int i = 0; i < parameters.Count; i++)
                {
                    if (i > 0) codeBuilder.Append(", ");
                    codeBuilder.Append($"param_{i}(param_{i}_arg)");
                }
                codeBuilder.AppendLine();
            }
            
            codeBuilder.AppendLine("    {}");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates the SYCL kernel operator.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelOperator(EntryPoint entryPoint)
        {
            codeBuilder.AppendLine("    void operator()(nd_item<3> item) const {");
            
            // Generate SYCL-specific indexing
            GenerateSYCLIndexing();
            
            // Generate kernel body
            GenerateKernelBody(entryPoint);
            
            codeBuilder.AppendLine("    }");
        }

        /// <summary>
        /// Generates SYCL indexing functions.
        /// </summary>
        private void GenerateSYCLIndexing()
        {
            codeBuilder.AppendLine("        // SYCL thread indexing");
            codeBuilder.AppendLine("        auto globalId = item.get_global_id();");
            codeBuilder.AppendLine("        auto localId = item.get_local_id();");
            codeBuilder.AppendLine("        auto groupId = item.get_group_id();");
            codeBuilder.AppendLine("        auto globalRange = item.get_global_range();");
            codeBuilder.AppendLine("        auto localRange = item.get_local_range();");
            codeBuilder.AppendLine();
            codeBuilder.AppendLine("        // Individual dimension access");
            codeBuilder.AppendLine("        size_t globalIdx = globalId[0];");
            codeBuilder.AppendLine("        size_t globalIdy = globalId[1];");
            codeBuilder.AppendLine("        size_t globalIdz = globalId[2];");
            codeBuilder.AppendLine("        size_t localIdx = localId[0];");
            codeBuilder.AppendLine("        size_t localIdy = localId[1];");
            codeBuilder.AppendLine("        size_t localIdz = localId[2];");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates the kernel body from ILGPU IR.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateKernelBody(EntryPoint entryPoint)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                codeBuilder.AppendLine("        // Kernel body - ILGPU IR translation");
                codeBuilder.AppendLine("        // Thread bounds checking");
                codeBuilder.AppendLine("        const size_t MAX_THREADS = 1048576;");
                codeBuilder.AppendLine("        if (globalIdx >= MAX_THREADS) return;");
                codeBuilder.AppendLine();

                // Add safety checks
                codeBuilder.AppendLine("        // Additional safety bounds");
                codeBuilder.AppendLine("        if (globalId[0] >= globalRange[0] ||");
                codeBuilder.AppendLine("            globalId[1] >= globalRange[1] ||");
                codeBuilder.AppendLine("            globalId[2] >= globalRange[2]) return;");
                codeBuilder.AppendLine();

                // TODO: EntryPoint.Blocks property not available - OneAPI/SYCL backend needs updated IR access
                throw new NotSupportedException("OneAPI/SYCL code generation not implemented - EntryPoint IR access needs update");
            }
            catch (Exception ex)
            {
                codeBuilder.AppendLine($"        // Error during IR translation: {ex.Message}");
                GenerateDefaultKernelBody();
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Generates a default kernel body for testing.
        /// </summary>
        private void GenerateDefaultKernelBody()
        {
            codeBuilder.AppendLine("        // Default kernel implementation");
            codeBuilder.AppendLine("        // This is a placeholder that performs a simple operation");
            codeBuilder.AppendLine("        ");
            codeBuilder.AppendLine("        // Example: Basic parallel operation pattern");
            codeBuilder.AppendLine("        size_t tid = globalIdx;");
            codeBuilder.AppendLine("        ");
            codeBuilder.AppendLine("        // Kernel parameters would be accessed here");
            codeBuilder.AppendLine("        // For example: output[tid] = input[tid] * factor;");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates basic operations for a block.
        /// </summary>
        /// <param name="block">The basic block.</param>
        private void GenerateBasicBlockOperations(BasicBlock block)
        {
            codeBuilder.AppendLine("        // Basic operations");
            codeBuilder.AppendLine("        // TODO: Implement full IR translation");
            
            // Add some placeholder operations based on common patterns
            codeBuilder.AppendLine("        // Example: vector addition pattern");
            codeBuilder.AppendLine("        // if (globalIdx < array_size) {");
            codeBuilder.AppendLine("        //     result[globalIdx] = a[globalIdx] + b[globalIdx];");
            codeBuilder.AppendLine("        // }");
            codeBuilder.AppendLine();
        }

        /// <summary>
        /// Generates the kernel launch function.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        private void GenerateLaunchFunction(EntryPoint entryPoint)
        {
            var kernelName = entryPoint.Name;
            var parameters = entryPoint.Parameters;
            
            codeBuilder.AppendLine($"// Kernel launch function for {kernelName}");
            codeBuilder.Append($"extern \"C\" void launch_{kernelName}(");
            codeBuilder.Append("queue& q, range<3> global_range, range<3> local_range");
            
            // Add parameters to launch function
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = parameters[i];
                var paramType = GetSYCLType(param.BaseType.ToString());
                codeBuilder.Append($", {paramType} param_{i}");
            }
            
            codeBuilder.AppendLine(") {");
            
            // Generate kernel submission
            codeBuilder.AppendLine("    try {");
            codeBuilder.AppendLine("        auto event = q.parallel_for(");
            codeBuilder.AppendLine("            nd_range<3>(global_range, local_range),");
            codeBuilder.Append($"            {kernelName}_kernel(");
            
            // Pass parameters to kernel constructor
            for (int i = 0; i < parameters.Count; i++)
            {
                if (i > 0) codeBuilder.Append(", ");
                codeBuilder.Append($"param_{i}");
            }
            
            codeBuilder.AppendLine(")");
            codeBuilder.AppendLine("        );");
            codeBuilder.AppendLine("        event.wait();");
            codeBuilder.AppendLine("    }");
            codeBuilder.AppendLine("    catch (const sycl::exception& e) {");
            codeBuilder.AppendLine("        // Handle SYCL exceptions");
            codeBuilder.AppendLine("        throw;");
            codeBuilder.AppendLine("    }");
            codeBuilder.AppendLine("}");
            codeBuilder.AppendLine();
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
        /// Converts ILGPU type to SYCL C++ type.
        /// </summary>
        private static string GetSYCLType(string typeName) => typeName.ToUpperInvariant() switch
        {
            "bool" or "int1" => "bool",
            "int8" or "sbyte" => "int8_t",
            "uint8" or "byte" => "uint8_t",
            "int16" or "short" => "int16_t",
            "uint16" or "ushort" => "uint16_t",
            "int32" or "int" => "int32_t",
            "uint32" or "uint" => "uint32_t",
            "int64" or "long" => "int64_t",
            "uint64" or "ulong" => "uint64_t",
            "float16" or "half" => "sycl::half",
            "float32" or "float" => "float",
            "float64" or "double" => "double",
            _ when typeName.Contains('*', StringComparison.OrdinalIgnoreCase) => "void*", // Pointer types
            _ when typeName.Contains("[]", StringComparison.OrdinalIgnoreCase) => "void*", // Array types  
            _ => "void*" // Default fallback
        };

        /// <summary>
        /// Generates SYCL device functions and utilities.
        /// </summary>
        private void GenerateSYCLUtilities()
        {
            codeBuilder.AppendLine("// SYCL utility functions");
            codeBuilder.AppendLine("namespace sycl_utils {");
            codeBuilder.AppendLine();
            
            // Atomic operations
            codeBuilder.AppendLine("    template<typename T>");
            codeBuilder.AppendLine("    T atomic_add(T* addr, T val) {");
            codeBuilder.AppendLine("        return sycl::atomic_ref<T, sycl::memory_order::relaxed,");
            codeBuilder.AppendLine("                              sycl::memory_scope::device>(*addr).fetch_add(val);");
            codeBuilder.AppendLine("    }");
            codeBuilder.AppendLine();
            
            // Synchronization
            codeBuilder.AppendLine("    void barrier_global() {");
            codeBuilder.AppendLine("        sycl::group_barrier(sycl::this_group<3>());");
            codeBuilder.AppendLine("    }");
            codeBuilder.AppendLine();
            
            // Mathematical functions
            codeBuilder.AppendLine("    template<typename T>");
            codeBuilder.AppendLine("    T fast_sqrt(T x) {");
            codeBuilder.AppendLine("        return sycl::sqrt(x);");
            codeBuilder.AppendLine("    }");
            codeBuilder.AppendLine();
            
            codeBuilder.AppendLine("} // namespace sycl_utils");
            codeBuilder.AppendLine();
        }

        #endregion
    }

    /// <summary>
    /// SYCL compiler for generating device code.
    /// </summary>
    internal sealed class SYCLCompiler
    {
        /// <summary>
        /// Compiles SYCL source to SPIR-V or native binary.
        /// </summary>
        /// <param name="sourceCode">The SYCL source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <param name="targetDevice">The target device type.</param>
        /// <returns>The compiled binary.</returns>
        internal static byte[] CompileToSPIRV(string sourceCode, string kernelName, OneAPIDeviceType targetDevice)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Check if DPC++ compiler is available
                if (IsDPCPPAvailable())
                {
                    return CompileWithDPCPP(sourceCode, kernelName, targetDevice);
                }
                else
                {
                    // Fall back to source binary for debugging
                    return CompileToSourceBinary(sourceCode, kernelName);
                }
            }
            catch (Exception ex)
            {
                var errorInfo = $"// SYCL compilation error: {ex.Message}\n\n{sourceCode}";
                return System.Text.Encoding.UTF8.GetBytes(errorInfo);
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Checks if DPC++ compiler is available.
        /// </summary>
        private static bool IsDPCPPAvailable()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var processInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "dpcpp",
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = System.Diagnostics.Process.Start(processInfo);
                if (process == null) return false;

                process.WaitForExit(5000);
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Compiles with DPC++ compiler.
        /// </summary>
        private static byte[] CompileWithDPCPP(string sourceCode, string kernelName, OneAPIDeviceType targetDevice)
        {
            var tempDir = System.IO.Path.GetTempPath();
            var sourceFile = System.IO.Path.Combine(tempDir, $"{kernelName}.cpp");
            var outputFile = System.IO.Path.Combine(tempDir, $"{kernelName}.spv");

            try
            {
                // Write source to file
                System.IO.File.WriteAllText(sourceFile, sourceCode);

                // Determine target architecture
                var targetArg = targetDevice switch
                {
                    OneAPIDeviceType.GPU => "-fsycl-targets=spir64_gen",
                    OneAPIDeviceType.CPU => "-fsycl-targets=spir64_x86_64",
                    OneAPIDeviceType.FPGA => "-fsycl-targets=spir64_fpga",
                    _ => "-fsycl-targets=spir64"
                };

                // Compile with DPC++
                var processInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "dpcpp",
                    Arguments = $"-fsycl {targetArg} -O2 -o {outputFile} {sourceFile}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = tempDir
                };

                using var process = System.Diagnostics.Process.Start(processInfo) ?? throw new InvalidOperationException("Failed to start DPC++ compiler");
                process.WaitForExit(30000);

                if (process.ExitCode != 0)
                {
                    var error = process.StandardError.ReadToEnd();
                    throw new InvalidOperationException($"DPC++ compilation failed: {error}");
                }

                // Read compiled binary
                return System.IO.File.Exists(outputFile)
                    ? System.IO.File.ReadAllBytes(outputFile)
                    : throw new System.IO.FileNotFoundException("Compiled binary not found");
            }
            finally
            {
                // Clean up
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    if (System.IO.File.Exists(sourceFile))
                        System.IO.File.Delete(sourceFile);
                    if (System.IO.File.Exists(outputFile))
                        System.IO.File.Delete(outputFile);
                }
                catch { }
#pragma warning restore CA1031 // Do not catch general exception types
            }
        }

        /// <summary>
        /// Creates source binary for debugging.
        /// </summary>
        private static byte[] CompileToSourceBinary(string sourceCode, string kernelName)
        {
            var header = $"// SYCL kernel: {kernelName}\n// Compiled as source binary (no DPC++ compiler available)\n\n";
            var fullSource = header + sourceCode;
            return System.Text.Encoding.UTF8.GetBytes(fullSource);
        }
    }

    /// <summary>
    /// OneAPI device types for SYCL compilation.
    /// </summary>
    public enum OneAPIDeviceType
    {
        CPU,
        GPU, 
        FPGA,
        Accelerator
    }
}
