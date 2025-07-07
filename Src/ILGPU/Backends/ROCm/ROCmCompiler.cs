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
using System.Diagnostics;
using System.IO;
using System.Text;

namespace ILGPU.Backends.ROCm
{
    /// <summary>
    /// Compiles HIP C++ source code to ROCm binary format.
    /// </summary>
    internal sealed class ROCmCompiler
    {
        #region Instance

        private readonly ROCmInstructionSet instructionSet;

        /// <summary>
        /// Initializes a new ROCm compiler.
        /// </summary>
        /// <param name="instructionSet">The instruction set.</param>
        internal ROCmCompiler(ROCmInstructionSet instructionSet)
        {
            this.instructionSet = instructionSet;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Compiles HIP source code to HSAIL/GCN binary.
        /// </summary>
        /// <param name="sourceCode">The HIP C++ source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <returns>The compiled binary data.</returns>
        internal byte[] CompileToHSAIL(string sourceCode, string kernelName)
        {
            try
            {
                // Try to use system HIP compiler if available
                if (IsHipCompilerAvailable())
                {
                    return CompileWithHipcc(sourceCode, kernelName);
                }
                else
                {
                    // Fall back to storing source code as binary for debugging
                    return CompileToSourceBinary(sourceCode, kernelName);
                }
            }
            catch (Exception ex)
            {
                // Create error binary containing source code and error information
                var errorInfo = $"// ROCm compilation error: {ex.Message}\n\n{sourceCode}";
                return Encoding.UTF8.GetBytes(errorInfo);
            }
        }

        /// <summary>
        /// Checks if HIP compiler (hipcc) is available on the system.
        /// </summary>
        /// <returns>True if HIP compiler is available; otherwise, false.</returns>
        private static bool IsHipCompilerAvailable()
        {
            try
            {
                var processInfo = new ProcessStartInfo
                {
                    FileName = "hipcc",
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process == null)
                    return false;

                process.WaitForExit(5000); // 5 second timeout
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Compiles HIP source using hipcc compiler.
        /// </summary>
        /// <param name="sourceCode">The source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <returns>The compiled binary.</returns>
        private byte[] CompileWithHipcc(string sourceCode, string kernelName)
        {
            var tempDir = Path.GetTempPath();
            var sourceFile = Path.Combine(tempDir, $"{kernelName}.hip");
            var objectFile = Path.Combine(tempDir, $"{kernelName}.co");

            try
            {
                // Write source to temporary file
                File.WriteAllText(sourceFile, sourceCode);

                // Determine target architecture
                var architecture = DetermineTargetArchitecture();

                // Compile with hipcc
                var processInfo = new ProcessStartInfo
                {
                    FileName = "hipcc",
                    Arguments = $"-c -fPIC --amdgpu-target={architecture} -o \"{objectFile}\" \"{sourceFile}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = tempDir
                };

                using var process = Process.Start(processInfo) ?? throw new InvalidOperationException("Failed to start hipcc compiler");
                process.WaitForExit(30000); // 30 second timeout

                if (process.ExitCode != 0)
                {
                    var error = process.StandardError.ReadToEnd();
                    throw new InvalidOperationException($"hipcc compilation failed: {error}");
                }

                // Read compiled binary
                return File.Exists(objectFile) ? File.ReadAllBytes(objectFile) : throw new FileNotFoundException("Compiled object file not found");
            }
            finally
            {
                // Clean up temporary files
                try
                {
                    if (File.Exists(sourceFile))
                        File.Delete(sourceFile);
                    if (File.Exists(objectFile))
                        File.Delete(objectFile);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        /// <summary>
        /// Determines the target ROCm architecture.
        /// </summary>
        /// <returns>The target architecture string.</returns>
        private string DetermineTargetArchitecture() =>
            // Map instruction set to ROCm architecture
            instructionSet.Architecture switch
            {
                ROCmArchitecture.GCN3 => "gfx803", // Fiji, Polaris
                ROCmArchitecture.GCN4 => "gfx900", // Vega
                ROCmArchitecture.GCN5 => "gfx906", // Vega 7nm
                ROCmArchitecture.RDNA1 => "gfx1010", // Navi 10
                ROCmArchitecture.RDNA2 => "gfx1030", // Navi 21
                ROCmArchitecture.RDNA3 => "gfx1100", // Navi 31
                ROCmArchitecture.CDNA1 => "gfx908", // MI100
                ROCmArchitecture.CDNA2 => "gfx90a", // MI200
                ROCmArchitecture.CDNA3 => "gfx942", // MI300
                _ => "gfx900" // Default to Vega
            };

        /// <summary>
        /// Creates a binary containing the source code for debugging purposes.
        /// </summary>
        /// <param name="sourceCode">The source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <returns>The source code as binary data.</returns>
        private static byte[] CompileToSourceBinary(string sourceCode, string kernelName)
        {
            var header = $"// HIP kernel: {kernelName}\n// Compiled as source binary (no HIP compiler available)\n\n";
            var fullSource = header + sourceCode;
            return Encoding.UTF8.GetBytes(fullSource);
        }

        /// <summary>
        /// Compiles source to LLVM IR (alternative compilation path).
        /// </summary>
        /// <param name="sourceCode">The source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <returns>The LLVM IR binary.</returns>
        internal static byte[] CompileToLLVMIR(string sourceCode, string kernelName)
        {
            try
            {
                return IsClangAvailable() ? CompileWithClang(sourceCode, kernelName) : CompileToSourceBinary(sourceCode, kernelName);
            }
            catch (Exception ex)
            {
                var errorInfo = $"// LLVM IR compilation error: {ex.Message}\n\n{sourceCode}";
                return Encoding.UTF8.GetBytes(errorInfo);
            }
        }

        /// <summary>
        /// Checks if Clang is available for compilation.
        /// </summary>
        /// <returns>True if Clang is available; otherwise, false.</returns>
        private static bool IsClangAvailable()
        {
            try
            {
                var processInfo = new ProcessStartInfo
                {
                    FileName = "clang",
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process == null)
                    return false;

                process.WaitForExit(5000);
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Compiles source to LLVM IR using Clang.
        /// </summary>
        /// <param name="sourceCode">The source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <returns>The LLVM IR binary.</returns>
        private static byte[] CompileWithClang(string sourceCode, string kernelName)
        {
            var tempDir = Path.GetTempPath();
            var sourceFile = Path.Combine(tempDir, $"{kernelName}.hip");
            var llvmFile = Path.Combine(tempDir, $"{kernelName}.ll");

            try
            {
                // Write source to temporary file
                File.WriteAllText(sourceFile, sourceCode);

                // Compile to LLVM IR with Clang
                var processInfo = new ProcessStartInfo
                {
                    FileName = "clang",
                    Arguments = $"-x hip -S -emit-llvm -o \"{llvmFile}\" \"{sourceFile}\" -I/opt/rocm/include",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = tempDir
                };

                using var process = Process.Start(processInfo) ?? throw new InvalidOperationException("Failed to start clang compiler");
                process.WaitForExit(30000);

                if (process.ExitCode != 0)
                {
                    var error = process.StandardError.ReadToEnd();
                    throw new InvalidOperationException($"Clang compilation failed: {error}");
                }

                // Read LLVM IR
                return File.Exists(llvmFile) ? File.ReadAllBytes(llvmFile) : throw new FileNotFoundException("LLVM IR file not found");
            }
            finally
            {
                // Clean up temporary files
                try
                {
                    if (File.Exists(sourceFile))
                        File.Delete(sourceFile);
                    if (File.Exists(llvmFile))
                        File.Delete(llvmFile);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        #endregion
    }
}