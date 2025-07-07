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

using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using System;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// Represents a compiled kernel in SYCL/SPIR-V form for Intel OneAPI devices.
    /// </summary>
    public sealed class SYCLCompiledKernel : CompiledKernel
    {
        #region Instance

        /// <summary>
        /// Constructs a new compiled kernel in SYCL/SPIR-V form.
        /// </summary>
        /// <param name="context">The associated context.</param>
        /// <param name="entryPoint">The entry point.</param>
        /// <param name="info">Detailed kernel information.</param>
        /// <param name="spirvBinary">The SPIR-V binary code.</param>
        /// <param name="syclSource">The SYCL/DPC++ source code.</param>
        internal SYCLCompiledKernel(
            Context context,
            EntryPoint entryPoint,
            KernelInfo? info,
            byte[] spirvBinary,
            string? syclSource = null)
            : base(context, entryPoint, info)
        {
            SPIRVBinary = spirvBinary ?? throw new ArgumentNullException(nameof(spirvBinary));
            SYCLSource = syclSource ?? string.Empty;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the SPIR-V binary code.
        /// </summary>
        /// <remarks>
        /// The SPIR-V binary is the compiled intermediate representation that can be
        /// executed on Intel OneAPI/SYCL-compatible devices. This includes Intel GPUs,
        /// CPUs, and potentially other accelerators that support SPIR-V.
        /// </remarks>
        public byte[] SPIRVBinary { get; }

        /// <summary>
        /// Returns the SYCL/DPC++ source code.
        /// </summary>
        /// <remarks>
        /// The SYCL source code is the high-level representation that can be compiled
        /// by the Intel DPC++ compiler. This is useful for debugging, optimization,
        /// and runtime compilation scenarios.
        /// </remarks>
        public string SYCLSource { get; }

        /// <summary>
        /// Gets whether this kernel has SYCL source code available.
        /// </summary>
        public bool HasSYCLSource => !string.IsNullOrEmpty(SYCLSource);

        /// <summary>
        /// Gets the size of the SPIR-V binary in bytes.
        /// </summary>
        public int BinarySize => SPIRVBinary.Length;

        #endregion

        #region Methods

        /// <summary>
        /// Gets a hash code for this compiled kernel based on its binary content.
        /// </summary>
        /// <returns>A hash code representing the kernel's binary content.</returns>
        public override int GetHashCode()
        {
            // Create a hash based on the SPIR-V binary content and entry point
            var hash = new HashCode();
            hash.Add(Name);
            hash.Add(IndexType);
            hash.Add(Specialization);
            
            // Include a subset of the binary in the hash for performance
            var binaryHash = SPIRVBinary.Length > 0 ? SPIRVBinary[0] : 0;
            if (SPIRVBinary.Length > 4)
            {
                binaryHash ^= SPIRVBinary[SPIRVBinary.Length / 2];
                binaryHash ^= SPIRVBinary[SPIRVBinary.Length - 1];
            }
            hash.Add(binaryHash);
            
            return hash.ToHashCode();
        }

        /// <summary>
        /// Determines whether this compiled kernel is equal to another object.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the objects are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            if (obj is not SYCLCompiledKernel other)
                return false;

            // Compare basic properties first
            if (!string.Equals(Name, other.Name, StringComparison.Ordinal) ||
                IndexType != other.IndexType ||
                !Specialization.Equals(other.Specialization))
            {
                return false;
            }

            // Compare binary content
            if (SPIRVBinary.Length != other.SPIRVBinary.Length)
                return false;

            for (int i = 0; i < SPIRVBinary.Length; i++)
            {
                if (SPIRVBinary[i] != other.SPIRVBinary[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns a string representation of this compiled kernel.
        /// </summary>
        /// <returns>A string describing the kernel.</returns>
        public override string ToString()
        {
            var sourceInfo = HasSYCLSource ? " [with SYCL source]" : " [SPIR-V only]";
            return $"SYCL Kernel: {Name} ({BinarySize} bytes){sourceInfo}";
        }

        #endregion
    }
}