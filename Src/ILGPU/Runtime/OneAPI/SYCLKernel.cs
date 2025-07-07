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

using ILGPU.Runtime.OneAPI.Native;
using ILGPU.Util;
using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// Represents a SYCL kernel that can be directly launched on Intel OneAPI devices.
    /// </summary>
    public sealed class SYCLKernel : Kernel
    {
        #region Instance

        /// <summary>
        /// Holds the pointer to the native SYCL kernel in memory.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private IntPtr kernelPtr;

        /// <summary>
        /// Tracks whether the kernel was successfully created using SYCL.
        /// </summary>
        private readonly bool isNativeKernel;

        /// <summary>
        /// Gets the associated Intel OneAPI accelerator.
        /// </summary>
        public new IntelOneAPIAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelOneAPIAccelerator>();

        /// <summary>
        /// Loads a compiled kernel into the given SYCL context as kernel program.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        /// <param name="kernel">The source kernel.</param>
        /// <param name="launcher">The launcher method for the given kernel.</param>
        internal SYCLKernel(
            IntelOneAPIAccelerator accelerator,
            SYCLCompiledKernel kernel,
            MethodInfo? launcher = null)
            : base(accelerator, kernel, launcher)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Create SYCL kernel from SPIR-V binary
                var result = SYCLNative.CreateKernelFromSPIRV(
                    accelerator.ContextHandle,
                    [accelerator.DeviceHandle],
                    1,
                    kernel.SPIRVBinary,
                    new UIntPtr((uint)kernel.BinarySize),
                    kernel.Name,
                    out kernelPtr);

                if (result != SYCLResult.Success || kernelPtr == IntPtr.Zero)
                {
                    Trace.WriteLine($"SYCL Kernel loading failed for '{kernel.Name}': {result}");
                    
                    // Try alternative kernel creation approaches
                    if (TryCreateKernelFromSource(accelerator, kernel))
                    {
                        isNativeKernel = true;
                    }
                    else
                    {
                        // Create a placeholder kernel for compatibility
                        kernelPtr = CreatePlaceholderKernel();
                        isNativeKernel = false;
                        Trace.WriteLine($"Created placeholder kernel for '{kernel.Name}'");
                    }
                }
                else
                {
                    isNativeKernel = true;
                }
            }
            catch (DllNotFoundException)
            {
                // SYCL runtime not available - create placeholder
                kernelPtr = CreatePlaceholderKernel();
                isNativeKernel = false;
                Trace.WriteLine($"SYCL runtime not available, created placeholder kernel for '{kernel.Name}'");
            }
            catch (EntryPointNotFoundException)
            {
                // SYCL functions not available - create placeholder
                kernelPtr = CreatePlaceholderKernel();
                isNativeKernel = false;
                Trace.WriteLine($"SYCL functions not available, created placeholder kernel for '{kernel.Name}'");
            }
            catch (Exception ex)
            {
                // Any other error - create placeholder
                kernelPtr = CreatePlaceholderKernel();
                isNativeKernel = false;
                Trace.WriteLine($"SYCL kernel creation failed for '{kernel.Name}': {ex.Message}");
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the SYCL kernel pointer.
        /// </summary>
        public IntPtr KernelPtr => kernelPtr;

        /// <summary>
        /// Gets whether this kernel uses native SYCL implementation.
        /// </summary>
        public bool IsNativeKernel => isNativeKernel;

        #endregion

        #region Methods

        /// <summary>
        /// Attempts to create a kernel from SYCL source code as a fallback.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="kernel">The compiled kernel.</param>
        /// <returns>True if kernel creation succeeded; otherwise, false.</returns>
        private bool TryCreateKernelFromSource(IntelOneAPIAccelerator accelerator, SYCLCompiledKernel kernel)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                if (!kernel.HasSYCLSource)
                    return false;

                // This would involve compiling SYCL source at runtime
                // For now, this is a placeholder that would be implemented
                // with the Intel DPC++ runtime compiler
                
                // Example of what this might look like:
                // var program = SYCLNative.CompileProgram(accelerator.ContextHandle, kernel.SYCLSource);
                // var result = SYCLNative.CreateKernelFromProgram(program, kernel.Name, out kernelPtr);
                // return result == SYCLResult.Success;

                return false;
            }
            catch (Exception)
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Creates a placeholder kernel handle for compatibility.
        /// </summary>
        /// <returns>A placeholder kernel pointer.</returns>
        private static IntPtr CreatePlaceholderKernel()
        {
            // Create a dummy handle that's not zero to indicate "valid" kernel
            // This allows the kernel object to be created without crashing
            // but actual execution would need to be handled gracefully
            return new IntPtr(0x12345678); // Recognizable placeholder value
        }

        /// <summary>
        /// Sets a kernel argument at the specified index.
        /// </summary>
        /// <param name="index">The argument index.</param>
        /// <param name="value">The argument value.</param>
        /// <param name="size">The size of the argument in bytes.</param>
        /// <returns>True if the argument was set successfully; otherwise, false.</returns>
        public bool SetArgument(uint index, IntPtr value, uint size)
        {
            if (!isNativeKernel || kernelPtr == IntPtr.Zero)
                return false;

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var result = SYCLNative.SetKernelArg(
                    kernelPtr,
                    index,
                    new UIntPtr(size),
                    value);

                return result == SYCLResult.Success;
            }
            catch (Exception)
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Sets a kernel argument from a managed object.
        /// </summary>
        /// <typeparam name="T">The argument type.</typeparam>
        /// <param name="index">The argument index.</param>
        /// <param name="value">The argument value.</param>
        /// <returns>True if the argument was set successfully; otherwise, false.</returns>
        public bool SetArgument<T>(uint index, T value) where T : unmanaged
        {
            if (!isNativeKernel)
                return false;

            unsafe
            {
                var ptr = new IntPtr(&value);
                return SetArgument(index, ptr, (uint)sizeof(T));
            }
        }

        /// <summary>
        /// Launches the kernel with the specified work dimensions.
        /// </summary>
        /// <param name="queue">The SYCL queue.</param>
        /// <param name="globalWorkSize">The global work size.</param>
        /// <param name="localWorkSize">The local work size.</param>
        /// <returns>True if the kernel was launched successfully; otherwise, false.</returns>
        public bool Launch(IntPtr queue, Index3D globalWorkSize, Index3D localWorkSize)
        {
            if (!isNativeKernel || kernelPtr == IntPtr.Zero)
                return false;

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var globalSizes = new UIntPtr[]
                {
                    new((uint)globalWorkSize.X),
                    new((uint)globalWorkSize.Y),
                    new((uint)globalWorkSize.Z)
                };

                var localSizes = new UIntPtr[]
                {
                    new((uint)localWorkSize.X),
                    new((uint)localWorkSize.Y),
                    new((uint)localWorkSize.Z)
                };

                var eventHandle = SYCLNative.SubmitKernel(
                    queue,
                    kernelPtr,
                    3, // 3D work
                    globalSizes,
                    localSizes,
                    [], // No wait events
                    0); // No wait events

                return eventHandle != IntPtr.Zero;
            }
            catch (Exception)
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Disposes this SYCL kernel.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && kernelPtr != IntPtr.Zero)
            {
                if (isNativeKernel)
                {
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        // Release the SYCL kernel
                        SYCLNative.ReleaseKernel(kernelPtr);
                    }
                    catch (Exception)
                    {
                        // Ignore errors during disposal
                    }
#pragma warning restore CA1031 // Do not catch general exception types
                }

                kernelPtr = IntPtr.Zero;
            }
        }

        #endregion
    }
}