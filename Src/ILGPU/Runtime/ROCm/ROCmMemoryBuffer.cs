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

using ILGPU.Runtime.ROCm.Native;
using ILGPU.Util;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// A ROCm-specific memory buffer implementation.
    /// </summary>
    public sealed class ROCmMemoryBuffer : MemoryBuffer
    {
        #region Instance

        /// <summary>
        /// The native HIP memory pointer.
        /// </summary>
        private IntPtr nativePtr;

        /// <summary>
        /// True if this buffer was allocated using HIP APIs.
        /// </summary>
        private readonly bool isNativeAllocation;

        /// <summary>
        /// Gets the associated ROCm accelerator.
        /// </summary>
        public new ROCmAccelerator Accelerator => base.Accelerator.AsNotNullCast<ROCmAccelerator>();

        /// <summary>
        /// Initializes a new ROCm memory buffer.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="length">The length of this buffer.</param>
        /// <param name="elementSize">The element size.</param>
        internal ROCmMemoryBuffer(
            ROCmAccelerator accelerator,
            long length,
            int elementSize)
            : base(accelerator, length, elementSize)
        {
            try
            {
                // Try to allocate using HIP
                var result = ROCmNative.Malloc(out nativePtr, (ulong)LengthInBytes);
                ROCmException.ThrowIfFailed(result);
                isNativeAllocation = true;
            }
            catch (DllNotFoundException)
            {
                // Fall back to host memory allocation
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                isNativeAllocation = false;
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to host memory allocation
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                isNativeAllocation = false;
            }
            catch (ROCmException)
            {
                // Fall back to host memory allocation on any ROCm error
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                isNativeAllocation = false;
            }

            NativePtr = nativePtr;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets whether this buffer uses native HIP allocation.
        /// </summary>
        public bool IsNativeAllocation => isNativeAllocation;

        #endregion

        #region Memory Operations

        /// <summary>
        /// Sets the contents of this buffer to the given byte value.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="value">The value to write into the buffer.</param>
        /// <param name="targetView">The target view to fill.</param>
        protected internal override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
            if (stream is ROCmStream rocmStream)
            {
                rocmStream.MemsetAsync(this, targetView.Index, value, targetView.Length);
            }
            else
            {
                // Synchronous fallback
                SetMemoryToValue(targetView.Index, value, targetView.Length);
            }
        }

        /// <summary>
        /// Copies data from the given buffer to this buffer.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="sourceView">The source buffer view.</param>
        /// <param name="targetView">The target buffer view.</param>
        protected internal override unsafe void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            if (stream is ROCmStream rocmStream)
            {
                // TODO: ROCm async copy not implemented - ArrayView.Source property not available
                throw new NotSupportedException("ROCm async copy operations not implemented");
            }
            else
            {
                // Synchronous fallback
                var sourcePtr = sourceView.LoadEffectiveAddress();
                var targetPtr = nativePtr + targetView.Index;

                if (isNativeAllocation)
                {
                    // Copy from host to device
                    try
                    {
                        var result = ROCmNative.Memcpy(
                            targetPtr, sourcePtr, (nuint)targetView.Length, 
                            HipMemcpyKind.HostToDevice);
                        ROCmException.ThrowIfFailed(result);
                    }
                    catch (Exception)
                    {
                        // Fall back to unsafe copy
                        unsafe
                        {
                            Buffer.MemoryCopy(
                                (void*)sourcePtr, (void*)targetPtr,
                                LengthInBytes - targetView.Index, targetView.Length);
                        }
                    }
                }
                else
                {
                    // Host-to-host copy
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)sourcePtr, (void*)targetPtr,
                            LengthInBytes - targetView.Index, targetView.Length);
                    }
                }
            }
        }

        /// <summary>
        /// Copies data from this buffer to the given buffer.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="sourceView">The source buffer view.</param>
        /// <param name="targetView">The target buffer view.</param>
        protected internal override unsafe void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            if (stream is ROCmStream rocmStream)
            {
                // TODO: ROCm async copy not implemented - ArrayView.Source property not available
                throw new NotSupportedException("ROCm async copy operations not implemented");
            }
            else
            {
                // Synchronous fallback
                var sourcePtr = nativePtr + sourceView.Index;
                var targetPtr = targetView.LoadEffectiveAddress();

                if (isNativeAllocation)
                {
                    // Copy from device to host
                    try
                    {
                        var result = ROCmNative.Memcpy(
                            targetPtr, sourcePtr, (nuint)sourceView.Length,
                            HipMemcpyKind.DeviceToHost);
                        ROCmException.ThrowIfFailed(result);
                    }
                    catch (Exception)
                    {
                        // Fall back to unsafe copy
                        unsafe
                        {
                            Buffer.MemoryCopy(
                                (void*)sourcePtr, (void*)targetPtr,
                                targetView.Length, sourceView.Length);
                        }
                    }
                }
                else
                {
                    // Host-to-host copy
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)sourcePtr, (void*)targetPtr,
                            targetView.Length, sourceView.Length);
                    }
                }
            }
        }

        /// <summary>
        /// Sets memory to a specific value synchronously.
        /// </summary>
        /// <param name="offset">The offset in bytes.</param>
        /// <param name="value">The value to set.</param>
        /// <param name="length">The length in bytes.</param>
        private unsafe void SetMemoryToValue(nint offset, byte value, nint length)
        {
            var ptr = nativePtr + offset;

            if (isNativeAllocation)
            {
                try
                {
                    var result = ROCmNative.Memset(ptr, value, (nuint)length);
                    ROCmException.ThrowIfFailed(result);
                }
                catch (Exception)
                {
                    // Fall back to CPU memset
                    var bytePtr = (byte*)ptr;
                    for (nint i = 0; i < length; i++)
                        bytePtr[i] = value;
                }
            }
            else
            {
                // CPU memset for host memory
                var bytePtr = (byte*)ptr;
                for (nint i = 0; i < length; i++)
                    bytePtr[i] = value;
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this ROCm memory buffer.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && nativePtr != IntPtr.Zero)
            {
                if (isNativeAllocation)
                {
                    try
                    {
                        // Free HIP memory
                        ROCmNative.Free(nativePtr);
                    }
                    catch
                    {
                        // Ignore errors during disposal
                    }
                }
                else
                {
                    // Free host memory
                    Marshal.FreeHGlobal(nativePtr);
                }

                nativePtr = IntPtr.Zero;
                NativePtr = IntPtr.Zero;
            }
        }

        #endregion
    }

    /// <summary>
    /// A ROCm-specific page-lock scope implementation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class ROCmPageLockScope<T> : PageLockScope<T>
        where T : unmanaged
    {
        /// <summary>
        /// Gets the associated ROCm accelerator.
        /// </summary>
        public new ROCmAccelerator Accelerator => base.Accelerator.AsNotNullCast<ROCmAccelerator>();

        /// <summary>
        /// Initializes a new ROCm page-lock scope.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="pinned">The pinned host memory pointer.</param>
        /// <param name="numElements">The number of elements.</param>
        internal ROCmPageLockScope(
            ROCmAccelerator accelerator,
            IntPtr pinned,
            long numElements)
            : base(accelerator, pinned, numElements)
        {
            // ROCm page-locking would be implemented here using hipHostRegister
            // For now, we assume the memory is already pinned
        }

        /// <summary>
        /// Disposes this page-lock scope.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                // ROCm page-unlocking would be implemented here using hipHostUnregister
                // For now, no action needed as we didn't register the memory
            }
        }
    }
}