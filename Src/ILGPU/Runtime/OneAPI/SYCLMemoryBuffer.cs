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
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// A SYCL-specific memory buffer implementation for Intel OneAPI devices.
    /// </summary>
    public sealed class SYCLMemoryBuffer : MemoryBuffer
    {
        #region Instance

        /// <summary>
        /// The native SYCL memory pointer.
        /// </summary>
        private IntPtr nativePtr;

        /// <summary>
        /// Gets the associated Intel OneAPI accelerator.
        /// </summary>
        public new IntelOneAPIAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelOneAPIAccelerator>();

        /// <summary>
        /// Initializes a new SYCL memory buffer.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="length">The length of this buffer.</param>
        /// <param name="elementSize">The element size.</param>
        internal SYCLMemoryBuffer(
            IntelOneAPIAccelerator accelerator,
            long length,
            int elementSize)
            : base(accelerator, length, elementSize)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var totalSize = new UIntPtr((ulong)LengthInBytes);
                
                // Try to allocate using SYCL device memory
                nativePtr = SYCLNative.MallocDevice(
                    totalSize,
                    accelerator.DeviceHandle,
                    accelerator.ContextHandle);

                if (nativePtr == IntPtr.Zero)
                {
                    // Fall back to shared memory allocation
                    nativePtr = SYCLNative.MallocShared(
                        totalSize,
                        accelerator.DeviceHandle,
                        accelerator.ContextHandle);
                }

                if (nativePtr == IntPtr.Zero)
                {
                    throw new InvalidOperationException("Failed to allocate SYCL memory");
                }

                IsNativeAllocation = true;
            }
            catch (DllNotFoundException)
            {
                // Fall back to host memory allocation
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                IsNativeAllocation = false;
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to host memory allocation
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                IsNativeAllocation = false;
            }
            catch (Exception)
            {
                // Fall back to host memory allocation on any SYCL error
                nativePtr = Marshal.AllocHGlobal(new IntPtr(LengthInBytes));
                IsNativeAllocation = false;
            }
#pragma warning restore CA1031 // Do not catch general exception types

            NativePtr = nativePtr;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets whether this buffer uses native SYCL allocation.
        /// </summary>
        public bool IsNativeAllocation { get; }

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
            if (stream is OneAPIStream syclStream && IsNativeAllocation)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // Use SYCL async memset
                    var targetPtr = nativePtr + (nint)targetView.Index;
                    SYCLNative.Memset(
                        syclStream.NativeQueue,
                        targetPtr,
                        value,
                        new UIntPtr((ulong)targetView.Length));
                }
                catch (Exception)
                {
                    // Fall back to synchronous implementation
                    SetMemoryToValue(targetView.Index, value, targetView.Length);
                }
#pragma warning restore CA1031 // Do not catch general exception types
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
            if (stream is OneAPIStream syclStream && 
                sourceView.Buffer is SYCLMemoryBuffer sourceBuffer &&
                IsNativeAllocation && sourceBuffer.IsNativeAllocation)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // SYCL device-to-device copy
                    var sourcePtr = sourceBuffer.nativePtr + (nint)sourceView.Index;
                    var targetPtr = nativePtr + (nint)targetView.Index;

                    SYCLNative.Memcpy(
                        syclStream.NativeQueue,
                        targetPtr,
                        sourcePtr,
                        new UIntPtr((ulong)targetView.Length));
                }
                catch (Exception)
                {
                    // Fall back to unsafe copy
                    var sourcePtr = sourceView.LoadEffectiveAddressAsPtr();
                    var targetPtr = nativePtr + (nint)targetView.Index;
                    Buffer.MemoryCopy(
                        (void*)sourcePtr, (void*)targetPtr,
                        LengthInBytes - targetView.Index, targetView.Length);
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
            else
            {
                // Synchronous fallback
                var sourcePtr = sourceView.LoadEffectiveAddressAsPtr();
                var targetPtr = nativePtr + (nint)targetView.Index;

                if (IsNativeAllocation)
                {
                    // Copy from host to device using SYCL
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        SYCLNative.Memcpy(
                            Accelerator.SYCLQueue,
                            targetPtr,
                            sourcePtr,
                            new UIntPtr((ulong)targetView.Length));
                    }
                    catch (Exception)
                    {
                        // Fall back to unsafe copy
                        Buffer.MemoryCopy(
                            (void*)sourcePtr, (void*)targetPtr,
                            LengthInBytes - targetView.Index, targetView.Length);
                    }
#pragma warning restore CA1031 // Do not catch general exception types
                }
                else
                {
                    // Host-to-host copy
                    Buffer.MemoryCopy(
                        (void*)sourcePtr, (void*)targetPtr,
                        LengthInBytes - targetView.Index, targetView.Length);
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
            if (stream is OneAPIStream syclStream && 
                targetView.Buffer is SYCLMemoryBuffer targetBuffer &&
                IsNativeAllocation && targetBuffer.IsNativeAllocation)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // SYCL device-to-device copy
                    var sourcePtr = nativePtr + (nint)sourceView.Index;
                    var targetPtr = targetBuffer.nativePtr + (nint)targetView.Index;

                    SYCLNative.Memcpy(
                        syclStream.NativeQueue,
                        targetPtr,
                        sourcePtr,
                        new UIntPtr((ulong)sourceView.Length));
                }
                catch (Exception)
                {
                    // Fall back to unsafe copy
                    var sourcePtr = nativePtr + (nint)sourceView.Index;
                    var targetPtr = targetView.LoadEffectiveAddressAsPtr();
                    Buffer.MemoryCopy(
                        (void*)sourcePtr, (void*)targetPtr,
                        targetView.Length, sourceView.Length);
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
            else
            {
                // Synchronous fallback
                var sourcePtr = nativePtr + (nint)sourceView.Index;
                var targetPtr = targetView.LoadEffectiveAddressAsPtr();

                if (IsNativeAllocation)
                {
                    // Copy from device to host using SYCL
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        SYCLNative.Memcpy(
                            Accelerator.SYCLQueue,
                            targetPtr,
                            sourcePtr,
                            new UIntPtr((ulong)sourceView.Length));
                    }
                    catch (Exception)
                    {
                        // Fall back to unsafe copy
                        Buffer.MemoryCopy(
                            (void*)sourcePtr, (void*)targetPtr,
                            targetView.Length, sourceView.Length);
                    }
#pragma warning restore CA1031 // Do not catch general exception types
                }
                else
                {
                    // Host-to-host copy
                    Buffer.MemoryCopy(
                        (void*)sourcePtr, (void*)targetPtr,
                        targetView.Length, sourceView.Length);
                }
            }
        }

        /// <summary>
        /// Sets memory to a specific value synchronously.
        /// </summary>
        /// <param name="offset">The offset in bytes.</param>
        /// <param name="value">The value to set.</param>
        /// <param name="length">The length in bytes.</param>
        private unsafe void SetMemoryToValue(long offset, byte value, long length)
        {
            var ptr = nativePtr + (nint)offset;

            if (IsNativeAllocation)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // Use SYCL synchronous memset
                    SYCLNative.Memset(
                        Accelerator.SYCLQueue,
                        ptr,
                        value,
                        new UIntPtr((ulong)length));
                }
                catch (Exception)
                {
                    // Fall back to CPU memset
                    var bytePtr = (byte*)(void*)ptr;
                    for (long i = 0; i < length; i++)
                        bytePtr[i] = value;
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
            else
            {
                // CPU memset for host memory
                var bytePtr = (byte*)(void*)ptr;
                for (long i = 0; i < length; i++)
                    bytePtr[i] = value;
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this SYCL memory buffer.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (nativePtr != IntPtr.Zero)
            {
                if (IsNativeAllocation)
                {
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        // Free SYCL memory
                        SYCLNative.Free(nativePtr, Accelerator.ContextHandle);
                    }
                    catch
                    {
                        // Ignore errors during disposal
                    }
#pragma warning restore CA1031 // Do not catch general exception types
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
}