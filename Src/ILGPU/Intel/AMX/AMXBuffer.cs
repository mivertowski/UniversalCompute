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
using ILGPU.Runtime;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// AMX memory buffer implementation using CPU memory.
    /// </summary>
    public sealed partial class AMXBuffer : MemoryBuffer
    {
        private readonly IntPtr _nativePtr;
        private readonly long _lengthInBytes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AMXBuffer class.
        /// </summary>
        /// <param name="accelerator">The AMX accelerator.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        public AMXBuffer(AMXAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _lengthInBytes = length * elementSize;
            
            // Allocate aligned memory for optimal AMX performance
            _nativePtr = AllocateAlignedMemory(_lengthInBytes, 64); // 64-byte alignment for AMX
            if (_nativePtr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to allocate AMX buffer memory");
                
            // Set the base class NativePtr property
            NativePtr = _nativePtr;
        }



        /// <summary>
        /// Sets the contents of this buffer to the specified byte value.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="value">The byte value to set.</param>
        /// <param name="targetView">The target view of this buffer.</param>
        protected internal override unsafe void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
            if (!targetView.IsValid)
                return;
                
            var targetPtr = (byte*)targetView.LoadEffectiveAddress();
            var lengthInBytes = targetView.LengthInBytes;
            
            // Use optimized memset for AMX buffers
            for (long i = 0; i < lengthInBytes; i++)
            {
                targetPtr[i] = value;
            }
        }

        /// <summary>
        /// Copies data from source view to target view of this buffer.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="sourceView">The source view.</param>
        /// <param name="targetView">The target view of this buffer.</param>
        protected internal override unsafe void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            if (!sourceView.IsValid || !targetView.IsValid)
                return;
                
            var sourcePtr = (byte*)sourceView.LoadEffectiveAddress();
            var targetPtr = (byte*)targetView.LoadEffectiveAddress();
            var lengthInBytes = Math.Min(sourceView.LengthInBytes, targetView.LengthInBytes);
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, targetView.LengthInBytes, lengthInBytes);
        }

        /// <summary>
        /// Copies data from source view of this buffer to target view.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="sourceView">The source view of this buffer.</param>
        /// <param name="targetView">The target view.</param>
        protected internal override unsafe void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            if (!sourceView.IsValid || !targetView.IsValid)
                return;
                
            var sourcePtr = (byte*)sourceView.LoadEffectiveAddress();
            var targetPtr = (byte*)targetView.LoadEffectiveAddress();
            var lengthInBytes = Math.Min(sourceView.LengthInBytes, targetView.LengthInBytes);
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, targetView.LengthInBytes, lengthInBytes);
        }

        /// <summary>
        /// Disposes the AMX buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativePtr != IntPtr.Zero)
                {
                    FreeAlignedMemory(_nativePtr);
                }
                _disposed = true;
            }
        }

        private static IntPtr AllocateAlignedMemory(long size, int alignment)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return _aligned_malloc((UIntPtr)size, (UIntPtr)alignment);
            }
            else
            {
                // Use posix_memalign for Unix-like systems
                return posix_memalign(out var ptr, (UIntPtr)alignment, (UIntPtr)size) == 0 ? ptr : nint.Zero;
            }
        }

        private static void FreeAlignedMemory(IntPtr ptr)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _aligned_free(ptr);
            }
            else
            {
                free(ptr);
            }
        }

        [LibraryImport("msvcrt.dll")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        private static partial IntPtr _aligned_malloc(UIntPtr size, UIntPtr alignment);

        [LibraryImport("msvcrt.dll")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        private static partial void _aligned_free(IntPtr ptr);

        [LibraryImport("libc", EntryPoint = "posix_memalign")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        private static partial int posix_memalign(out IntPtr ptr, UIntPtr alignment, UIntPtr size);

        [LibraryImport("libc", EntryPoint = "free")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        private static partial void free(IntPtr ptr);
    }

    /// <summary>
    /// AMX kernel implementation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the AMXKernel class.
    /// </remarks>
    /// <param name="accelerator">The AMX accelerator.</param>
    /// <param name="compiledKernel">The compiled kernel.</param>
    public sealed class AMXKernel(AMXAccelerator accelerator, CompiledKernel compiledKernel) : Kernel(accelerator, compiledKernel, null)
    {
        [SuppressMessage("Microsoft.Usage", "CA2213:DisposableFieldsShouldBeDisposed", 
            Justification = "AMXAccelerator disposal is handled appropriately in DisposeAcceleratorObject")]
        private readonly AMXAccelerator _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));


        /// <summary>
        /// Disposes the AMX kernel.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                _accelerator?.Dispose();
            }
        }
    }
}
