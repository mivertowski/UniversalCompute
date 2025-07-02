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

#if ENABLE_ONEAPI_ACCELERATOR
namespace ILGPU.Backends.OneAPI
{
    /// <summary>
    /// OneAPI memory buffer implementation.
    /// </summary>
    public sealed class OneAPIBuffer : MemoryBuffer
    {
        private readonly IntPtr _context;
        private readonly IntPtr _device;
        private readonly IntPtr _queue;
        private readonly IntPtr _memObject;
        private readonly long _sizeInBytes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the OneAPIBuffer class.
        /// </summary>
        /// <param name="accelerator">The OneAPI accelerator.</param>
        /// <param name="context">The context handle.</param>
        /// <param name="device">The device handle.</param>
        /// <param name="queue">The queue handle.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        public OneAPIBuffer(
            OneAPIAccelerator accelerator,
            IntPtr context,
            IntPtr device,
            IntPtr queue,
            long length,
            int elementSize)
            : base(accelerator, length, elementSize)
        {
            _context = context;
            _device = device;
            _queue = queue;
            _sizeInBytes = length * elementSize;

            // Allocate device memory
            _memObject = OneAPIBufferNative.AllocateBuffer(context, _sizeInBytes);
            if (_memObject == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate OneAPI buffer");
        }

        /// <summary>
        /// Gets a pointer to the buffer data.
        /// </summary>
        /// <returns>A pointer to the buffer data.</returns>
        public override unsafe void* NativePtr => (void*)_memObject;

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        /// <param name="source">The source CPU memory.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyFromCPU(
            IntPtr source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            OneAPIBufferNative.EnqueueWriteBuffer(
                _queue,
                _memObject,
                true, // blocking
                (nuint)targetOffset,
                (nuint)length,
                (byte*)source + sourceOffset,
                0,
                null,
                IntPtr.Zero);
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        /// <param name="target">The target CPU memory.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyToCPU(
            IntPtr target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            OneAPIBufferNative.EnqueueReadBuffer(
                _queue,
                _memObject,
                true, // blocking
                (nuint)sourceOffset,
                (nuint)length,
                (byte*)target + targetOffset,
                0,
                null,
                IntPtr.Zero);
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override void CopyFrom(
            MemoryBuffer source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (source is OneAPIBuffer oneapiSource)
            {
                // Device-to-device copy
                OneAPIBufferNative.EnqueueCopyBuffer(
                    _queue,
                    oneapiSource._memObject,
                    _memObject,
                    (nuint)sourceOffset,
                    (nuint)targetOffset,
                    (nuint)length,
                    0,
                    null,
                    IntPtr.Zero);
            }
            else
            {
                // Cross-accelerator copy
                base.CopyFrom(source, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        /// <param name="target">The target buffer.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override void CopyTo(
            MemoryBuffer target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (target is OneAPIBuffer oneapiTarget)
            {
                // Device-to-device copy
                OneAPIBufferNative.EnqueueCopyBuffer(
                    _queue,
                    _memObject,
                    oneapiTarget._memObject,
                    (nuint)sourceOffset,
                    (nuint)targetOffset,
                    (nuint)length,
                    0,
                    null,
                    IntPtr.Zero);
            }
            else
            {
                // Cross-accelerator copy
                base.CopyTo(target, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Disposes the OneAPI buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_memObject != IntPtr.Zero)
                {
                    OneAPIBufferNative.ReleaseBuffer(_memObject);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// OneAPI Unified Shared Memory (USM) buffer.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class OneAPIUSMBuffer<T> : MemoryBuffer<T> where T : unmanaged
    {
        private readonly IntPtr _context;
        private readonly IntPtr _device;
        private readonly IntPtr _usmPtr;
        private readonly long _sizeInBytes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the OneAPIUSMBuffer class.
        /// </summary>
        /// <param name="accelerator">The OneAPI accelerator.</param>
        /// <param name="context">The context handle.</param>
        /// <param name="device">The device handle.</param>
        /// <param name="length">The number of elements.</param>
        public OneAPIUSMBuffer(
            OneAPIAccelerator accelerator,
            IntPtr context,
            IntPtr device,
            long length)
            : base(accelerator, length)
        {
            _context = context;
            _device = device;
            _sizeInBytes = length * Interop.SizeOf<T>();

            // Allocate USM memory
            _usmPtr = OneAPIBufferNative.AllocateUSM(context, device, _sizeInBytes, USMType.Shared);
            if (_usmPtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate USM buffer");
        }

        /// <summary>
        /// Gets a pointer to the USM data accessible from both host and device.
        /// </summary>
        public unsafe T* Ptr => (T*)_usmPtr;

        /// <summary>
        /// Gets or sets an element at the specified index.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>The element value.</returns>
        public unsafe T this[long index]
        {
            get
            {
                if (index < 0 || index >= Length)
                    throw new ArgumentOutOfRangeException(nameof(index));
                return Ptr[index];
            }
            set
            {
                if (index < 0 || index >= Length)
                    throw new ArgumentOutOfRangeException(nameof(index));
                Ptr[index] = value;
            }
        }

        /// <summary>
        /// Copies data from a CPU array to this USM buffer.
        /// </summary>
        /// <param name="source">The source array.</param>
        /// <param name="sourceIndex">The source starting index.</param>
        /// <param name="targetIndex">The target starting index.</param>
        /// <param name="length">The number of elements to copy.</param>
        public unsafe void CopyFromCPU(T[] source, long sourceIndex, long targetIndex, long length)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (sourceIndex < 0 || sourceIndex + length > source.Length)
                throw new ArgumentOutOfRangeException(nameof(sourceIndex));
            if (targetIndex < 0 || targetIndex + length > Length)
                throw new ArgumentOutOfRangeException(nameof(targetIndex));

            fixed (T* sourcePtr = &source[sourceIndex])
            {
                var sizeInBytes = length * Interop.SizeOf<T>();
                Buffer.MemoryCopy(sourcePtr, Ptr + targetIndex, sizeInBytes, sizeInBytes);
            }
        }

        /// <summary>
        /// Copies data from this USM buffer to a CPU array.
        /// </summary>
        /// <param name="target">The target array.</param>
        /// <param name="sourceIndex">The source starting index.</param>
        /// <param name="targetIndex">The target starting index.</param>
        /// <param name="length">The number of elements to copy.</param>
        public unsafe void CopyToCPU(T[] target, long sourceIndex, long targetIndex, long length)
        {
            if (target == null)
                throw new ArgumentNullException(nameof(target));
            if (sourceIndex < 0 || sourceIndex + length > Length)
                throw new ArgumentOutOfRangeException(nameof(sourceIndex));
            if (targetIndex < 0 || targetIndex + length > target.Length)
                throw new ArgumentOutOfRangeException(nameof(targetIndex));

            fixed (T* targetPtr = &target[targetIndex])
            {
                var sizeInBytes = length * Interop.SizeOf<T>();
                Buffer.MemoryCopy(Ptr + sourceIndex, targetPtr, sizeInBytes, sizeInBytes);
            }
        }

        /// <summary>
        /// Disposes the USM buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_usmPtr != IntPtr.Zero)
                {
                    OneAPIBufferNative.FreeUSM(_context, _usmPtr);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Native OneAPI buffer operations.
    /// </summary>
    internal static extern class OneAPIBufferNative
    {
        [DllImport("OpenCL")]
        internal static extern IntPtr clCreateBuffer(
            IntPtr context,
            ulong flags,
            nuint size,
            IntPtr hostPtr,
            out int errCodeRet);

        [DllImport("OpenCL")]
        internal static extern int clReleaseMemObject(IntPtr memobj);

        [DllImport("OpenCL")]
        internal static unsafe partial int clEnqueueReadBuffer(
            IntPtr commandQueue,
            IntPtr buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blockingRead,
            nuint offset,
            nuint size,
            void* ptr,
            uint numEventsInWaitList,
            IntPtr* eventWaitList,
            IntPtr @event);

        [DllImport("OpenCL")]
        internal static unsafe partial int clEnqueueWriteBuffer(
            IntPtr commandQueue,
            IntPtr buffer,
            [MarshalAs(UnmanagedType.Bool)] bool blockingWrite,
            nuint offset,
            nuint size,
            void* ptr,
            uint numEventsInWaitList,
            IntPtr* eventWaitList,
            IntPtr @event);

        [DllImport("OpenCL")]
        internal static unsafe partial int clEnqueueCopyBuffer(
            IntPtr commandQueue,
            IntPtr srcBuffer,
            IntPtr dstBuffer,
            nuint srcOffset,
            nuint dstOffset,
            nuint size,
            uint numEventsInWaitList,
            IntPtr* eventWaitList,
            IntPtr @event);

        internal static IntPtr AllocateBuffer(IntPtr context, long sizeInBytes)
        {
            const ulong CL_MEM_READ_WRITE = 1 << 0;
            var buffer = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                (nuint)sizeInBytes,
                IntPtr.Zero,
                out var errCode);
            
            if (errCode != 0)
                return IntPtr.Zero;
            
            return buffer;
        }

        internal static void ReleaseBuffer(IntPtr buffer)
        {
            clReleaseMemObject(buffer);
        }

        internal static unsafe void EnqueueReadBuffer(
            IntPtr queue,
            IntPtr buffer,
            bool blocking,
            nuint offset,
            nuint size,
            void* ptr,
            uint numEvents,
            IntPtr* events,
            IntPtr @event)
        {
            clEnqueueReadBuffer(queue, buffer, blocking, offset, size, ptr, numEvents, events, @event);
        }

        internal static unsafe void EnqueueWriteBuffer(
            IntPtr queue,
            IntPtr buffer,
            bool blocking,
            nuint offset,
            nuint size,
            void* ptr,
            uint numEvents,
            IntPtr* events,
            IntPtr @event)
        {
            clEnqueueWriteBuffer(queue, buffer, blocking, offset, size, ptr, numEvents, events, @event);
        }

        internal static unsafe void EnqueueCopyBuffer(
            IntPtr queue,
            IntPtr srcBuffer,
            IntPtr dstBuffer,
            nuint srcOffset,
            nuint dstOffset,
            nuint size,
            uint numEvents,
            IntPtr* events,
            IntPtr @event)
        {
            clEnqueueCopyBuffer(queue, srcBuffer, dstBuffer, srcOffset, dstOffset, size, numEvents, events, @event);
        }

        // Intel USM extension functions
        [DllImport("OpenCL", EntryPoint = "clDeviceMemAllocINTEL")]
        private static extern IntPtr clDeviceMemAllocINTEL(
            IntPtr context,
            IntPtr device,
            IntPtr properties,
            nuint size,
            uint alignment,
            out int errCodeRet);

        [DllImport("OpenCL", EntryPoint = "clHostMemAllocINTEL")]
        private static extern IntPtr clHostMemAllocINTEL(
            IntPtr context,
            IntPtr properties,
            nuint size,
            uint alignment,
            out int errCodeRet);

        [DllImport("OpenCL", EntryPoint = "clSharedMemAllocINTEL")]
        private static extern IntPtr clSharedMemAllocINTEL(
            IntPtr context,
            IntPtr device,
            IntPtr properties,
            nuint size,
            uint alignment,
            out int errCodeRet);

        [DllImport("OpenCL", EntryPoint = "clMemFreeINTEL")]
        private static extern int clMemFreeINTEL(IntPtr context, IntPtr ptr);

        internal static IntPtr AllocateUSM(IntPtr context, IntPtr device, long size, USMType type)
        {
            try
            {
                const uint alignment = 32; // 32-byte alignment for optimal performance
                
                return type switch
                {
                    USMType.Device => AllocateDeviceUSM(context, device, size, alignment),
                    USMType.Host => AllocateHostUSM(context, size, alignment),
                    USMType.Shared => AllocateSharedUSM(context, device, size, alignment),
                    _ => IntPtr.Zero
                };
            }
            catch
            {
                // Fall back to regular OpenCL buffer allocation
                return AllocateBuffer(context, size);
            }
        }

        private static IntPtr AllocateDeviceUSM(IntPtr context, IntPtr device, long size, uint alignment)
        {
            try
            {
                var ptr = clDeviceMemAllocINTEL(
                    context,
                    device,
                    IntPtr.Zero, // properties
                    (nuint)size,
                    alignment,
                    out var errCode);

                return errCode == 0 ? ptr : IntPtr.Zero;
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        private static IntPtr AllocateHostUSM(IntPtr context, long size, uint alignment)
        {
            try
            {
                var ptr = clHostMemAllocINTEL(
                    context,
                    IntPtr.Zero, // properties
                    (nuint)size,
                    alignment,
                    out var errCode);

                return errCode == 0 ? ptr : IntPtr.Zero;
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        private static IntPtr AllocateSharedUSM(IntPtr context, IntPtr device, long size, uint alignment)
        {
            try
            {
                var ptr = clSharedMemAllocINTEL(
                    context,
                    device,
                    IntPtr.Zero, // properties
                    (nuint)size,
                    alignment,
                    out var errCode);

                return errCode == 0 ? ptr : IntPtr.Zero;
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        internal static void FreeUSM(IntPtr context, IntPtr ptr)
        {
            try
            {
                if (ptr != IntPtr.Zero)
                {
                    var result = clMemFreeINTEL(context, ptr);
                    if (result != 0)
                    {
                        System.Diagnostics.Debug.WriteLine($"Warning: clMemFreeINTEL failed with error: {result}");
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Warning: USM free failed: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// USM memory types.
    /// </summary>
    internal enum USMType
    {
        /// <summary>
        /// Device memory only.
        /// </summary>
        Device,

        /// <summary>
        /// Host memory only.
        /// </summary>
        Host,

        /// <summary>
        /// Shared memory accessible by both host and device.
        /// </summary>
        Shared
    }
}
#endif