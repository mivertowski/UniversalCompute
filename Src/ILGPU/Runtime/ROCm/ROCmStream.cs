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
using System;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// A ROCm stream for asynchronous kernel execution and memory operations.
    /// </summary>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix
    public sealed class ROCmStream : AcceleratorStream
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
    {
        #region Instance

        /// <summary>
        /// The native HIP stream handle.
        /// </summary>
        internal IntPtr NativePtr { get; private set; }

        /// <summary>
        /// The associated ROCm accelerator.
        /// </summary>
        public new ROCmAccelerator Accelerator => (ROCmAccelerator)base.Accelerator;

        /// <summary>
        /// Initializes a new ROCm stream.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        internal ROCmStream(ROCmAccelerator accelerator)
            : base(accelerator)
        {
            try
            {
                // Create HIP stream
                var result = ROCmNative.StreamCreate(out var stream);
                ROCmException.ThrowIfFailed(result);
                NativePtr = stream;
            }
            catch (DllNotFoundException)
            {
                // If ROCm is not available, use a dummy stream
                NativePtr = new IntPtr(-1); // Dummy stream handle
            }
            catch (EntryPointNotFoundException)
            {
                // If ROCm functions are not found, use a dummy stream
                NativePtr = new IntPtr(-1); // Dummy stream handle
            }
        }

        #endregion

        #region Stream Operations

        /// <summary>
        /// Synchronizes this stream and waits for all operations to complete.
        /// </summary>
        public override void Synchronize()
        {
            if (NativePtr == new IntPtr(-1))
            {
                // Dummy stream - no operation needed
                return;
            }

            try
            {
                var result = ROCmNative.StreamSynchronize(NativePtr);
                ROCmException.ThrowIfFailed(result);
            }
            catch (DllNotFoundException)
            {
                // ROCm not available - assume synchronous execution
            }
            catch (EntryPointNotFoundException)
            {
                // ROCm functions not found - assume synchronous execution
            }
        }

        /// <summary>
        /// Adds a profiling marker to this stream.
        /// </summary>
        /// <returns>The created profiling marker.</returns>
        protected override ProfilingMarker AddProfilingMarkerInternal()
        {
            using var binding = Accelerator.BindScoped();
            return new ROCmProfilingMarker(Accelerator);
        }

        #endregion

        #region Memory Operations

        /// <summary>
        /// Performs an asynchronous memory copy operation.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        /// <param name="target">The target buffer.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The length in bytes.</param>
        internal void CopyMemoryAsync(
            MemoryBuffer source,
            MemoryBuffer target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            var sourcePtr = new IntPtr(source.NativePtr.ToInt64() + sourceOffset);
            var targetPtr = new IntPtr(target.NativePtr.ToInt64() + targetOffset);

            HipMemcpyKind kind;
            if (source is ROCmMemoryBuffer && target is ROCmMemoryBuffer)
                kind = HipMemcpyKind.DeviceToDevice;
            else if (source is ROCmMemoryBuffer)
                kind = HipMemcpyKind.DeviceToHost;
            else kind = target is ROCmMemoryBuffer ? HipMemcpyKind.HostToDevice : HipMemcpyKind.HostToHost;

            try
            {
                var result = ROCmNative.MemcpyAsync(
                    targetPtr, sourcePtr, (ulong)length, kind, NativePtr);
                ROCmException.ThrowIfFailed(result);
            }
            catch (DllNotFoundException)
            {
                // Fall back to synchronous copy
                unsafe
                {
                    Buffer.MemoryCopy(
                        sourcePtr.ToPointer(),
                        targetPtr.ToPointer(),
                        length, length);
                }
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to synchronous copy
                unsafe
                {
                    Buffer.MemoryCopy(
                        sourcePtr.ToPointer(),
                        targetPtr.ToPointer(),
                        length, length);
                }
            }
        }

        /// <summary>
        /// Sets memory to a specific value asynchronously.
        /// </summary>
        /// <param name="buffer">The target buffer.</param>
        /// <param name="offset">The offset in bytes.</param>
        /// <param name="value">The value to set.</param>
        /// <param name="length">The length in bytes.</param>
        internal void MemsetAsync(MemoryBuffer buffer, long offset, byte value, long length)
        {
            var ptr = new IntPtr(buffer.NativePtr.ToInt64() + offset);

            try
            {
                var result = ROCmNative.Memset(ptr, value, (ulong)length);
                ROCmException.ThrowIfFailed(result);
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU memset
                unsafe
                {
                    var bytePtr = (byte*)ptr.ToPointer();
                    for (long i = 0; i < length; i++)
                        bytePtr[i] = value;
                }
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU memset
                unsafe
                {
                    var bytePtr = (byte*)ptr.ToPointer();
                    for (long i = 0; i < length; i++)
                        bytePtr[i] = value;
                }
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this ROCm stream.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && NativePtr != IntPtr.Zero && NativePtr != new IntPtr(-1))
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    ROCmNative.StreamDestroy(NativePtr);
                }
                catch
                {
                    // Ignore errors during disposal
                }
                finally
                {
                    NativePtr = IntPtr.Zero;
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
        }

        #endregion
    }

    /// <summary>
    /// ROCm profiling marker implementation.
    /// </summary>
    internal sealed class ROCmProfilingMarker : ProfilingMarker
    {
        private readonly DateTime _timestamp;

        /// <summary>
        /// Initializes a new ROCm profiling marker.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        internal ROCmProfilingMarker(Accelerator accelerator)
            : base(accelerator)
        {
            _timestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Synchronizes this profiling marker.
        /// </summary>
        public override void Synchronize()
        {
            // ROCm events would be synchronized here in a real implementation
        }

        /// <summary>
        /// Measures the elapsed time from another marker.
        /// </summary>
        /// <param name="marker">The starting marker.</param>
        /// <returns>The elapsed time.</returns>
        public override TimeSpan MeasureFrom(ProfilingMarker marker) => marker is ROCmProfilingMarker rocmMarker
                ? _timestamp - rocmMarker._timestamp
                : throw new ArgumentException("Marker must be a ROCm profiling marker", nameof(marker));

        /// <summary>
        /// Disposes this profiling marker.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose for simple timestamp markers
        }
    }
}