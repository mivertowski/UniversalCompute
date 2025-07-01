// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed under an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

using ILGPU.Backends;
using ILGPU.Runtime.AMX.Native;
using System;
using static ILGPU.Runtime.AMX.Native.AMXNative;

namespace ILGPU.Runtime.AMX
{
    /// <summary>
    /// An Intel AMX stream for matrix operations execution.
    /// </summary>
    public sealed class AMXStream : AcceleratorStream
    {
        #region Instance

        /// <summary>
        /// The associated Intel AMX accelerator.
        /// </summary>
        public new IntelAMXAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelAMXAccelerator>();

        /// <summary>
        /// Gets whether this stream is currently configured for tile operations.
        /// </summary>
        public bool IsTileConfigured { get; private set; }

        /// <summary>
        /// Initializes a new Intel AMX stream.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        internal AMXStream(IntelAMXAccelerator accelerator)
            : base(accelerator)
        {
            IsTileConfigured = false;
        }

        #endregion

        #region Stream Operations

        /// <summary>
        /// Synchronizes this stream and waits for all operations to complete.
        /// </summary>
        public override void Synchronize()
        {
            // AMX operations are synchronous on CPU, no explicit synchronization needed
        }

        /// <summary>
        /// Adds a profiling marker to this stream.
        /// </summary>
        /// <returns>The created profiling marker.</returns>
        protected override ProfilingMarker AddProfilingMarkerInternal()
        {
            using var binding = Accelerator.BindScoped();
            return new AMXProfilingMarker(Accelerator);
        }

        #endregion

        #region AMX Operations

        /// <summary>
        /// Configures AMX tiles for matrix operations.
        /// </summary>
        /// <param name="config">Tile configuration.</param>
        internal void ConfigureTiles(AMXTileConfig config)
        {
            try
            {
                var result = AMXNative.ConfigureTiles(ref config);
                if (result != 0)
                    throw new AMXException($"Failed to configure AMX tiles: {result}");
                
                IsTileConfigured = true;
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel AMX runtime not available");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel AMX functions not found");
            }
        }

        /// <summary>
        /// Releases AMX tile configuration.
        /// </summary>
        public void ReleaseTiles()
        {
            try
            {
                if (IsTileConfigured)
                {
                    AMXNative.ReleaseTiles();
                    IsTileConfigured = false;
                }
            }
            catch (Exception)
            {
                // Ignore errors during tile release
            }
        }

        /// <summary>
        /// Executes a matrix multiplication operation on this stream.
        /// </summary>
        /// <param name="a">Matrix A data.</param>
        /// <param name="b">Matrix B data.</param>
        /// <param name="c">Result matrix C data.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="dataType">Data type for computation.</param>
        internal unsafe void ExecuteMatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            AMXDataType dataType)
        {
            Accelerator.ExecuteMatMul(a, b, c, m, k, n, dataType, this);
        }

        /// <summary>
        /// Loads data into an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        /// <param name="src">Source data pointer.</param>
        /// <param name="stride">Stride in bytes.</param>
        public void TileLoad(byte tile, IntPtr src, long stride)
        {
            if (!IsTileConfigured)
                throw new InvalidOperationException("AMX tiles must be configured before loading data");

            try
            {
                AMXNative.TileLoad(tile, src, stride);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel AMX runtime not available");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel AMX functions not found");
            }
        }

        /// <summary>
        /// Stores data from an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        /// <param name="dst">Destination data pointer.</param>
        /// <param name="stride">Stride in bytes.</param>
        public void TileStore(byte tile, IntPtr dst, long stride)
        {
            if (!IsTileConfigured)
                throw new InvalidOperationException("AMX tiles must be configured before storing data");

            try
            {
                AMXNative.TileStore(tile, dst, stride);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel AMX runtime not available");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel AMX functions not found");
            }
        }

        /// <summary>
        /// Zeros an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        public void TileZero(byte tile)
        {
            if (!IsTileConfigured)
                throw new InvalidOperationException("AMX tiles must be configured before zeroing");

            try
            {
                AMXNative.TileZero(tile);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel AMX runtime not available");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel AMX functions not found");
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this Intel AMX stream.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                try
                {
                    // Release tile configuration
                    ReleaseTiles();
                }
                catch
                {
                    // Ignore errors during disposal
                }
            }
        }

        #endregion
    }

    /// <summary>
    /// Intel AMX profiling marker implementation.
    /// </summary>
    internal sealed class AMXProfilingMarker : ProfilingMarker
    {
        private readonly DateTime _timestamp;

        /// <summary>
        /// Initializes a new Intel AMX profiling marker.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        internal AMXProfilingMarker(Accelerator accelerator)
            : base(accelerator)
        {
            _timestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Synchronizes this profiling marker.
        /// </summary>
        public override void Synchronize()
        {
            // AMX operations are synchronous - no action needed
        }

        /// <summary>
        /// Measures the elapsed time from another marker.
        /// </summary>
        /// <param name="marker">The starting marker.</param>
        /// <returns>The elapsed time.</returns>
        public override TimeSpan MeasureFrom(ProfilingMarker marker)
        {
            if (marker is AMXProfilingMarker amxMarker)
                return _timestamp - amxMarker._timestamp;
            throw new ArgumentException("Marker must be an AMX profiling marker", nameof(marker));
        }

        /// <summary>
        /// Disposes this profiling marker.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose for timestamp markers
        }
    }

    /// <summary>
    /// Intel AMX memory buffer implementation.
    /// </summary>
    public sealed class AMXMemoryBuffer : MemoryBuffer
    {
        private IntPtr nativePtr;

        public new IntelAMXAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelAMXAccelerator>();

        internal AMXMemoryBuffer(IntelAMXAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            // Allocate aligned memory for optimal AMX performance (64-byte aligned)
            nativePtr = System.Runtime.InteropServices.Marshal.AllocHGlobal(
                new IntPtr(LengthInBytes + 64));
            
            // Align to 64-byte boundary
            long aligned = (nativePtr.ToInt64() + 63) & ~63L;
            nativePtr = new IntPtr(aligned);
            
            NativePtr = nativePtr;
        }

        protected internal override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
        {
            unsafe
            {
                var ptr = (byte*)nativePtr + targetView.Index;
                for (long i = 0; i < targetView.Length; i++)
                    ptr[i] = value;
            }
        }

        protected internal override unsafe void CopyFrom(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            var sourcePtr = sourceView.LoadEffectiveAddress();
            var targetPtr = nativePtr + targetView.Index;
            Buffer.MemoryCopy(sourcePtr.ToPointer(), targetPtr.ToPointer(), 
                LengthInBytes - targetView.Index, targetView.Length);
        }

        protected internal override unsafe void CopyTo(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            var sourcePtr = nativePtr + sourceView.Index;
            var targetPtr = targetView.LoadEffectiveAddress();
            Buffer.MemoryCopy(sourcePtr.ToPointer(), targetPtr.ToPointer(), 
                targetView.Length, sourceView.Length);
        }

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && nativePtr != IntPtr.Zero)
            {
                // Need to free the original unaligned pointer
                var originalPtr = new IntPtr((nativePtr.ToInt64() & ~63L) - 64);
                System.Runtime.InteropServices.Marshal.FreeHGlobal(originalPtr);
                nativePtr = IntPtr.Zero;
                NativePtr = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Intel AMX page-lock scope implementation.
    /// </summary>
    public sealed class AMXPageLockScope<T> : PageLockScope<T> where T : unmanaged
    {
        public new IntelAMXAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelAMXAccelerator>();

        internal AMXPageLockScope(IntelAMXAccelerator accelerator, IntPtr pinned, long numElements)
            : base(accelerator, pinned, numElements)
        {
            // AMX doesn't require explicit page-locking like GPU accelerators
        }

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No special cleanup needed for AMX
        }
    }

    /// <summary>
    /// Intel AMX compiled kernel placeholder.
    /// </summary>
    public class AMXCompiledKernel : CompiledKernel
    {
        public byte[] NativeCode { get; }
        public string FunctionName { get; }

        public AMXCompiledKernel(Context context, byte[] nativeCode, string functionName, KernelInfo info)
            : base(context, info, null)
        {
            NativeCode = nativeCode ?? throw new ArgumentNullException(nameof(nativeCode));
            FunctionName = functionName ?? throw new ArgumentNullException(nameof(functionName));
        }

        protected override void DisposeAcceleratorObject(bool disposing) { }
    }

    /// <summary>
    /// Intel AMX kernel implementation.
    /// </summary>
    public sealed class AMXKernel : Kernel
    {
        public new IntelAMXAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelAMXAccelerator>();
        public new AMXCompiledKernel CompiledKernel => base.CompiledKernel.AsNotNullCast<AMXCompiledKernel>();

        internal AMXKernel(IntelAMXAccelerator accelerator, AMXCompiledKernel compiledKernel)
            : base(accelerator, compiledKernel, null)
        {
        }

        protected override void DisposeAcceleratorObject(bool disposing) { }
    }
}