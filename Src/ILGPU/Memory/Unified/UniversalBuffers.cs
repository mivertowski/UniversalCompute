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

using ILGPU.Runtime;
using System;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Base implementation for universal buffers with common stub functionality.
    /// </summary>
    public abstract class BaseUniversalBuffer<T>(long size, UniversalMemoryManager manager) : IUniversalBuffer<T>
        where T : unmanaged
    {
        protected readonly long _length = size;
        protected readonly UniversalMemoryManager _manager = manager ?? throw new ArgumentNullException(nameof(manager));

        public long Length => _length;
        public long SizeInBytes => _length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        public abstract MemoryPlacement Placement { get; }
        public virtual bool SupportsZeroCopy => false;
        public virtual bool SupportsAutoMigration => true;
        public abstract DataLocation CurrentLocation { get; }
        public virtual ArrayView<T> View
        {
            get
            {
                // Default implementation requires GetNativeBuffer to be implemented
                var nativeBuffer = GetNativeBuffer(null);
                return nativeBuffer?.View ?? throw new InvalidOperationException("No native buffer available for view access");
            }
        }

        public virtual ArrayView2D<T, Stride2D.DenseX> As2DView(int width, int height)
        {
            if (width * height > Length)
                throw new ArgumentException("2D view dimensions exceed buffer size");
            return View.As2DView(width, height);
        }
        public virtual ArrayView3D<T, Stride3D.DenseXY> As3DView(int width, int height, int depth)
        {
            if (width * height * depth > Length)
                throw new ArgumentException("3D view dimensions exceed buffer size");
            return View.As3DView(width, height, depth);
        }
        public virtual void CopyFromCPU(ReadOnlySpan<T> source)
        {
            if (source.Length > Length)
                throw new ArgumentException("Source data exceeds buffer capacity");
            
            // Default implementation uses array view
            var view = View.SubView(0, source.Length);
            view.CopyFromCPU(source.ToArray());
        }
        public virtual void CopyToCPU(Span<T> destination)
        {
            if (destination.Length > Length)
                throw new ArgumentException("Destination buffer too small");
            
            // Default implementation uses array view
            var tempArray = new T[destination.Length];
            var view = View.SubView(0, destination.Length);
            view.CopyToCPU(tempArray);
            tempArray.AsSpan().CopyTo(destination);
        }
        public virtual Task CopyFromCPUAsync(ReadOnlyMemory<T> source) =>
            // Default async implementation delegates to sync version
            Task.Run(() => CopyFromCPU(source.Span));
        public virtual Task CopyToCPUAsync(Memory<T> destination) =>
            // Default async implementation delegates to sync version
            Task.Run(() => CopyToCPU(destination.Span));
        public virtual void CopyFrom(IUniversalBuffer<T> source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            
            if (source.Length > Length)
                throw new ArgumentException("Source buffer exceeds destination capacity");
            
            // Default implementation via CPU
            var tempData = new T[source.Length];
            source.CopyToCPU(tempData);
            CopyFromCPU(tempData);
        }
        public virtual Task CopyFromAsync(IUniversalBuffer<T> source) =>
            // Default async implementation delegates to sync version
            Task.Run(() => CopyFrom(source));
        public virtual void EnsureAvailable(Accelerator accelerator)
        {
            // Default implementation - data is already available if native buffer exists
            // Derived classes should override for specific migration logic
            var nativeBuffer = GetNativeBuffer(accelerator);
            if (nativeBuffer == null)
                throw new InvalidOperationException($"Buffer not available on accelerator {accelerator.Name}");
        }
        public virtual Task EnsureAvailableAsync(Accelerator accelerator) =>
            // Default async implementation delegates to sync version
            Task.Run(() => EnsureAvailable(accelerator));
        public virtual void Prefetch(Accelerator accelerator)
        {
            // Default implementation is a no-op
            // Derived classes can override for specific prefetching logic
        }
        public virtual MemoryBuffer1D<T, Stride1D.Dense>? GetNativeBuffer(Accelerator accelerator) =>
            // Base implementation returns null - derived classes must override
            // This method is the core abstraction for accessing native buffers
            null;
        public virtual void InvalidateCache()
        {
            // Default implementation is a no-op
            // Derived classes can override for specific cache invalidation logic
        }
        public virtual UniversalBufferStats GetStats() =>
            // Default implementation returns basic stats
            new()
            {
                SizeInBytes = SizeInBytes,
                CurrentLocation = CurrentLocation,
                Placement = Placement,
                AccessCount = 0,
                LastAccessTime = DateTime.UtcNow
            };

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the buffer.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            // Stub implementation for derived classes to override
        }
    }
    /// <summary>
    /// Apple Silicon unified memory buffer implementation.
    /// </summary>
    public class AppleUnifiedBuffer<T>(long size, UniversalMemoryManager manager) : BaseUniversalBuffer<T>(size, manager)
        where T : unmanaged
    {
        public override MemoryPlacement Placement => MemoryPlacement.AppleUnified;
        public override bool SupportsZeroCopy => true;
        public override DataLocation CurrentLocation => DataLocation.Unified;
    }

    /// <summary>
    /// CUDA managed memory buffer implementation.
    /// </summary>
    public class CudaManagedBuffer<T>(long size, UniversalMemoryManager manager) : BaseUniversalBuffer<T>(size, manager)
        where T : unmanaged
    {
        public override MemoryPlacement Placement => MemoryPlacement.CudaManaged;
        public override DataLocation CurrentLocation => DataLocation.Unified;
    }

    /// <summary>
    /// Intel integrated GPU shared memory buffer implementation.
    /// </summary>
    public class IntelSharedBuffer<T>(long size, UniversalMemoryManager manager) : BaseUniversalBuffer<T>(size, manager)
        where T : unmanaged
    {
        public override MemoryPlacement Placement => MemoryPlacement.IntelShared;
        public override DataLocation CurrentLocation => DataLocation.Both;
    }

    /// <summary>
    /// Pinned host memory buffer implementation.
    /// </summary>
    public class PinnedHostBuffer<T>(long size, UniversalMemoryManager manager) : BaseUniversalBuffer<T>(size, manager)
        where T : unmanaged
    {
        public override MemoryPlacement Placement => MemoryPlacement.HostPinned;
        public override DataLocation CurrentLocation => DataLocation.Host;
    }

    /// <summary>
    /// Standard universal buffer implementation.
    /// </summary>
    public class StandardUniversalBuffer<T>(long size, MemoryPlacement placement, UniversalMemoryManager manager) : BaseUniversalBuffer<T>(size, manager)
        where T : unmanaged
    {
        private readonly MemoryPlacement _placement = placement;

        public override MemoryPlacement Placement => _placement;
        public override DataLocation CurrentLocation => DataLocation.Host;
    }

    /// <summary>
    /// Wraps an existing memory buffer to provide universal buffer functionality.
    /// </summary>
    public class WrappedMemoryBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> memoryBuffer, UniversalMemoryManager manager, bool takeOwnership) : BaseUniversalBuffer<T>(memoryBuffer.Length, manager)
        where T : unmanaged
    {
        private readonly MemoryBuffer1D<T, Stride1D.Dense> _wrappedBuffer = memoryBuffer ?? throw new ArgumentNullException(nameof(memoryBuffer));
        private readonly bool _takeOwnership = takeOwnership;

        public override MemoryPlacement Placement => MemoryPlacement.DeviceLocal;
        public override DataLocation CurrentLocation => DataLocation.Device;
        public override ArrayView<T> View => _wrappedBuffer.View;

        protected override void Dispose(bool disposing)
        {
            if (disposing && _takeOwnership)
            {
                _wrappedBuffer?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}