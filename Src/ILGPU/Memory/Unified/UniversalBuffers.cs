// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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
using System.Numerics;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Base implementation for universal buffers with common stub functionality.
    /// </summary>
    public abstract class BaseUniversalBuffer<T> : IUniversalBuffer<T>
        where T : unmanaged
    {
        protected readonly long _length;
        protected readonly UniversalMemoryManager _manager;

        protected BaseUniversalBuffer(long size, UniversalMemoryManager manager)
        {
            _length = size;
            _manager = manager ?? throw new ArgumentNullException(nameof(manager));
        }

        public long Length => _length;
        public long SizeInBytes => _length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        public abstract MemoryPlacement Placement { get; }
        public virtual bool SupportsZeroCopy => false;
        public virtual bool SupportsAutoMigration => true;
        public abstract DataLocation CurrentLocation { get; }
        public virtual ArrayView<T> View => throw new NotImplementedException();

        public virtual ArrayView2D<T, Stride2D.DenseX> As2DView(int width, int height) => throw new NotImplementedException();
        public virtual ArrayView3D<T, Stride3D.DenseXY> As3DView(int width, int height, int depth) => throw new NotImplementedException();
        public virtual void CopyFromCPU(ReadOnlySpan<T> source) => throw new NotImplementedException();
        public virtual void CopyToCPU(Span<T> destination) => throw new NotImplementedException();
        public virtual Task CopyFromCPUAsync(ReadOnlyMemory<T> source) => throw new NotImplementedException();
        public virtual Task CopyToCPUAsync(Memory<T> destination) => throw new NotImplementedException();
        public virtual void CopyFrom(IUniversalBuffer<T> source) => throw new NotImplementedException();
        public virtual Task CopyFromAsync(IUniversalBuffer<T> source) => throw new NotImplementedException();
        public virtual void EnsureAvailable(Accelerator accelerator) => throw new NotImplementedException();
        public virtual Task EnsureAvailableAsync(Accelerator accelerator) => throw new NotImplementedException();
        public virtual void Prefetch(Accelerator accelerator) => throw new NotImplementedException();
        public virtual MemoryBuffer1D<T, Stride1D.Dense>? GetNativeBuffer(Accelerator accelerator) => throw new NotImplementedException();
        public virtual void InvalidateCache() => throw new NotImplementedException();
        public virtual UniversalBufferStats GetStats() => throw new NotImplementedException();

        public virtual void Dispose()
        {
            // Stub implementation
        }
    }
    /// <summary>
    /// Apple Silicon unified memory buffer implementation.
    /// </summary>
    public class AppleUnifiedBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        public AppleUnifiedBuffer(long size, UniversalMemoryManager manager) : base(size, manager) { }

        public override MemoryPlacement Placement => MemoryPlacement.AppleUnified;
        public override bool SupportsZeroCopy => true;
        public override DataLocation CurrentLocation => DataLocation.Unified;
    }

    /// <summary>
    /// CUDA managed memory buffer implementation.
    /// </summary>
    public class CudaManagedBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        public CudaManagedBuffer(long size, UniversalMemoryManager manager) : base(size, manager) { }

        public override MemoryPlacement Placement => MemoryPlacement.CudaManaged;
        public override DataLocation CurrentLocation => DataLocation.Unified;
    }

    /// <summary>
    /// Intel integrated GPU shared memory buffer implementation.
    /// </summary>
    public class IntelSharedBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        public IntelSharedBuffer(long size, UniversalMemoryManager manager) : base(size, manager) { }

        public override MemoryPlacement Placement => MemoryPlacement.IntelShared;
        public override DataLocation CurrentLocation => DataLocation.Both;
    }

    /// <summary>
    /// Pinned host memory buffer implementation.
    /// </summary>
    public class PinnedHostBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        public PinnedHostBuffer(long size, UniversalMemoryManager manager) : base(size, manager) { }

        public override MemoryPlacement Placement => MemoryPlacement.HostPinned;
        public override DataLocation CurrentLocation => DataLocation.Host;
    }

    /// <summary>
    /// Standard universal buffer implementation.
    /// </summary>
    public class StandardUniversalBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        private readonly MemoryPlacement _placement;

        public StandardUniversalBuffer(long size, MemoryPlacement placement, UniversalMemoryManager manager) 
            : base(size, manager)
        {
            _placement = placement;
        }

        public override MemoryPlacement Placement => _placement;
        public override DataLocation CurrentLocation => DataLocation.Host;
    }

    /// <summary>
    /// Wraps an existing memory buffer to provide universal buffer functionality.
    /// </summary>
    public class WrappedMemoryBuffer<T> : BaseUniversalBuffer<T>
        where T : unmanaged
    {
        private readonly MemoryBuffer1D<T, Stride1D.Dense> _wrappedBuffer;
        private readonly bool _takeOwnership;

        public WrappedMemoryBuffer(MemoryBuffer1D<T, Stride1D.Dense> memoryBuffer, UniversalMemoryManager manager, bool takeOwnership)
            : base(memoryBuffer.Length, manager)
        {
            _wrappedBuffer = memoryBuffer ?? throw new ArgumentNullException(nameof(memoryBuffer));
            _takeOwnership = takeOwnership;
        }

        public override MemoryPlacement Placement => MemoryPlacement.DeviceLocal;
        public override DataLocation CurrentLocation => DataLocation.Device;
        public override ArrayView<T> View => _wrappedBuffer.View;

        public override void Dispose()
        {
            if (_takeOwnership)
            {
                _wrappedBuffer?.Dispose();
            }
            base.Dispose();
        }
    }
}