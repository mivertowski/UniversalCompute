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
using System.Collections.Generic;

namespace ILGPU.Numerics
{
    /// <summary>
    /// Interface for tensor operations.
    /// </summary>
    public interface ITensor : IDisposable
    {
        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        int[] Shape { get; }

        /// <summary>
        /// Gets the total number of elements.
        /// </summary>
        long ElementCount { get; }

        /// <summary>
        /// Gets the data type of the tensor elements.
        /// </summary>
        Type ElementType { get; }

        /// <summary>
        /// Gets the memory location of the tensor data.
        /// </summary>
        MemoryLocation Location { get; }

        /// <summary>
        /// Gets the associated accelerator.
        /// </summary>
        Accelerator Accelerator { get; }

        /// <summary>
        /// Copies data to the CPU.
        /// </summary>
        Array CopyToCPU();

        /// <summary>
        /// Reshapes the tensor to new dimensions.
        /// </summary>
        ITensor Reshape(int[] newShape);

        /// <summary>
        /// Creates a view of the tensor data.
        /// </summary>
        ITensor View(int[] start, int[] end);
    }

    /// <summary>
    /// Generic tensor interface.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public interface ITensor<T> : ITensor
        where T : unmanaged
    {
        /// <summary>
        /// Gets the raw data buffer.
        /// </summary>
        MemoryBuffer<T> Buffer { get; }

        /// <summary>
        /// Gets a view of the tensor data.
        /// </summary>
        ArrayView<T> View { get; }

        /// <summary>
        /// Copies data from a CPU array.
        /// </summary>
        void CopyFromCPU(T[] data);

        /// <summary>
        /// Copies data to a CPU array.
        /// </summary>
        T[] CopyToCPU();

        /// <summary>
        /// Creates a typed view of the tensor.
        /// </summary>
        new ITensor<T> Reshape(int[] newShape);

        /// <summary>
        /// Creates a typed view of the tensor data.
        /// </summary>
        new ITensor<T> View(int[] start, int[] end);
    }

    /// <summary>
    /// Factory for creating tensors.
    /// </summary>
    public interface ITensorFactory
    {
        /// <summary>
        /// Creates a tensor with the specified shape and data type.
        /// </summary>
        ITensor Create(Type elementType, int[] shape);

        /// <summary>
        /// Creates a typed tensor with the specified shape.
        /// </summary>
        ITensor<T> Create<T>(int[] shape) where T : unmanaged;

        /// <summary>
        /// Creates a tensor from CPU data.
        /// </summary>
        ITensor<T> FromArray<T>(T[] data, int[] shape) where T : unmanaged;

        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        ITensor<T> Zeros<T>(int[] shape) where T : unmanaged;

        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        ITensor<T> Ones<T>(int[] shape) where T : unmanaged;

        /// <summary>
        /// Creates a tensor filled with random values.
        /// </summary>
        ITensor<T> Random<T>(int[] shape, Random random = null) where T : unmanaged;
    }

    /// <summary>
    /// Memory manager interface for tensor operations.
    /// </summary>
    public interface IMemoryManager : IDisposable
    {
        /// <summary>
        /// Allocates memory for a tensor.
        /// </summary>
        MemoryBuffer<T> Allocate<T>(long elementCount) where T : unmanaged;

        /// <summary>
        /// Deallocates memory.
        /// </summary>
        void Deallocate<T>(MemoryBuffer<T> buffer) where T : unmanaged;

        /// <summary>
        /// Gets memory usage statistics.
        /// </summary>
        MemoryUsageInfo GetUsageInfo();

        /// <summary>
        /// Copies data between memory locations.
        /// </summary>
        void Copy<T>(MemoryBuffer<T> source, MemoryBuffer<T> destination, long elementCount)
            where T : unmanaged;

        /// <summary>
        /// Checks if memory transfer is required.
        /// </summary>
        bool RequiresTransfer(MemoryLocation source, MemoryLocation destination);
    }

    /// <summary>
    /// Memory usage information.
    /// </summary>
    public struct MemoryUsageInfo
    {
        /// <summary>
        /// Total allocated memory in bytes.
        /// </summary>
        public long TotalAllocatedBytes { get; set; }

        /// <summary>
        /// Total available memory in bytes.
        /// </summary>
        public long TotalAvailableBytes { get; set; }

        /// <summary>
        /// Number of active allocations.
        /// </summary>
        public int ActiveAllocations { get; set; }

        /// <summary>
        /// Largest single allocation size.
        /// </summary>
        public long LargestAllocationBytes { get; set; }

        /// <summary>
        /// Memory fragmentation percentage.
        /// </summary>
        public double FragmentationPercent { get; set; }
    }

    /// <summary>
    /// Memory location enumeration.
    /// </summary>
    public enum MemoryLocation
    {
        /// <summary>
        /// Data is on the CPU.
        /// </summary>
        CPU,

        /// <summary>
        /// Data is on the GPU.
        /// </summary>
        GPU,

        /// <summary>
        /// Data is in unified memory.
        /// </summary>
        Unified,

        /// <summary>
        /// Data location is unknown or mixed.
        /// </summary>
        Unknown
    }

    /// <summary>
    /// Tensor implementation using ILGPU buffers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public class ILGPUTensor<T> : ITensor<T>
        where T : unmanaged
    {
        private MemoryBuffer<T> _buffer;
        private int[] _shape;
        private readonly Accelerator _accelerator;
        private bool _disposed = false;

        /// <summary>
        /// Initializes a new instance of the ILGPUTensor class.
        /// </summary>
        public ILGPUTensor(Accelerator accelerator, int[] shape)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            
            var elementCount = CalculateElementCount(shape);
            _buffer = accelerator.Allocate1D<T>(elementCount);
        }

        /// <inheritdoc/>
        public int[] Shape => _shape;

        /// <inheritdoc/>
        public long ElementCount => CalculateElementCount(_shape);

        /// <inheritdoc/>
        public Type ElementType => typeof(T);

        /// <inheritdoc/>
        public MemoryLocation Location => MemoryLocation.GPU;

        /// <inheritdoc/>
        public Accelerator Accelerator => _accelerator;

        /// <inheritdoc/>
        public MemoryBuffer<T> Buffer => _buffer;

        /// <inheritdoc/>
        public ArrayView<T> View => _buffer.View;

        /// <inheritdoc/>
        public void CopyFromCPU(T[] data)
        {
            if (data.Length != ElementCount)
                throw new ArgumentException("Data length doesn't match tensor size");
            
            _buffer.CopyFromCPU(data);
        }

        /// <inheritdoc/>
        public T[] CopyToCPU()
        {
            return _buffer.GetAsArray1D();
        }

        /// <inheritdoc/>
        Array ITensor.CopyToCPU()
        {
            return CopyToCPU();
        }

        /// <inheritdoc/>
        public ITensor<T> Reshape(int[] newShape)
        {
            if (CalculateElementCount(newShape) != ElementCount)
                throw new ArgumentException("New shape must have same element count");

            var result = new ILGPUTensor<T>(_accelerator, newShape);
            result._buffer.CopyFrom(_buffer, 0, 0, ElementCount);
            return result;
        }

        /// <inheritdoc/>
        ITensor ITensor.Reshape(int[] newShape)
        {
            return Reshape(newShape);
        }

        /// <inheritdoc/>
        public ITensor<T> View(int[] start, int[] end)
        {
            // Simplified view implementation
            var newShape = new int[start.Length];
            for (int i = 0; i < start.Length; i++)
            {
                newShape[i] = end[i] - start[i];
            }
            
            return new ILGPUTensor<T>(_accelerator, newShape);
        }

        /// <inheritdoc/>
        ITensor ITensor.View(int[] start, int[] end)
        {
            return View(start, end);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!_disposed)
            {
                _buffer?.Dispose();
                _disposed = true;
            }
        }

        private static long CalculateElementCount(int[] shape)
        {
            long count = 1;
            foreach (var dim in shape)
            {
                count *= dim;
            }
            return count;
        }
    }

    /// <summary>
    /// Factory for creating ILGPU tensors.
    /// </summary>
    public class ILGPUTensorFactory : ITensorFactory
    {
        private readonly Accelerator _accelerator;

        /// <summary>
        /// Initializes a new instance of the ILGPUTensorFactory class.
        /// </summary>
        public ILGPUTensorFactory(Accelerator accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        /// <inheritdoc/>
        public ITensor Create(Type elementType, int[] shape)
        {
            if (elementType == typeof(float))
                return Create<float>(shape);
            if (elementType == typeof(double))
                return Create<double>(shape);
            if (elementType == typeof(int))
                return Create<int>(shape);

            throw new NotSupportedException($"Element type {elementType} is not supported");
        }

        /// <inheritdoc/>
        public ITensor<T> Create<T>(int[] shape) where T : unmanaged
        {
            return new ILGPUTensor<T>(_accelerator, shape);
        }

        /// <inheritdoc/>
        public ITensor<T> FromArray<T>(T[] data, int[] shape) where T : unmanaged
        {
            var tensor = Create<T>(shape);
            tensor.CopyFromCPU(data);
            return tensor;
        }

        /// <inheritdoc/>
        public ITensor<T> Zeros<T>(int[] shape) where T : unmanaged
        {
            var tensor = Create<T>(shape);
            var zeros = new T[tensor.ElementCount];
            tensor.CopyFromCPU(zeros);
            return tensor;
        }

        /// <inheritdoc/>
        public ITensor<T> Ones<T>(int[] shape) where T : unmanaged
        {
            var tensor = Create<T>(shape);
            var ones = new T[tensor.ElementCount];
            
            // Fill with ones (simplified for common types)
            if (typeof(T) == typeof(float))
            {
                var floatOnes = new float[tensor.ElementCount];
                Array.Fill(floatOnes, 1.0f);
                tensor.CopyFromCPU((T[])(object)floatOnes);
            }
            else if (typeof(T) == typeof(double))
            {
                var doubleOnes = new double[tensor.ElementCount];
                Array.Fill(doubleOnes, 1.0);
                tensor.CopyFromCPU((T[])(object)doubleOnes);
            }
            else if (typeof(T) == typeof(int))
            {
                var intOnes = new int[tensor.ElementCount];
                Array.Fill(intOnes, 1);
                tensor.CopyFromCPU((T[])(object)intOnes);
            }
            
            return tensor;
        }

        /// <inheritdoc/>
        public ITensor<T> Random<T>(int[] shape, Random random = null) where T : unmanaged
        {
            random ??= new Random();
            var tensor = Create<T>(shape);
            
            // Generate random data (simplified for common types)
            if (typeof(T) == typeof(float))
            {
                var randomData = new float[tensor.ElementCount];
                for (int i = 0; i < randomData.Length; i++)
                {
                    randomData[i] = (float)random.NextDouble();
                }
                tensor.CopyFromCPU((T[])(object)randomData);
            }
            else if (typeof(T) == typeof(double))
            {
                var randomData = new double[tensor.ElementCount];
                for (int i = 0; i < randomData.Length; i++)
                {
                    randomData[i] = random.NextDouble();
                }
                tensor.CopyFromCPU((T[])(object)randomData);
            }
            
            return tensor;
        }
    }

    /// <summary>
    /// ILGPU memory manager for tensor operations.
    /// </summary>
    public class ILGPUMemoryManager : IMemoryManager
    {
        private readonly Accelerator _accelerator;
        private readonly Dictionary<IntPtr, AllocationInfo> _allocations = new();
        private long _totalAllocated = 0;
        private bool _disposed = false;

        /// <summary>
        /// Initializes a new instance of the ILGPUMemoryManager class.
        /// </summary>
        public ILGPUMemoryManager(Accelerator accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        /// <inheritdoc/>
        public MemoryBuffer<T> Allocate<T>(long elementCount) where T : unmanaged
        {
            var buffer = _accelerator.Allocate1D<T>(elementCount);
            var sizeBytes = elementCount * Interop.SizeOf<T>();
            
            lock (_allocations)
            {
                _allocations[buffer.NativePtr] = new AllocationInfo
                {
                    SizeBytes = sizeBytes,
                    AllocatedAt = DateTime.UtcNow
                };
                _totalAllocated += sizeBytes;
            }
            
            return buffer;
        }

        /// <inheritdoc/>
        public void Deallocate<T>(MemoryBuffer<T> buffer) where T : unmanaged
        {
            lock (_allocations)
            {
                if (_allocations.TryGetValue(buffer.NativePtr, out var info))
                {
                    _allocations.Remove(buffer.NativePtr);
                    _totalAllocated -= info.SizeBytes;
                }
            }
            
            buffer.Dispose();
        }

        /// <inheritdoc/>
        public MemoryUsageInfo GetUsageInfo()
        {
            lock (_allocations)
            {
                var maxAllocation = _allocations.Count > 0 
                    ? _allocations.Values.Max(a => a.SizeBytes) 
                    : 0;

                return new MemoryUsageInfo
                {
                    TotalAllocatedBytes = _totalAllocated,
                    TotalAvailableBytes = _accelerator.MemorySize,
                    ActiveAllocations = _allocations.Count,
                    LargestAllocationBytes = maxAllocation,
                    FragmentationPercent = CalculateFragmentation()
                };
            }
        }

        /// <inheritdoc/>
        public void Copy<T>(MemoryBuffer<T> source, MemoryBuffer<T> destination, long elementCount)
            where T : unmanaged
        {
            destination.CopyFrom(source, 0, 0, elementCount);
        }

        /// <inheritdoc/>
        public bool RequiresTransfer(MemoryLocation source, MemoryLocation destination)
        {
            return source != destination;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_allocations)
                {
                    _allocations.Clear();
                    _totalAllocated = 0;
                }
                _disposed = true;
            }
        }

        private double CalculateFragmentation()
        {
            // Simplified fragmentation calculation
            if (_allocations.Count <= 1) return 0.0;
            
            var avgSize = _totalAllocated / _allocations.Count;
            var variance = _allocations.Values
                .Select(a => Math.Pow(a.SizeBytes - avgSize, 2))
                .Average();
            
            return Math.Sqrt(variance) / avgSize * 100.0;
        }

        private class AllocationInfo
        {
            public long SizeBytes { get; set; }
            public DateTime AllocatedAt { get; set; }
        }
    }
}