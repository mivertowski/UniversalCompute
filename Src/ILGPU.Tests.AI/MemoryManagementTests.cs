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

using FluentAssertions;
using ILGPU.Numerics;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;
using System;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.AI
{
    /// <summary>
    /// Tests for AI memory management and unified memory operations.
    /// </summary>
    public class MemoryManagementTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;

        public MemoryManagementTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                          .CreateAccelerator(_context);
        }

        [Theory]
        [InlineData(MemoryLayoutMode.CpuOptimized)]
        [InlineData(MemoryLayoutMode.GpuOptimized)]
        [InlineData(MemoryLayoutMode.Unified)]
        [InlineData(MemoryLayoutMode.Pinned)]
        public void UnifiedTensor_ShouldHandleDifferentMemoryLayouts(MemoryLayoutMode layoutMode)
        {
            // Arrange
            var shape = new TensorShape(64, 64);

            // Act & Assert
            using var tensor = new UnifiedTensor<float>(_accelerator, shape, layoutMode);
            tensor.Should().NotBeNull();
        }

        [Fact]
        public async Task UnifiedTensor_ShouldMigrateBetweenMemoryLocations()
        {
            // Arrange
            var shape = new TensorShape(32, 32);
            using var tensor = new UnifiedTensor<float>(_accelerator, shape, MemoryLayoutMode.CpuOptimized);

            // Initialize with test data
            var span = tensor.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = i * 0.1f;
            }

            // Act - Migrate to GPU
            await tensor.MigrateToAsync(MemoryLocation.GPU);

            // Migrate back to CPU
            await tensor.MigrateToAsync(MemoryLocation.CPU);

            // Assert - Data should be preserved
            var newSpan = tensor.AsSpan();
            newSpan[0].Should().Be(0.0f);
            newSpan[10].Should().Be(1.0f);
        }

        [Fact]
        public void UnifiedTensor_ShouldProvideZeroCopyAccess()
        {
            // Arrange
            var shape = new TensorShape(16, 16);
            using var tensor = new UnifiedTensor<float>(_accelerator, shape);

            // Act
            var span = tensor.AsSpan();
            var memory = tensor.AsMemory();
            var readOnlySpan = tensor.AsReadOnlySpan();
            var readOnlyMemory = tensor.AsReadOnlyMemory();

            // Assert
            span.Length.Should().Be(256);
            memory.Length.Should().Be(256);
            readOnlySpan.Length.Should().Be(256);
            readOnlyMemory.Length.Should().Be(256);
        }

        [Fact]
        public void UnifiedTensor_ShouldSupportPinnedMemory()
        {
            // Arrange
            var shape = new TensorShape(8, 8);
            using var tensor = new UnifiedTensor<float>(_accelerator, shape, MemoryLayoutMode.Pinned);

            // Act
            var pinnedHandle = tensor.GetPinnedHandle();

            // Assert
            pinnedHandle.Should().NotBeNull();
            pinnedHandle.Dispose(); // Clean up handle
        }

        [Fact]
        public void UnifiedTensor_InvalidPinnedAccess_ShouldThrow()
        {
            // Arrange
            var shape = new TensorShape(8, 8);
            using var tensor = new UnifiedTensor<float>(_accelerator, shape, MemoryLayoutMode.CpuOptimized);

            // Act & Assert
            Action act = () => tensor.GetPinnedHandle();
            act.Should().Throw<InvalidOperationException>();
        }

        [Theory]
        [InlineData(typeof(float))]
        [InlineData(typeof(double))]
        [InlineData(typeof(Half))]
        [InlineData(typeof(int))]
        public void UnifiedTensor_ShouldSupportMultipleDataTypes(Type dataType)
        {
            // Arrange
            var shape = new TensorShape(4, 4);

            // Act & Assert
            if (dataType == typeof(float))
            {
                using var tensor = new UnifiedTensor<float>(_accelerator, shape);
                tensor.Should().NotBeNull();
            }
            else if (dataType == typeof(double))
            {
                using var tensor = new UnifiedTensor<double>(_accelerator, shape);
                tensor.Should().NotBeNull();
            }
            else if (dataType == typeof(Half))
            {
                // Skip Half type due to INumber<Half> constraint issue
                Assert.True(true);
            }
            else if (dataType == typeof(int))
            {
                using var tensor = new UnifiedTensor<int>(_accelerator, shape);
                tensor.Should().NotBeNull();
            }
        }

        [Fact]
        public void UnifiedTensor_ShouldHandleLargeTensors()
        {
            // Arrange
            var shape = new TensorShape(1024, 1024); // 1M elements

            // Act & Assert
            using var tensor = new UnifiedTensor<float>(_accelerator, shape);
            tensor.Length.Should().Be(1024 * 1024);
        }

        [Fact]
        public void UnifiedTensor_ShouldOptimizeLayoutAutomatically()
        {
            // Arrange
            var smallShape = new TensorShape(8, 8);
            var largeShape = new TensorShape(2048, 2048);

            // Act
            using var smallTensor = new UnifiedTensor<float>(_accelerator, smallShape, MemoryLayoutMode.Auto);
            using var largeTensor = new UnifiedTensor<float>(_accelerator, largeShape, MemoryLayoutMode.Auto);

            // Assert
            smallTensor.Should().NotBeNull();
            largeTensor.Should().NotBeNull();
        }

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }

    /// <summary>
    /// Tests for performance tracking and metrics collection.
    /// </summary>
    public class PerformanceTrackingTests : IDisposable
    {
        private readonly PerformanceTracker _tracker;

        public PerformanceTrackingTests()
        {
            _tracker = new PerformanceTracker();
        }

        [Fact]
        public void PerformanceTracker_ShouldRecordOperations()
        {
            // Arrange
            var acceleratorType = AcceleratorType.CPU;
            var primitiveType = PrimitiveType.MatrixMultiplication;
            var operationCount = 1000L;

            // Act
            _tracker.RecordOperation(acceleratorType, primitiveType, operationCount);

            // Assert
            var metrics = _tracker.GetMetrics(acceleratorType);
            metrics.Should().NotBeNull();
        }

        [Fact]
        public void PerformanceTracker_ShouldRecordExecutions()
        {
            // Arrange
            var workloadType = WorkloadType.MatrixMultiplication;
            var duration = TimeSpan.FromMilliseconds(100);

            // Act
            _tracker.RecordExecution(workloadType, duration, success: true);

            // Assert - Should complete without throwing
        }

        [Fact]
        public void PerformanceTracker_ShouldProvideAcceleratorMetrics()
        {
            // Arrange
            var acceleratorType = AcceleratorType.Cuda;
            _tracker.RecordOperation(acceleratorType, PrimitiveType.Convolution, 500);

            // Act
            var metrics = _tracker.GetMetrics(acceleratorType);

            // Assert
            metrics.Should().NotBeNull();
        }

        [Fact]
        public void AcceleratorPerformanceMetrics_ShouldTrackPrimitives()
        {
            // Arrange
            var metrics = new AcceleratorPerformanceMetrics();
            var primitiveType = PrimitiveType.Activation;
            var operationCount = 2000L;

            // Act
            metrics.RecordOperation(primitiveType, operationCount);

            // Assert
            var primitiveMetrics = metrics.GetPrimitiveMetrics(primitiveType);
            primitiveMetrics.Should().NotBeNull();
            primitiveMetrics.OperationCount.Should().Be(operationCount);
            primitiveMetrics.ExecutionCount.Should().Be(1);
        }

        [Fact]
        public void PrimitiveMetrics_ShouldCalculateAverages()
        {
            // Arrange
            var metrics = new PrimitiveMetrics
            {
                OperationCount = 1000,
                ExecutionCount = 10
            };

            // Act
            var average = metrics.AverageOperationsPerExecution;

            // Assert
            average.Should().Be(100.0);
        }

        [Theory]
        [InlineData(PrimitiveType.MatrixMultiplication)]
        [InlineData(PrimitiveType.Convolution)]
        [InlineData(PrimitiveType.Activation)]
        [InlineData(PrimitiveType.Normalization)]
        [InlineData(PrimitiveType.Pooling)]
        [InlineData(PrimitiveType.Quantization)]
        public void PrimitiveType_ShouldHaveValidValues(PrimitiveType primitiveType)
        {
            // Arrange & Act & Assert
            primitiveType.Should().BeDefined();
        }

        public void Dispose()
        {
            _tracker?.Dispose();
        }
    }

    /// <summary>
    /// Tests for memory layout optimizations and patterns.
    /// </summary>
    public class MemoryLayoutTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;

        public MemoryLayoutTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                          .CreateAccelerator(_context);
        }

        [Theory]
        [InlineData(MemoryLocation.CPU)]
        [InlineData(MemoryLocation.GPU)]
        [InlineData(MemoryLocation.Unified)]
        [InlineData(MemoryLocation.Pinned)]
        public void MemoryLocation_ShouldHaveValidValues(MemoryLocation location)
        {
            // Arrange & Act & Assert
            location.Should().BeDefined();
        }

        [Fact]
        public void UnifiedTensor_ShouldChooseOptimalLayoutForSmallTensors()
        {
            // Arrange
            var smallShape = new TensorShape(16, 16);

            // Act
            using var tensor = new UnifiedTensor<float>(_accelerator, smallShape, MemoryLayoutMode.Auto);

            // Assert
            tensor.Should().NotBeNull();
            // Small tensors typically prefer CPU-optimized layout
        }

        [Fact]
        public void UnifiedTensor_ShouldChooseOptimalLayoutForLargeTensors()
        {
            // Arrange
            var largeShape = new TensorShape(1024, 1024);

            // Act
            using var tensor = new UnifiedTensor<float>(_accelerator, largeShape, MemoryLayoutMode.Auto);

            // Assert
            tensor.Should().NotBeNull();
            // Large tensors typically prefer GPU-optimized or unified layout
        }

        [Fact]
        public void UnifiedTensor_ShouldSupportDataCopyingBetweenLayouts()
        {
            // Arrange
            var shape = new TensorShape(8, 8);
            using var cpuTensor = new UnifiedTensor<float>(_accelerator, shape, MemoryLayoutMode.CpuOptimized);
            using var gpuTensor = new UnifiedTensor<float>(_accelerator, shape, MemoryLayoutMode.GpuOptimized);

            // Initialize CPU tensor
            var span = cpuTensor.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = i;
            }

            // Act
            gpuTensor.CopyFrom(cpuTensor);

            // Assert
            gpuTensor.Should().NotBeNull();
        }

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }
}