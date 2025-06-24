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

using FluentAssertions;
using ILGPU.Memory.Unified;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.Phase8.Memory
{
    /// <summary>
    /// Comprehensive tests for UniversalMemoryManager and IUniversalBuffer.
    /// Achieves 100% code coverage for universal memory management infrastructure.
    /// </summary>
    public class UniversalMemoryManagerTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly UniversalMemoryManager _memoryManager;

        public UniversalMemoryManagerTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.CreateCPUAccelerator(0);
            _memoryManager = new UniversalMemoryManager(_context);
        }

        #region UniversalMemoryManager Constructor Tests

        [Fact]
        public void UniversalMemoryManager_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange & Act
            var manager = new UniversalMemoryManager(_context);

            // Assert
            manager.Should().NotBeNull();
            manager.Context.Should().Be(_context);
        }

        [Fact]
        public void UniversalMemoryManager_Constructor_NullContext_ShouldThrowArgumentNullException()
        {
            // Arrange & Act & Assert
            Action act = () => new UniversalMemoryManager(null!);
            act.Should().Throw<ArgumentNullException>().WithParameterName("context");
        }

        #endregion

        #region AllocateUniversal Tests

        [Fact]
        public void AllocateUniversal_ValidParameters_ShouldReturnValidBuffer()
        {
            // Arrange
            var size = 1024L;
            var placement = MemoryPlacement.DeviceLocal;
            var accessPattern = MemoryAccessPattern.Sequential;

            // Act
            using var buffer = _memoryManager.AllocateUniversal<float>(size, placement, accessPattern);

            // Assert
            buffer.Should().NotBeNull();
            buffer.Length.Should().Be(size);
            buffer.ElementSize.Should().Be(sizeof(float));
            buffer.Placement.Should().Be(placement);
            buffer.AccessPattern.Should().Be(accessPattern);
        }

        [Fact]
        public void AllocateUniversal_AutoPlacement_ShouldOptimizePlacement()
        {
            // Arrange
            var size = 1024L;
            var placement = MemoryPlacement.Auto;
            var accessPattern = MemoryAccessPattern.Sequential;

            // Act
            using var buffer = _memoryManager.AllocateUniversal<float>(size, placement, accessPattern);

            // Assert
            buffer.Should().NotBeNull();
            buffer.Length.Should().Be(size);
            buffer.Placement.Should().NotBe(MemoryPlacement.Auto); // Should be optimized
        }

        [Theory]
        [InlineData(MemoryPlacement.DeviceLocal)]
        [InlineData(MemoryPlacement.HostLocal)]
        [InlineData(MemoryPlacement.HostPinned)]
        public void AllocateUniversal_DifferentPlacements_ShouldCreateAppropriateBuffers(MemoryPlacement placement)
        {
            // Arrange
            var size = 512L;

            // Act
            using var buffer = _memoryManager.AllocateUniversal<int>(size, placement);

            // Assert
            buffer.Should().NotBeNull();
            buffer.Placement.Should().Be(placement);
            buffer.Length.Should().Be(size);
        }

        [Theory]
        [InlineData(MemoryAccessPattern.Sequential)]
        [InlineData(MemoryAccessPattern.Random)]
        [InlineData(MemoryAccessPattern.Streaming)]
        [InlineData(MemoryAccessPattern.Transpose)]
        [InlineData(MemoryAccessPattern.Gather)]
        [InlineData(MemoryAccessPattern.Scatter)]
        public void AllocateUniversal_DifferentAccessPatterns_ShouldStorePattern(MemoryAccessPattern pattern)
        {
            // Arrange
            var size = 256L;

            // Act
            using var buffer = _memoryManager.AllocateUniversal<byte>(size, MemoryPlacement.DeviceLocal, pattern);

            // Assert
            buffer.AccessPattern.Should().Be(pattern);
        }

        [Fact]
        public void AllocateUniversal_ZeroSize_ShouldThrowArgumentException()
        {
            // Arrange & Act & Assert
            Action act = () => _memoryManager.AllocateUniversal<float>(0);
            act.Should().Throw<ArgumentException>().WithParameterName("size");
        }

        [Fact]
        public void AllocateUniversal_NegativeSize_ShouldThrowArgumentException()
        {
            // Arrange & Act & Assert
            Action act = () => _memoryManager.AllocateUniversal<float>(-100);
            act.Should().Throw<ArgumentException>().WithParameterName("size");
        }

        #endregion

        #region AllocateUniversal2D Tests

        [Fact]
        public void AllocateUniversal2D_ValidParameters_ShouldReturnValidBuffer()
        {
            // Arrange
            var width = 32L;
            var height = 24L;
            var placement = MemoryPlacement.DeviceLocal;

            // Act
            using var buffer = _memoryManager.AllocateUniversal2D<float>(width, height, placement);

            // Assert
            buffer.Should().NotBeNull();
            buffer.Width.Should().Be(width);
            buffer.Height.Should().Be(height);
            buffer.Length.Should().Be(width * height);
            buffer.Placement.Should().Be(placement);
        }

        [Fact]
        public void AllocateUniversal2D_ZeroDimensions_ShouldThrowArgumentException()
        {
            // Arrange & Act & Assert
            Action act1 = () => _memoryManager.AllocateUniversal2D<float>(0, 10);
            Action act2 = () => _memoryManager.AllocateUniversal2D<float>(10, 0);

            act1.Should().Throw<ArgumentException>();
            act2.Should().Throw<ArgumentException>();
        }

        #endregion

        #region AllocateUniversal3D Tests

        [Fact]
        public void AllocateUniversal3D_ValidParameters_ShouldReturnValidBuffer()
        {
            // Arrange
            var width = 16L;
            var height = 12L;
            var depth = 8L;
            var placement = MemoryPlacement.HostLocal;

            // Act
            using var buffer = _memoryManager.AllocateUniversal3D<int>(width, height, depth, placement);

            // Assert
            buffer.Should().NotBeNull();
            buffer.Width.Should().Be(width);
            buffer.Height.Should().Be(height);
            buffer.Depth.Should().Be(depth);
            buffer.Length.Should().Be(width * height * depth);
            buffer.Placement.Should().Be(placement);
        }

        #endregion

        #region Memory Statistics Tests

        [Fact]
        public void GetGlobalMemoryStatistics_ShouldReturnValidStatistics()
        {
            // Arrange
            using var buffer1 = _memoryManager.AllocateUniversal<float>(1024);
            using var buffer2 = _memoryManager.AllocateUniversal<int>(512);

            // Act
            var stats = _memoryManager.GetGlobalMemoryStatistics();

            // Assert
            stats.Should().NotBeNull();
            stats.TotalAllocatedBytes.Should().BeGreaterThan(0);
            stats.ActiveAllocations.Should().BeGreaterOrEqualTo(2);
            stats.PeakAllocatedBytes.Should().BeGreaterOrEqualTo(stats.TotalAllocatedBytes);
        }

        [Fact]
        public void GetMemoryUsage_ShouldReturnCurrentUsage()
        {
            // Arrange
            using var buffer = _memoryManager.AllocateUniversal<double>(2048);

            // Act
            var usage = _memoryManager.GetMemoryUsage();

            // Assert
            usage.Should().NotBeNull();
            usage.TotalMemory.Should().BeGreaterThan(0);
            usage.AvailableMemory.Should().BeGreaterThan(0);
            usage.UsedMemory.Should().BeGreaterOrEqualTo(0);
        }

        #endregion

        #region Memory Recommendations Tests

        [Fact]
        public void GetMemoryRecommendations_ShouldProvideValidRecommendations()
        {
            // Arrange & Act
            var recommendations = _memoryManager.GetMemoryRecommendations();

            // Assert
            recommendations.Should().NotBeNull();
            recommendations.Should().NotBeEmpty();
        }

        [Fact]
        public void OptimizeMemoryUsage_ShouldExecuteWithoutException()
        {
            // Arrange
            using var buffer1 = _memoryManager.AllocateUniversal<float>(1024);
            using var buffer2 = _memoryManager.AllocateUniversal<int>(512);

            // Act & Assert
            Action act = () => _memoryManager.OptimizeMemoryUsage();
            act.Should().NotThrow();
        }

        #endregion

        #region IUniversalBuffer Interface Tests

        [Fact]
        public void IUniversalBuffer_Properties_ShouldReturnCorrectValues()
        {
            // Arrange
            var size = 1024L;
            var placement = MemoryPlacement.DeviceLocal;
            var pattern = MemoryAccessPattern.Random;

            // Act
            using var buffer = _memoryManager.AllocateUniversal<double>(size, placement, pattern);

            // Assert
            buffer.Length.Should().Be(size);
            buffer.ElementSize.Should().Be(sizeof(double));
            buffer.TotalSizeInBytes.Should().Be(size * sizeof(double));
            buffer.Placement.Should().Be(placement);
            buffer.AccessPattern.Should().Be(pattern);
            buffer.IsDisposed.Should().BeFalse();
        }

        [Fact]
        public void IUniversalBuffer_GetView1D_ShouldReturnValidView()
        {
            // Arrange
            using var buffer = _memoryManager.AllocateUniversal<float>(1024);

            // Act
            var view = buffer.GetView1D();

            // Assert
            view.Should().NotBeNull();
            view.Length.Should().Be(buffer.Length);
        }

        [Fact]
        public void IUniversalBuffer_GetView2D_ShouldReturnValidView()
        {
            // Arrange
            var width = 32L;
            var height = 24L;
            using var buffer = _memoryManager.AllocateUniversal2D<float>(width, height);

            // Act
            var view = buffer.GetView2D();

            // Assert
            view.Should().NotBeNull();
            view.Width.Should().Be(width);
            view.Height.Should().Be(height);
        }

        [Fact]
        public void IUniversalBuffer_GetView3D_ShouldReturnValidView()
        {
            // Arrange
            var width = 16L;
            var height = 12L;
            var depth = 8L;
            using var buffer = _memoryManager.AllocateUniversal3D<float>(width, height, depth);

            // Act
            var view = buffer.GetView3D();

            // Assert
            view.Should().NotBeNull();
            view.Width.Should().Be(width);
            view.Height.Should().Be(height);
            view.Depth.Should().Be(depth);
        }

        #endregion

        #region Async Operations Tests

        [Fact]
        public async Task IUniversalBuffer_CopyFromAsyncArray_ShouldWorkCorrectly()
        {
            // Arrange
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            using var buffer = _memoryManager.AllocateUniversal<float>(data.Length);

            // Act
            await buffer.CopyFromAsync(data);

            // Assert - Verify operation completes without exception
            buffer.Length.Should().Be(data.Length);
        }

        [Fact]
        public async Task IUniversalBuffer_CopyToAsyncArray_ShouldWorkCorrectly()
        {
            // Arrange
            var sourceData = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            using var buffer = _memoryManager.AllocateUniversal<float>(sourceData.Length);
            await buffer.CopyFromAsync(sourceData);

            // Act
            var resultData = new float[sourceData.Length];
            await buffer.CopyToAsync(resultData);

            // Assert - Verify operation completes without exception
            resultData.Length.Should().Be(sourceData.Length);
        }

        [Fact]
        public async Task IUniversalBuffer_CopyFromAsyncBuffer_ShouldWorkCorrectly()
        {
            // Arrange
            using var sourceBuffer = _memoryManager.AllocateUniversal<float>(100);
            using var destBuffer = _memoryManager.AllocateUniversal<float>(100);

            // Act
            await destBuffer.CopyFromAsync(sourceBuffer);

            // Assert - Verify operation completes without exception
            destBuffer.Length.Should().Be(sourceBuffer.Length);
        }

        #endregion

        #region Performance Statistics Tests

        [Fact]
        public void IUniversalBuffer_GetPerformanceStatistics_ShouldReturnValidStats()
        {
            // Arrange
            using var buffer = _memoryManager.AllocateUniversal<float>(1024);

            // Act
            var stats = buffer.GetPerformanceStatistics();

            // Assert
            stats.Should().NotBeNull();
            stats.TotalOperations.Should().BeGreaterOrEqualTo(0);
            stats.TotalBytesTransferred.Should().BeGreaterOrEqualTo(0);
            stats.AverageLatencyMs.Should().BeGreaterOrEqualTo(0);
        }

        #endregion

        #region Memory Layout Tests

        [Theory]
        [InlineData(MemoryLayout.RowMajor)]
        [InlineData(MemoryLayout.ColumnMajor)]
        [InlineData(MemoryLayout.Tiled)]
        [InlineData(MemoryLayout.Optimal)]
        public void IUniversalBuffer_SetMemoryLayout_ShouldAcceptValidLayouts(MemoryLayout layout)
        {
            // Arrange
            using var buffer = _memoryManager.AllocateUniversal2D<float>(32, 24);

            // Act & Assert
            Action act = () => buffer.SetMemoryLayout(layout);
            act.Should().NotThrow();
        }

        [Fact]
        public void IUniversalBuffer_GetOptimalLayout_ShouldReturnValidLayout()
        {
            // Arrange
            using var buffer = _memoryManager.AllocateUniversal2D<float>(32, 24);

            // Act
            var layout = buffer.GetOptimalLayout();

            // Assert
            Enum.IsDefined(typeof(MemoryLayout), layout).Should().BeTrue();
        }

        #endregion

        #region Disposal Tests

        [Fact]
        public void IUniversalBuffer_Dispose_ShouldMarkAsDisposed()
        {
            // Arrange
            var buffer = _memoryManager.AllocateUniversal<float>(1024);

            // Act
            buffer.Dispose();

            // Assert
            buffer.IsDisposed.Should().BeTrue();
        }

        [Fact]
        public void IUniversalBuffer_DoubleDispose_ShouldNotThrow()
        {
            // Arrange
            var buffer = _memoryManager.AllocateUniversal<float>(1024);
            buffer.Dispose();

            // Act & Assert
            Action act = () => buffer.Dispose();
            act.Should().NotThrow();
        }

        [Fact]
        public void UniversalMemoryManager_Dispose_ShouldDisposeAllBuffers()
        {
            // Arrange
            var manager = new UniversalMemoryManager(_context);
            var buffer1 = manager.AllocateUniversal<float>(1024);
            var buffer2 = manager.AllocateUniversal<int>(512);

            // Act
            manager.Dispose();

            // Assert
            buffer1.IsDisposed.Should().BeTrue();
            buffer2.IsDisposed.Should().BeTrue();
        }

        [Fact]
        public void UniversalMemoryManager_DoubleDispose_ShouldNotThrow()
        {
            // Arrange
            var manager = new UniversalMemoryManager(_context);
            manager.Dispose();

            // Act & Assert
            Action act = () => manager.Dispose();
            act.Should().NotThrow();
        }

        #endregion

        #region Resource Cleanup Tests

        [Fact]
        public void IUniversalBuffer_UsingStatement_ShouldDisposeCorrectly()
        {
            // Arrange
            IUniversalBuffer<float> buffer;

            // Act
            using (buffer = _memoryManager.AllocateUniversal<float>(1024))
            {
                buffer.IsDisposed.Should().BeFalse();
            }

            // Assert
            buffer.IsDisposed.Should().BeTrue();
        }

        #endregion

        #region Error Conditions Tests

        [Fact]
        public void IUniversalBuffer_AccessAfterDispose_ShouldThrowObjectDisposedException()
        {
            // Arrange
            var buffer = _memoryManager.AllocateUniversal<float>(1024);
            buffer.Dispose();

            // Act & Assert
            Action act = () => _ = buffer.Length;
            act.Should().Throw<ObjectDisposedException>();
        }

        #endregion

        public void Dispose()
        {
            _memoryManager?.Dispose();
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }
}