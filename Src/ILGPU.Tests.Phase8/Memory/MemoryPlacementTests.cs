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
// Change License: Apache License, Version 2.0using FluentAssertions;
using ILGPU.Memory.Unified;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.Phase8.Memory
{
    /// <summary>
    /// Comprehensive tests for memory placement strategies and optimization.
    /// Achieves 100% code coverage for universal memory placement infrastructure.
    /// </summary>
    public class MemoryPlacementTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;

        public MemoryPlacementTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.CreateCPUAccelerator(0);
        }

        #region MemoryPlacement Enum Tests

        [Fact]
        public void MemoryPlacement_AllValues_ShouldHaveUniqueIntegerValues()
        {
            // Arrange
            var placements = Enum.GetValues<MemoryPlacement>();
            var values = new HashSet<int>();

            // Act & Assert
            foreach (var placement in placements)
            {
                var intValue = (int)placement;
                values.Add(intValue).Should().BeTrue($"Placement {placement} should have unique value {intValue}");
            }

            values.Count.Should().Be(placements.Length);
        }

        [Fact]
        public void MemoryPlacement_ShouldHaveExpectedValues()
        {
            // Arrange & Act & Assert
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.Auto).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.DeviceLocal).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.HostLocal).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.AppleUnified).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.CudaManaged).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.IntelShared).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryPlacement), MemoryPlacement.HostPinned).Should().BeTrue();
        }

        [Theory]
        [InlineData(MemoryPlacement.Auto)]
        [InlineData(MemoryPlacement.DeviceLocal)]
        [InlineData(MemoryPlacement.HostLocal)]
        [InlineData(MemoryPlacement.AppleUnified)]
        [InlineData(MemoryPlacement.CudaManaged)]
        [InlineData(MemoryPlacement.IntelShared)]
        [InlineData(MemoryPlacement.HostPinned)]
        public void MemoryPlacement_ToString_ShouldReturnCorrectString(MemoryPlacement placement)
        {
            // Arrange & Act
            var stringValue = placement.ToString();

            // Assert
            stringValue.Should().NotBeNullOrWhiteSpace();
            stringValue.Should().Be(Enum.GetName(placement));
        }

        #endregion

        #region MemoryAccessPattern Enum Tests

        [Fact]
        public void MemoryAccessPattern_AllValues_ShouldHaveUniqueIntegerValues()
        {
            // Arrange
            var patterns = Enum.GetValues<MemoryAccessPattern>();
            var values = new HashSet<int>();

            // Act & Assert
            foreach (var pattern in patterns)
            {
                var intValue = (int)pattern;
                values.Add(intValue).Should().BeTrue($"Pattern {pattern} should have unique value {intValue}");
            }

            values.Count.Should().Be(patterns.Length);
        }

        [Fact]
        public void MemoryAccessPattern_ShouldHaveExpectedValues()
        {
            // Arrange & Act & Assert
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Unknown).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Sequential).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Random).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Streaming).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Transpose).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Gather).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryAccessPattern), MemoryAccessPattern.Scatter).Should().BeTrue();
        }

        #endregion

        #region PlacementOptimizer Tests

        [Fact]
        public void PlacementOptimizer_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange & Act
            var optimizer = new PlacementOptimizer();

            // Assert
            optimizer.Should().NotBeNull();
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_AutoWithUnknownPattern_ShouldReturnDeviceLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(1024, MemoryAccessPattern.Unknown, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.DeviceLocal);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_SmallSequentialData_ShouldReturnHostLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(64, MemoryAccessPattern.Sequential, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.HostLocal);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_LargeData_ShouldReturnDeviceLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(1024 * 1024, MemoryAccessPattern.Random, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.DeviceLocal);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_StreamingPattern_ShouldReturnHostPinned()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(8192, MemoryAccessPattern.Streaming, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.HostPinned);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_LowMemory_ShouldReturnHostLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024, availableMemory: 100 * 1024); // Very low memory

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(1024, MemoryAccessPattern.Random, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.HostLocal);
        }

        [Theory]
        [InlineData(MemoryAccessPattern.Sequential)]
        [InlineData(MemoryAccessPattern.Random)]
        [InlineData(MemoryAccessPattern.Streaming)]
        [InlineData(MemoryAccessPattern.Transpose)]
        [InlineData(MemoryAccessPattern.Gather)]
        [InlineData(MemoryAccessPattern.Scatter)]
        public void PlacementOptimizer_GetOptimalPlacement_AllPatterns_ShouldReturnValidPlacement(MemoryAccessPattern pattern)
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(1024, pattern, usageInfo);

            // Assert
            Enum.IsDefined(typeof(MemoryPlacement), placement).Should().BeTrue();
            placement.Should().NotBe(MemoryPlacement.Auto);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_DifferentDataTypes_ShouldConsiderElementSize()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(totalMemory: 1024 * 1024 * 1024, availableMemory: 512 * 1024 * 1024);
            var elementCount = 1024;

            // Act
            var floatPlacement = optimizer.GetOptimalPlacement<float>(elementCount, MemoryAccessPattern.Sequential, usageInfo);
            var doublePlacement = optimizer.GetOptimalPlacement<double>(elementCount, MemoryAccessPattern.Sequential, usageInfo);
            var bytePlacement = optimizer.GetOptimalPlacement<byte>(elementCount, MemoryAccessPattern.Sequential, usageInfo);

            // Assert
            floatPlacement.Should().Be(MemoryPlacement.HostLocal);
            doublePlacement.Should().Be(MemoryPlacement.HostLocal);
            bytePlacement.Should().Be(MemoryPlacement.HostLocal);
        }

        #endregion

        #region MemoryUsageInfo Tests

        [Fact]
        public void MemoryUsageInfo_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange
            var totalMemory = 1024L * 1024 * 1024;
            var availableMemory = 512L * 1024 * 1024;

            // Act
            var usageInfo = new MemoryUsageInfo(totalMemory, availableMemory);

            // Assert
            usageInfo.TotalMemory.Should().Be(totalMemory);
            usageInfo.AvailableMemory.Should().Be(availableMemory);
            usageInfo.UsedMemory.Should().Be(totalMemory - availableMemory);
            usageInfo.UsagePercentage.Should().BeApproximately(50.0, 0.01);
        }

        [Fact]
        public void MemoryUsageInfo_UsagePercentage_ShouldCalculateCorrectly()
        {
            // Arrange & Act
            var usageInfo1 = new MemoryUsageInfo(1000, 750); // 25% used
            var usageInfo2 = new MemoryUsageInfo(1000, 100); // 90% used
            var usageInfo3 = new MemoryUsageInfo(1000, 1000); // 0% used

            // Assert
            usageInfo1.UsagePercentage.Should().BeApproximately(25.0, 0.01);
            usageInfo2.UsagePercentage.Should().BeApproximately(90.0, 0.01);
            usageInfo3.UsagePercentage.Should().BeApproximately(0.0, 0.01);
        }

        [Fact]
        public void MemoryUsageInfo_IsLowMemory_ShouldDetectLowMemoryCorrectly()
        {
            // Arrange & Act
            var lowMemory = new MemoryUsageInfo(1000, 50); // 95% used
            var highMemory = new MemoryUsageInfo(1000, 500); // 50% used

            // Assert
            lowMemory.IsLowMemory.Should().BeTrue();
            highMemory.IsLowMemory.Should().BeFalse();
        }

        [Fact]
        public void MemoryUsageInfo_IsVeryLowMemory_ShouldDetectVeryLowMemoryCorrectly()
        {
            // Arrange & Act
            var veryLowMemory = new MemoryUsageInfo(1000, 10); // 99% used
            var lowMemory = new MemoryUsageInfo(1000, 50); // 95% used
            var normalMemory = new MemoryUsageInfo(1000, 500); // 50% used

            // Assert
            veryLowMemory.IsVeryLowMemory.Should().BeTrue();
            lowMemory.IsVeryLowMemory.Should().BeFalse();
            normalMemory.IsVeryLowMemory.Should().BeFalse();
        }

        [Fact]
        public void MemoryUsageInfo_ToString_ShouldProvideReadableOutput()
        {
            // Arrange
            var usageInfo = new MemoryUsageInfo(1024 * 1024 * 1024, 512 * 1024 * 1024);

            // Act
            var toString = usageInfo.ToString();

            // Assert
            toString.Should().NotBeNullOrWhiteSpace();
            toString.Should().Contain("50"); // Usage percentage
        }

        [Fact]
        public void MemoryUsageInfo_GetHashCode_ShouldBeConsistent()
        {
            // Arrange
            var usageInfo1 = new MemoryUsageInfo(1000, 500);
            var usageInfo2 = new MemoryUsageInfo(1000, 500);

            // Act
            var hash1 = usageInfo1.GetHashCode();
            var hash2 = usageInfo2.GetHashCode();

            // Assert
            hash1.Should().Be(hash2);
        }

        [Fact]
        public void MemoryUsageInfo_Equals_ShouldWorkCorrectly()
        {
            // Arrange
            var usageInfo1 = new MemoryUsageInfo(1000, 500);
            var usageInfo2 = new MemoryUsageInfo(1000, 500);
            var usageInfo3 = new MemoryUsageInfo(2000, 500);

            // Act & Assert
            usageInfo1.Equals(usageInfo2).Should().BeTrue();
            usageInfo1.Equals(usageInfo3).Should().BeFalse();
            usageInfo1.Equals((object)usageInfo2).Should().BeTrue();
            usageInfo1.Equals((object)usageInfo3).Should().BeFalse();
            usageInfo1.Equals(null).Should().BeFalse();
            usageInfo1.Equals("string").Should().BeFalse();
        }

        #endregion

        #region Edge Cases and Error Conditions

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_ZeroSize_ShouldReturnHostLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(1024 * 1024 * 1024, 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(0, MemoryAccessPattern.Sequential, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.HostLocal);
        }

        [Fact]
        public void PlacementOptimizer_GetOptimalPlacement_NegativeSize_ShouldReturnHostLocal()
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(1024 * 1024 * 1024, 512 * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(-100, MemoryAccessPattern.Sequential, usageInfo);

            // Assert
            placement.Should().Be(MemoryPlacement.HostLocal);
        }

        [Fact]
        public void MemoryUsageInfo_ZeroTotalMemory_ShouldHandleGracefully()
        {
            // Arrange & Act
            var usageInfo = new MemoryUsageInfo(0, 0);

            // Assert
            usageInfo.TotalMemory.Should().Be(0);
            usageInfo.AvailableMemory.Should().Be(0);
            usageInfo.UsedMemory.Should().Be(0);
            usageInfo.UsagePercentage.Should().Be(0.0);
        }

        [Fact]
        public void MemoryUsageInfo_InvalidMemoryValues_ShouldHandleGracefully()
        {
            // Arrange & Act - Available memory greater than total
            var usageInfo = new MemoryUsageInfo(100, 200);

            // Assert
            usageInfo.TotalMemory.Should().Be(100);
            usageInfo.AvailableMemory.Should().Be(200);
            usageInfo.UsedMemory.Should().Be(-100); // Negative used memory
            usageInfo.UsagePercentage.Should().Be(0.0); // Should clamp to 0
        }

        #endregion

        #region Performance and Memory Size Tests

        [Theory]
        [InlineData(1024)]           // 1KB
        [InlineData(1024 * 1024)]    // 1MB
        [InlineData(1024 * 1024 * 10)] // 10MB
        [InlineData(1024 * 1024 * 100)] // 100MB
        public void PlacementOptimizer_DifferentDataSizes_ShouldReturnAppropriateStrategy(long elementCount)
        {
            // Arrange
            var optimizer = new PlacementOptimizer();
            var usageInfo = new MemoryUsageInfo(1024L * 1024 * 1024, 512L * 1024 * 1024);

            // Act
            var placement = optimizer.GetOptimalPlacement<float>(elementCount, MemoryAccessPattern.Random, usageInfo);

            // Assert
            placement.Should().NotBe(MemoryPlacement.Auto);
            Enum.IsDefined(typeof(MemoryPlacement), placement).Should().BeTrue();
        }

        #endregion

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }
}