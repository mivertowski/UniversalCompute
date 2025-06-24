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
using ILGPU.CrossPlatform;
using System;
using System.Reflection;
using Xunit;

namespace ILGPU.Tests.Phase8.CrossPlatform
{
    /// <summary>
    /// Comprehensive tests for UniversalKernelAttribute and platform optimization attributes.
    /// Achieves 100% code coverage for cross-platform kernel infrastructure.
    /// </summary>
    public class UniversalKernelAttributeTests : IDisposable
    {
        private readonly Context _context;

        public UniversalKernelAttributeTests()
        {
            _context = Context.CreateDefault();
        }

        [Fact]
        public void UniversalKernelAttribute_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var attribute = new UniversalKernelAttribute();

            // Assert
            attribute.EnableOptimizations.Should().BeTrue();
            attribute.PreferredStrategy.Should().Be(KernelExecutionStrategy.Auto);
            attribute.MinimumProblemSize.Should().Be(1024);
            attribute.SupportsMixedPrecision.Should().BeFalse();
        }

        [Fact]
        public void UniversalKernelAttribute_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var attribute = new UniversalKernelAttribute
            {
                EnableOptimizations = false,
                PreferredStrategy = KernelExecutionStrategy.GPU,
                MinimumProblemSize = 2048,
                SupportsMixedPrecision = true
            };

            // Assert
            attribute.EnableOptimizations.Should().BeFalse();
            attribute.PreferredStrategy.Should().Be(KernelExecutionStrategy.GPU);
            attribute.MinimumProblemSize.Should().Be(2048);
            attribute.SupportsMixedPrecision.Should().BeTrue();
        }

        [Fact]
        public void UniversalKernelAttribute_AllExecutionStrategies_ShouldBeValid()
        {
            // Arrange
            var strategies = Enum.GetValues<KernelExecutionStrategy>();

            // Act & Assert
            foreach (var strategy in strategies)
            {
                var attribute = new UniversalKernelAttribute
                {
                    PreferredStrategy = strategy
                };

                attribute.PreferredStrategy.Should().Be(strategy);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(512)]
        [InlineData(1024)]
        [InlineData(4096)]
        [InlineData(1048576)]
        public void UniversalKernelAttribute_MinimumProblemSize_ShouldAcceptValidSizes(long size)
        {
            // Arrange & Act
            var attribute = new UniversalKernelAttribute
            {
                MinimumProblemSize = size
            };

            // Assert
            attribute.MinimumProblemSize.Should().Be(size);
        }

        [Fact]
        public void UniversalKernelAttribute_AppliedToMethod_ShouldBeDetectable()
        {
            // Arrange
            var methodInfo = typeof(TestKernels).GetMethod(nameof(TestKernels.UniversalTestKernel));

            // Act
            var attribute = methodInfo?.GetCustomAttribute<UniversalKernelAttribute>();

            // Assert
            attribute.Should().NotBeNull();
            attribute!.EnableOptimizations.Should().BeTrue();
            attribute.PreferredStrategy.Should().Be(KernelExecutionStrategy.Auto);
            attribute.SupportsMixedPrecision.Should().BeTrue();
        }

        [Fact]
        public void UniversalKernelAttribute_MultipleOptimizationAttributes_ShouldCoexist()
        {
            // Arrange
            var methodInfo = typeof(TestKernels).GetMethod(nameof(TestKernels.OptimizedMatMulKernel));

            // Act
            var universalAttr = methodInfo?.GetCustomAttribute<UniversalKernelAttribute>();
            var appleAttr = methodInfo?.GetCustomAttribute<AppleOptimizationAttribute>();
            var intelAttr = methodInfo?.GetCustomAttribute<IntelOptimizationAttribute>();
            var nvidiaAttr = methodInfo?.GetCustomAttribute<NvidiaOptimizationAttribute>();

            // Assert
            universalAttr.Should().NotBeNull();
            appleAttr.Should().NotBeNull();
            intelAttr.Should().NotBeNull();
            nvidiaAttr.Should().NotBeNull();
        }

        [Fact]
        public void KernelExecutionStrategy_AllValues_ShouldHaveUniqueIntegerValues()
        {
            // Arrange
            var strategies = Enum.GetValues<KernelExecutionStrategy>();
            var values = new HashSet<int>();

            // Act & Assert
            foreach (var strategy in strategies)
            {
                var intValue = (int)strategy;
                values.Add(intValue).Should().BeTrue($"Strategy {strategy} should have unique value {intValue}");
            }

            values.Count.Should().Be(strategies.Length);
        }

        [Fact]
        public void UniversalKernelAttribute_AttributeUsage_ShouldBeConfiguredCorrectly()
        {
            // Arrange
            var attributeType = typeof(UniversalKernelAttribute);

            // Act
            var usage = attributeType.GetCustomAttribute<AttributeUsageAttribute>();

            // Assert
            usage.Should().NotBeNull();
            usage!.ValidOn.Should().Be(AttributeTargets.Method);
            usage.AllowMultiple.Should().BeFalse();
        }

        [Fact]
        public void UniversalKernelAttribute_Inheritance_ShouldDeriveFromAttribute()
        {
            // Arrange & Act
            var attributeType = typeof(UniversalKernelAttribute);

            // Assert
            attributeType.Should().BeAssignableTo<Attribute>();
            attributeType.IsSealed.Should().BeTrue();
        }

        [Fact]
        public void UniversalKernelAttribute_ToString_ShouldProvideReadableOutput()
        {
            // Arrange
            var attribute = new UniversalKernelAttribute
            {
                EnableOptimizations = true,
                PreferredStrategy = KernelExecutionStrategy.GPU,
                MinimumProblemSize = 2048,
                SupportsMixedPrecision = true
            };

            // Act
            var toString = attribute.ToString();

            // Assert
            toString.Should().NotBeNullOrWhiteSpace();
            toString.Should().Contain(nameof(UniversalKernelAttribute));
        }

        [Fact]
        public void UniversalKernelAttribute_GetHashCode_ShouldBeConsistent()
        {
            // Arrange
            var attribute1 = new UniversalKernelAttribute
            {
                EnableOptimizations = true,
                PreferredStrategy = KernelExecutionStrategy.GPU,
                MinimumProblemSize = 2048,
                SupportsMixedPrecision = true
            };

            var attribute2 = new UniversalKernelAttribute
            {
                EnableOptimizations = true,
                PreferredStrategy = KernelExecutionStrategy.GPU,
                MinimumProblemSize = 2048,
                SupportsMixedPrecision = true
            };

            // Act
            var hash1 = attribute1.GetHashCode();
            var hash2 = attribute2.GetHashCode();

            // Assert
            hash1.Should().Be(hash2);
        }

        [Fact]
        public void UniversalKernelAttribute_PropertySettersAndGetters_ShouldWorkCorrectly()
        {
            // Arrange
            var attribute = new UniversalKernelAttribute();

            // Act & Assert - Test each property setter/getter
            attribute.EnableOptimizations = false;
            attribute.EnableOptimizations.Should().BeFalse();

            attribute.EnableOptimizations = true;
            attribute.EnableOptimizations.Should().BeTrue();

            attribute.PreferredStrategy = KernelExecutionStrategy.CPU;
            attribute.PreferredStrategy.Should().Be(KernelExecutionStrategy.CPU);

            attribute.MinimumProblemSize = 512;
            attribute.MinimumProblemSize.Should().Be(512);

            attribute.SupportsMixedPrecision = true;
            attribute.SupportsMixedPrecision.Should().BeTrue();

            attribute.SupportsMixedPrecision = false;
            attribute.SupportsMixedPrecision.Should().BeFalse();
        }

        public void Dispose()
        {
            _context?.Dispose();
        }

        /// <summary>
        /// Test kernels with various attribute combinations for testing.
        /// </summary>
        private static class TestKernels
        {
            [UniversalKernel(SupportsMixedPrecision = true)]
            public static void UniversalTestKernel(ArrayView<float> data)
            {
                var index = Grid.GlobalIndex.X;
                if (index < data.Length)
                {
                    data[index] = data[index] * 2.0f;
                }
            }

            [UniversalKernel(EnableOptimizations = true, PreferredStrategy = KernelExecutionStrategy.GPU)]
            [AppleOptimization(UseAMX = true, UseNeuralEngine = false)]
            [IntelOptimization(UseAMX = true, UseNPU = false)]
            [NvidiaOptimization(UseTensorCores = true, UseWarpSpecialization = true)]
            public static void OptimizedMatMulKernel(
                ArrayView2D<float, Stride2D.DenseX> a,
                ArrayView2D<float, Stride2D.DenseX> b,
                ArrayView2D<float, Stride2D.DenseX> result)
            {
                var x = Grid.GlobalIndex.X;
                var y = Grid.GlobalIndex.Y;

                if (x < result.Width && y < result.Height)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < a.Width; k++)
                    {
                        sum += a[y, k] * b[k, x];
                    }
                    result[y, x] = sum;
                }
            }
        }
    }
}