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
    /// Comprehensive tests for platform-specific optimization attributes.
    /// Achieves 100% code coverage for platform optimization infrastructure.
    /// </summary>
    public class PlatformOptimizationAttributeTests : IDisposable
    {
        private readonly Context _context;

        public PlatformOptimizationAttributeTests()
        {
            _context = Context.CreateDefault();
        }

        #region AppleOptimizationAttribute Tests

        [Fact]
        public void AppleOptimizationAttribute_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var attribute = new AppleOptimizationAttribute();

            // Assert
            attribute.UseAMX.Should().BeFalse();
            attribute.UseNeuralEngine.Should().BeTrue();
            attribute.UseMetalPerformanceShaders.Should().BeTrue();
            attribute.OptimizeForLatency.Should().BeFalse();
            attribute.UseTiledMemory.Should().BeTrue();
            attribute.Priority.Should().Be(OptimizationPriority.Balanced);
        }

        [Fact]
        public void AppleOptimizationAttribute_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var attribute = new AppleOptimizationAttribute
            {
                UseAMX = true,
                UseNeuralEngine = false,
                UseMetalPerformanceShaders = false,
                OptimizeForLatency = true,
                UseTiledMemory = false,
                Priority = OptimizationPriority.Performance
            };

            // Assert
            attribute.UseAMX.Should().BeTrue();
            attribute.UseNeuralEngine.Should().BeFalse();
            attribute.UseMetalPerformanceShaders.Should().BeFalse();
            attribute.OptimizeForLatency.Should().BeTrue();
            attribute.UseTiledMemory.Should().BeFalse();
            attribute.Priority.Should().Be(OptimizationPriority.Performance);
        }

        [Fact]
        public void AppleOptimizationAttribute_InheritsFromPlatformOptimizationAttribute()
        {
            // Arrange & Act
            var attribute = new AppleOptimizationAttribute();

            // Assert
            attribute.Should().BeAssignableTo<PlatformOptimizationAttribute>();
            attribute.Platform.Should().Be("Apple");
        }

        #endregion

        #region IntelOptimizationAttribute Tests

        [Fact]
        public void IntelOptimizationAttribute_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var attribute = new IntelOptimizationAttribute();

            // Assert
            attribute.UseAMX.Should().BeFalse();
            attribute.UseAVX512.Should().BeTrue();
            attribute.UseMKL.Should().BeTrue();
            attribute.UseDLBoost.Should().BeFalse();
            attribute.UseNPU.Should().BeFalse();
            attribute.OptimizeForThroughput.Should().BeTrue();
            attribute.Priority.Should().Be(OptimizationPriority.Balanced);
        }

        [Fact]
        public void IntelOptimizationAttribute_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var attribute = new IntelOptimizationAttribute
            {
                UseAMX = true,
                UseAVX512 = false,
                UseMKL = false,
                UseDLBoost = true,
                UseNPU = true,
                OptimizeForThroughput = false,
                Priority = OptimizationPriority.EnergyEfficiency
            };

            // Assert
            attribute.UseAMX.Should().BeTrue();
            attribute.UseAVX512.Should().BeFalse();
            attribute.UseMKL.Should().BeFalse();
            attribute.UseDLBoost.Should().BeTrue();
            attribute.UseNPU.Should().BeTrue();
            attribute.OptimizeForThroughput.Should().BeFalse();
            attribute.Priority.Should().Be(OptimizationPriority.EnergyEfficiency);
        }

        [Fact]
        public void IntelOptimizationAttribute_InheritsFromPlatformOptimizationAttribute()
        {
            // Arrange & Act
            var attribute = new IntelOptimizationAttribute();

            // Assert
            attribute.Should().BeAssignableTo<PlatformOptimizationAttribute>();
            attribute.Platform.Should().Be("Intel");
        }

        #endregion

        #region NvidiaOptimizationAttribute Tests

        [Fact]
        public void NvidiaOptimizationAttribute_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var attribute = new NvidiaOptimizationAttribute();

            // Assert
            attribute.UseTensorCores.Should().BeTrue();
            attribute.UseCuBLAS.Should().BeTrue();
            attribute.UseCuDNN.Should().BeTrue();
            attribute.UseWarpSpecialization.Should().BeTrue();
            attribute.OptimizePTXGeneration.Should().BeTrue();
            attribute.ComputeCapability.Should().Be("Auto");
            attribute.Priority.Should().Be(OptimizationPriority.Balanced);
        }

        [Fact]
        public void NvidiaOptimizationAttribute_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var attribute = new NvidiaOptimizationAttribute
            {
                UseTensorCores = false,
                UseCuBLAS = false,
                UseCuDNN = false,
                UseWarpSpecialization = false,
                OptimizePTXGeneration = false,
                ComputeCapability = "8.6",
                Priority = OptimizationPriority.Performance
            };

            // Assert
            attribute.UseTensorCores.Should().BeFalse();
            attribute.UseCuBLAS.Should().BeFalse();
            attribute.UseCuDNN.Should().BeFalse();
            attribute.UseWarpSpecialization.Should().BeFalse();
            attribute.OptimizePTXGeneration.Should().BeFalse();
            attribute.ComputeCapability.Should().Be("8.6");
            attribute.Priority.Should().Be(OptimizationPriority.Performance);
        }

        [Fact]
        public void NvidiaOptimizationAttribute_InheritsFromPlatformOptimizationAttribute()
        {
            // Arrange & Act
            var attribute = new NvidiaOptimizationAttribute();

            // Assert
            attribute.Should().BeAssignableTo<PlatformOptimizationAttribute>();
            attribute.Platform.Should().Be("NVIDIA");
        }

        #endregion

        #region AMDOptimizationAttribute Tests

        [Fact]
        public void AMDOptimizationAttribute_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var attribute = new AMDOptimizationAttribute();

            // Assert
            attribute.UseROCmBLAS.Should().BeTrue();
            attribute.UseMFMA.Should().BeTrue();
            attribute.UseWavefront64.Should().BeTrue();
            attribute.GFXArchitecture.Should().Be("Auto");
            attribute.OptimizeForOccupancy.Should().BeTrue();
            attribute.Priority.Should().Be(OptimizationPriority.Balanced);
        }

        [Fact]
        public void AMDOptimizationAttribute_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var attribute = new AMDOptimizationAttribute
            {
                UseROCmBLAS = false,
                UseMFMA = false,
                UseWavefront64 = false,
                GFXArchitecture = "gfx1030",
                OptimizeForOccupancy = false,
                Priority = OptimizationPriority.EnergyEfficiency
            };

            // Assert
            attribute.UseROCmBLAS.Should().BeFalse();
            attribute.UseMFMA.Should().BeFalse();
            attribute.UseWavefront64.Should().BeFalse();
            attribute.GFXArchitecture.Should().Be("gfx1030");
            attribute.OptimizeForOccupancy.Should().BeFalse();
            attribute.Priority.Should().Be(OptimizationPriority.EnergyEfficiency);
        }

        [Fact]
        public void AMDOptimizationAttribute_InheritsFromPlatformOptimizationAttribute()
        {
            // Arrange & Act
            var attribute = new AMDOptimizationAttribute();

            // Assert
            attribute.Should().BeAssignableTo<PlatformOptimizationAttribute>();
            attribute.Platform.Should().Be("AMD");
        }

        #endregion

        #region Base PlatformOptimizationAttribute Tests

        [Fact]
        public void PlatformOptimizationAttribute_IsAbstract()
        {
            // Arrange & Act
            var type = typeof(PlatformOptimizationAttribute);

            // Assert
            type.IsAbstract.Should().BeTrue();
            type.Should().BeAssignableTo<Attribute>();
        }

        [Fact]
        public void OptimizationPriority_AllValues_ShouldHaveUniqueIntegerValues()
        {
            // Arrange
            var priorities = Enum.GetValues<OptimizationPriority>();
            var values = new HashSet<int>();

            // Act & Assert
            foreach (var priority in priorities)
            {
                var intValue = (int)priority;
                values.Add(intValue).Should().BeTrue($"Priority {priority} should have unique value {intValue}");
            }

            values.Count.Should().Be(priorities.Length);
        }

        [Fact]
        public void OptimizationPriority_ShouldHaveExpectedValues()
        {
            // Arrange & Act & Assert
            Enum.IsDefined(typeof(OptimizationPriority), OptimizationPriority.Performance).Should().BeTrue();
            Enum.IsDefined(typeof(OptimizationPriority), OptimizationPriority.Balanced).Should().BeTrue();
            Enum.IsDefined(typeof(OptimizationPriority), OptimizationPriority.EnergyEfficiency).Should().BeTrue();
        }

        #endregion

        #region AttributeUsage Tests

        [Fact]
        public void PlatformOptimizationAttributes_AttributeUsage_ShouldBeConfiguredCorrectly()
        {
            // Arrange
            var attributeTypes = new[]
            {
                typeof(AppleOptimizationAttribute),
                typeof(IntelOptimizationAttribute),
                typeof(NvidiaOptimizationAttribute),
                typeof(AMDOptimizationAttribute)
            };

            // Act & Assert
            foreach (var attributeType in attributeTypes)
            {
                var usage = attributeType.GetCustomAttribute<AttributeUsageAttribute>();
                
                usage.Should().NotBeNull($"{attributeType.Name} should have AttributeUsage");
                usage!.ValidOn.Should().Be(AttributeTargets.Method | AttributeTargets.Class | AttributeTargets.Assembly);
                usage.AllowMultiple.Should().BeFalse();
            }
        }

        #endregion

        #region Method Attribute Detection Tests

        [Fact]
        public void PlatformOptimizationAttributes_AppliedToMethod_ShouldBeDetectable()
        {
            // Arrange
            var methodInfo = typeof(TestKernelsWithPlatformOptimizations).GetMethod(nameof(TestKernelsWithPlatformOptimizations.MultiPlatformOptimizedKernel));

            // Act
            var appleAttr = methodInfo?.GetCustomAttribute<AppleOptimizationAttribute>();
            var intelAttr = methodInfo?.GetCustomAttribute<IntelOptimizationAttribute>();
            var nvidiaAttr = methodInfo?.GetCustomAttribute<NvidiaOptimizationAttribute>();
            var amdAttr = methodInfo?.GetCustomAttribute<AMDOptimizationAttribute>();

            // Assert
            appleAttr.Should().NotBeNull();
            intelAttr.Should().NotBeNull();
            nvidiaAttr.Should().NotBeNull();
            amdAttr.Should().NotBeNull();

            appleAttr!.UseAMX.Should().BeTrue();
            intelAttr!.UseNPU.Should().BeTrue();
            nvidiaAttr!.UseTensorCores.Should().BeTrue();
            amdAttr!.UseMFMA.Should().BeTrue();
        }

        #endregion

        #region Property Access Coverage Tests

        [Fact]
        public void AllPlatformOptimizationAttributes_AllProperties_ShouldBeAccessible()
        {
            // Test Apple Optimization
            var appleAttr = new AppleOptimizationAttribute();
            TestAllAppleProperties(appleAttr);

            // Test Intel Optimization  
            var intelAttr = new IntelOptimizationAttribute();
            TestAllIntelProperties(intelAttr);

            // Test NVIDIA Optimization
            var nvidiaAttr = new NvidiaOptimizationAttribute();
            TestAllNvidiaProperties(nvidiaAttr);

            // Test AMD Optimization
            var amdAttr = new AMDOptimizationAttribute();
            TestAllAMDProperties(amdAttr);
        }

        private static void TestAllAppleProperties(AppleOptimizationAttribute attr)
        {
            attr.UseAMX = true;
            attr.UseAMX.Should().BeTrue();
            
            attr.UseNeuralEngine = false;
            attr.UseNeuralEngine.Should().BeFalse();
            
            attr.UseMetalPerformanceShaders = false;
            attr.UseMetalPerformanceShaders.Should().BeFalse();
            
            attr.OptimizeForLatency = true;
            attr.OptimizeForLatency.Should().BeTrue();
            
            attr.UseTiledMemory = false;
            attr.UseTiledMemory.Should().BeFalse();
            
            attr.Priority = OptimizationPriority.Performance;
            attr.Priority.Should().Be(OptimizationPriority.Performance);
        }

        private static void TestAllIntelProperties(IntelOptimizationAttribute attr)
        {
            attr.UseAMX = true;
            attr.UseAMX.Should().BeTrue();
            
            attr.UseAVX512 = false;
            attr.UseAVX512.Should().BeFalse();
            
            attr.UseMKL = false;
            attr.UseMKL.Should().BeFalse();
            
            attr.UseDLBoost = true;
            attr.UseDLBoost.Should().BeTrue();
            
            attr.UseNPU = true;
            attr.UseNPU.Should().BeTrue();
            
            attr.OptimizeForThroughput = false;
            attr.OptimizeForThroughput.Should().BeFalse();
        }

        private static void TestAllNvidiaProperties(NvidiaOptimizationAttribute attr)
        {
            attr.UseTensorCores = false;
            attr.UseTensorCores.Should().BeFalse();
            
            attr.UseCuBLAS = false;
            attr.UseCuBLAS.Should().BeFalse();
            
            attr.UseCuDNN = false;
            attr.UseCuDNN.Should().BeFalse();
            
            attr.UseWarpSpecialization = false;
            attr.UseWarpSpecialization.Should().BeFalse();
            
            attr.OptimizePTXGeneration = false;
            attr.OptimizePTXGeneration.Should().BeFalse();
            
            attr.ComputeCapability = "8.6";
            attr.ComputeCapability.Should().Be("8.6");
        }

        private static void TestAllAMDProperties(AMDOptimizationAttribute attr)
        {
            attr.UseROCmBLAS = false;
            attr.UseROCmBLAS.Should().BeFalse();
            
            attr.UseMFMA = false;
            attr.UseMFMA.Should().BeFalse();
            
            attr.UseWavefront64 = false;
            attr.UseWavefront64.Should().BeFalse();
            
            attr.GFXArchitecture = "gfx1030";
            attr.GFXArchitecture.Should().Be("gfx1030");
            
            attr.OptimizeForOccupancy = false;
            attr.OptimizeForOccupancy.Should().BeFalse();
        }

        #endregion

        public void Dispose()
        {
            _context?.Dispose();
        }

        /// <summary>
        /// Test kernels with platform optimization attributes for testing.
        /// </summary>
        private static class TestKernelsWithPlatformOptimizations
        {
            [AppleOptimization(UseAMX = true, UseNeuralEngine = true)]
            [IntelOptimization(UseAMX = true, UseNPU = true)]
            [NvidiaOptimization(UseTensorCores = true, ComputeCapability = "8.6")]
            [AMDOptimization(UseMFMA = true, GFXArchitecture = "gfx1030")]
            public static void MultiPlatformOptimizedKernel(ArrayView<float> data)
            {
                var index = Grid.GlobalIndex.X;
                if (index < data.Length)
                {
                    data[index] = data[index] * data[index];
                }
            }
        }
    }
}