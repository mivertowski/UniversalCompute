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

using ILGPU.Runtime.HardwareDetection;
using System;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Tests for hardware detection system.
    /// </summary>
    public class HardwareDetectionTests : IDisposable
    {
        private readonly ITestOutputHelper output;
        private readonly Context context;

        public HardwareDetectionTests(ITestOutputHelper output)
        {
            this.output = output;
            context = Context.CreateDefault();
        }

        [Fact]
        public void HardwareManagerInitializes()
        {
            // Act
            HardwareManager.Initialize();

            // Assert
            Assert.True(HardwareManager.IsInitialized);
            Assert.NotNull(HardwareManager.Capabilities);
        }

        [Fact]
        public void HardwareManagerDetectsCapabilities()
        {
            // Arrange
            HardwareManager.Initialize();

            // Act
            var capabilities = HardwareManager.Capabilities;

            // Assert
            Assert.NotNull(capabilities);
            
            // At least one backend should be available (CPU/Velocity at minimum)
            var hasAnyBackend = 
                capabilities.CUDA.IsSupported ||
                capabilities.ROCm.IsSupported ||
                capabilities.OneAPI.IsSupported ||
                capabilities.AMX.IsSupported ||
                capabilities.Apple.IsSupported ||
                capabilities.OpenCL.IsSupported ||
                capabilities.Vulkan.IsSupported ||
                capabilities.Velocity.IsSupported;
                
            Assert.True(hasAnyBackend, "No hardware backends detected");
            
            // Log detection results
            output.WriteLine($"CUDA: {capabilities.CUDA.IsSupported}");
            output.WriteLine($"ROCm: {capabilities.ROCm.IsSupported}");
            output.WriteLine($"OneAPI: {capabilities.OneAPI.IsSupported}");
            output.WriteLine($"AMX: {capabilities.AMX.IsSupported}");
            output.WriteLine($"Apple: {capabilities.Apple.IsSupported}");
            output.WriteLine($"OpenCL: {capabilities.OpenCL.IsSupported}");
            output.WriteLine($"Vulkan: {capabilities.Vulkan.IsSupported}");
            output.WriteLine($"Velocity: {capabilities.Velocity.IsSupported}");
        }

        [Theory]
        [InlineData(WorkloadType.GeneralCompute)]
        [InlineData(WorkloadType.MatrixOperations)]
        [InlineData(WorkloadType.FFTOperations)]
        [InlineData(WorkloadType.AIInference)]
        [InlineData(WorkloadType.ImageProcessing)]
        public void GetBestAcceleratorReturnsValidAccelerator(WorkloadType workloadType)
        {
            // Act
            var accelerator = HardwareManager.GetBestAccelerator(workloadType, context);

            // Assert
            Assert.NotNull(accelerator);
            output.WriteLine($"Best accelerator for {workloadType}: {accelerator.Name} ({accelerator.AcceleratorType})");
            
            // Verify accelerator is functional
            Assert.True(accelerator.MaxNumThreadsPerGroup > 0);
            Assert.True(accelerator.MemorySize > 0);
            
            accelerator.Dispose();
        }

        [Fact]
        public void HardwareManagerRefreshWorks()
        {
            // Arrange
            HardwareManager.Initialize();
            var initialCapabilities = HardwareManager.Capabilities;

            // Act
            HardwareManager.Refresh();
            var refreshedCapabilities = HardwareManager.Capabilities;

            // Assert
            Assert.NotNull(refreshedCapabilities);
            Assert.True(HardwareManager.IsInitialized);
        }

        [Fact]
        public void PrintHardwareInfoExecutesWithoutError()
        {
            // This test ensures the diagnostic method doesn't throw
            var exception = Record.Exception(() => HardwareManager.PrintHardwareInfo());
            Assert.Null(exception);
        }

        [Fact]
        public void CUDACapabilitiesDetectionWorks()
        {
            // Arrange
            HardwareManager.Initialize();

            // Act
            var cuda = HardwareManager.Capabilities.CUDA;

            // Assert
            if (cuda.IsSupported)
            {
                Assert.True(cuda.DeviceCount > 0);
                Assert.True(cuda.MaxComputeCapability > 0);
                Assert.True(cuda.TotalMemory > 0);
                output.WriteLine($"CUDA devices: {cuda.DeviceCount}, Compute: {cuda.MaxComputeCapability / 10.0:F1}");
            }
            else
            {
                output.WriteLine($"CUDA not supported: {cuda.ErrorMessage ?? "No error message"}");
            }
        }

        [Fact]
        public void AMXCapabilitiesDetectionWorks()
        {
            // Arrange
            HardwareManager.Initialize();

            // Act
            var amx = HardwareManager.Capabilities.AMX;

            // Assert
            if (amx.IsSupported)
            {
                Assert.True(amx.MaxTileSize > 0);
                Assert.True(amx.TileCount > 0);
                output.WriteLine($"AMX tiles: {amx.TileCount}x{amx.MaxTileSize}x{amx.MaxTileSize}");
                output.WriteLine($"BF16: {amx.SupportsBF16}, INT8: {amx.SupportsINT8}");
            }
            else
            {
                output.WriteLine($"AMX not supported: {amx.ErrorMessage ?? "No error message"}");
            }
        }

        [Fact]
        public void VelocityCapabilitiesAlwaysAvailable()
        {
            // Arrange
            HardwareManager.Initialize();

            // Act
            var velocity = HardwareManager.Capabilities.Velocity;

            // Assert
            Assert.True(velocity.IsSupported || velocity.ErrorMessage != null);
            
            if (velocity.IsSupported)
            {
                output.WriteLine($"Velocity SIMD support:");
                output.WriteLine($"  AVX2: {velocity.SupportsAVX2}");
                output.WriteLine($"  AVX-512: {velocity.SupportsAVX512}");
                output.WriteLine($"  ARM NEON: {velocity.SupportsNEON}");
                output.WriteLine($"  Vector size: {velocity.VectorSize} bits");
            }
        }

        [Fact]
        public void AcceleratorSelectionPrioritizesCorrectly()
        {
            // Arrange
            HardwareManager.Initialize();
            var capabilities = HardwareManager.Capabilities;

            // Act & Assert for Matrix Operations
            using (var matrixAccel = HardwareManager.GetBestAccelerator(WorkloadType.MatrixOperations, context))
            {
                Assert.NotNull(matrixAccel);
                
                // Should prefer AMX for matrix operations if available
                if (capabilities.AMX.IsSupported && capabilities.AMX.SupportsBF16)
                {
                    output.WriteLine($"Matrix accelerator should prefer AMX, got: {matrixAccel.AcceleratorType}");
                }
            }

            // Act & Assert for AI Inference
            using (var aiAccel = HardwareManager.GetBestAccelerator(WorkloadType.AIInference, context))
            {
                Assert.NotNull(aiAccel);
                
                // Should prefer Apple ANE for AI if available
                if (capabilities.Apple.IsSupported && capabilities.Apple.SupportsNeuralEngine)
                {
                    output.WriteLine($"AI accelerator should prefer ANE, got: {aiAccel.AcceleratorType}");
                }
            }
        }

        public void Dispose()
        {
            context?.Dispose();
        }
    }
}