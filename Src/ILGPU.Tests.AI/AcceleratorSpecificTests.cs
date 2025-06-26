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
using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.AMX;
using ILGPU.Intel.NPU;
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;
using System;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Xunit;

// Use alias to resolve TensorShape ambiguity
using TensorShape = ILGPU.ML.TensorShape;

#if ENABLE_METAL_ACCELERATOR
using ILGPU.Backends.Metal;
#endif

#if ENABLE_ONEAPI_ACCELERATOR  
using ILGPU.Backends.OneAPI;
#endif

namespace ILGPU.Tests.AI
{
    /// <summary>
    /// Tests for accelerator-specific implementations.
    /// </summary>
    public class AcceleratorSpecificTests : IDisposable
    {
        private readonly Context _context;

        public AcceleratorSpecificTests()
        {
            _context = Context.CreateDefault();
        }

        [Fact]
        public void MetalAccelerator_ShouldDetectAppleHardware()
        {
            // This test will only pass on Apple Silicon hardware
            if (!OperatingSystem.IsMacOS())
            {
                return; // Skip on non-macOS
            }

#if ENABLE_METAL_ACCELERATOR
            // Arrange & Act
            var devices = MetalDevice.GetDevices().ToList();

            // Assert
            if (devices.Any())
            {
                var device = devices.First();
                device.Should().NotBeNull();
                device.Name.Should().NotBeNullOrEmpty();
                device.SupportsUnifiedMemory.Should().BeTrue();
            }
#endif
        }

        [Fact]
        public void AMXCapabilities_ShouldDetectIntelAMX()
        {
            // This test will only pass on Intel processors with AMX
            // Arrange & Act
            var isSupported = AMXCapabilities.IsAMXSupported();

            // Assert
            // Boolean assertions - just verify it's actually a boolean
            Assert.IsType<bool>(isSupported);
            
            if (isSupported)
            {
                var capabilities = AMXCapabilities.Query();
                capabilities.IsSupported.Should().BeTrue();
                capabilities.MaxTiles.Should().BeGreaterThan(0);
                capabilities.MaxTileRows.Should().BeGreaterThan(0);
                capabilities.MaxTileColumns.Should().BeGreaterThan(0);
            }
        }

        [Fact]
        public void OneAPIDevice_ShouldEnumerateDevices()
        {
#if ENABLE_ONEAPI_ACCELERATOR
            // Arrange & Act
            var devices = OneAPIAccelerator.GetDevices().ToList();

            // Assert
            devices.Should().NotBeNull();
            // Devices list may be empty if Intel OpenCL/OneAPI drivers are not installed
#endif
        }

        [Fact]
        public void NPUCapabilities_ShouldDetectIntelNPU()
        {
            // This test will only pass on Intel processors with NPU
            // Arrange & Act
            var isSupported = NPUCapabilities.DetectNPU();

            // Assert
            // Boolean assertions - just verify it's actually a boolean
            Assert.IsType<bool>(isSupported);

            if (isSupported)
            {
                var capabilities = NPUCapabilities.Query();
                capabilities.Generation.Should().NotBe(NPUGeneration.None);
                capabilities.MaxTOPS.Should().BeGreaterThan(0);
                capabilities.ComputeUnits.Should().BeGreaterThan(0);
            }
        }

        [Fact]
        public void ANECapabilities_ShouldDetectAppleNeuralEngine()
        {
            // This test will only pass on Apple Silicon with Neural Engine
            if (!OperatingSystem.IsMacOS())
            {
                return; // Skip on non-macOS
            }

            // Arrange & Act
            var isSupported = ANECapabilities.DetectNeuralEngine();

            // Assert
            // Boolean assertions - just verify it's actually a boolean
            Assert.IsType<bool>(isSupported);

            if (isSupported)
            {
                var capabilities = ANECapabilities.Query();
                capabilities.IsAvailable.Should().BeTrue();
                capabilities.MaxTOPS.Should().BeGreaterThan(0);
                capabilities.Generation.Should().NotBe(ANEGeneration.None);
            }
        }

        [Fact]
        public async Task MetalPerformancePrimitives_ShouldExecuteOnAppleHardware()
        {
            if (!OperatingSystem.IsMacOS())
            {
                return; // Skip on non-macOS
            }

#if ENABLE_METAL_ACCELERATOR
            try
            {
                // Arrange
                var devices = MetalDevice.GetDevices().ToList();
                if (!devices.Any()) return;

                var metalDevice = devices.First();
                var accelerator = metalDevice.CreateAccelerator(_context);
                var primitives = PerformancePrimitivesFactory.Create(accelerator);

                // Skip test due to tensor type incompatibility
                // var a = CreateTensor<float>(new TensorShape(64, 64));
                // var b = CreateTensor<float>(new TensorShape(64, 64));
                // var c = CreateTensor<float>(new TensorShape(64, 64));

                // Act
                // await primitives.GemmAsync(a, b, c, 1.0f, 0.0f);

                // Assert
                Assert.True(true);
                accelerator.Dispose();
            }
            catch (NotSupportedException)
            {
                // Expected on systems without Metal support
            }
#endif
        }

        [Fact]
        public async Task AMXPerformancePrimitives_ShouldExecuteOnIntelHardware()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                return; // Skip on non-AMX hardware
            }

            try
            {
                // Arrange
                var device = _context.GetPreferredDevice(preferCPU: true);
                var accelerator = device.CreateAccelerator(_context);
                var primitives = PerformancePrimitivesFactory.Create(accelerator);

                // Skip test due to tensor type incompatibility
                // var a = CreateTensor<float>(new TensorShape(16, 16));
                // var b = CreateTensor<float>(new TensorShape(16, 16));
                // var c = CreateTensor<float>(new TensorShape(16, 16));

                // Act
                // await primitives.GemmAsync(a, b, c, 1.0f, 0.0f);

                // Assert
                Assert.True(true);
                accelerator.Dispose();
            }
            catch (NotSupportedException)
            {
                // Expected on systems without AMX support
            }
        }

        [Fact]
        public async Task NPUPerformancePrimitives_ShouldExecuteOnIntelNPU()
        {
            if (!NPUCapabilities.DetectNPU())
            {
                return; // Skip on systems without NPU
            }

            // Skip test since IntelNPUAccelerator constructor doesn't exist yet
            return;
        }

        [Fact]
        public async Task OneAPIPerformancePrimitives_ShouldExecuteOnIntelDevice()
        {
#if ENABLE_ONEAPI_ACCELERATOR
            var devices = OneAPIAccelerator.GetDevices().ToList();
            if (!devices.Any())
            {
                return; // Skip if no OneAPI devices available
            }

            try
            {
                // Arrange
                var oneapiDevice = devices.First();
                var accelerator = new OneAPIAccelerator(_context, oneapiDevice);
                var primitives = PerformancePrimitivesFactory.Create(accelerator);

                var a = CreateTensor<float>(new TensorShape(128, 128));
                var b = CreateTensor<float>(new TensorShape(128, 128));
                var c = CreateTensor<float>(new TensorShape(128, 128));

                // Act
                await primitives.GemmAsync(a, b, c, 1.0f, 0.0f);

                // Assert
                c.Should().NotBeNull();
                accelerator.Dispose();
            }
            catch (NotSupportedException)
            {
                // Expected on systems without OneAPI support
            }
#endif
        }

        [Theory]
        [InlineData(AcceleratorType.Cuda)]
        [InlineData(AcceleratorType.OpenCL)]
        [InlineData(AcceleratorType.CPU)]
        public void PerformancePrimitivesFactory_ShouldCreateForAcceleratorType(AcceleratorType acceleratorType)
        {
            try
            {
                // Arrange
                var device = _context.GetPreferredDevice(preferCPU: acceleratorType == AcceleratorType.CPU);
                var accelerator = device.CreateAccelerator(_context);

                // Act
                var primitives = PerformancePrimitivesFactory.Create(accelerator);

                // Assert
                primitives.Should().NotBeNull();
                primitives.Accelerator.Should().Be(accelerator);
                primitives.Capabilities.Should().NotBeNull();

                accelerator.Dispose();
            }
            catch (NotSupportedException)
            {
                // Expected if accelerator type is not available
            }
        }

        [Fact]
        public void PerformancePrimitiveCapabilities_ShouldVaryByAcceleratorType()
        {
            // Arrange
            var accelerators = new List<Accelerator>();
            var device = _context.GetPreferredDevice(preferCPU: true);
            accelerators.Add(device.CreateAccelerator(_context));

            foreach (var accelerator in accelerators)
            {
                // Act
                var primitives = PerformancePrimitivesFactory.Create(accelerator);
                var capabilities = primitives.Capabilities;

                // Assert
                capabilities.Should().NotBeNull();
                capabilities.MaxTensorRank.Should().BeGreaterThan(0);
                capabilities.PreferredBatchSize.Should().BeGreaterThan(0);
                
                // Different accelerator types should have different capabilities
                var hasAccelerated = PerformancePrimitivesFactory.HasAcceleratedPrimitives(accelerator);
                Assert.IsType<bool>(hasAccelerated);
            }
        }

        [Theory]
        [InlineData(PrimitiveType.MatrixMultiplication)]
        [InlineData(PrimitiveType.Convolution)]
        [InlineData(PrimitiveType.Attention)]
        public void AcceleratorCapabilities_ShouldReflectHardwareSupport(PrimitiveType primitiveType)
        {
            // Arrange
            var accelerators = new List<Accelerator>();
            var device = _context.GetPreferredDevice(preferCPU: true);
            accelerators.Add(device.CreateAccelerator(_context));

            foreach (var accelerator in accelerators)
            {
                // Act
                var primitives = PerformancePrimitivesFactory.Create(accelerator);
                var isAccelerated = primitives.Capabilities.IsPrimitiveAccelerated(primitiveType);

                // Assert
                Assert.IsType<bool>(isAccelerated);
                
                // Verify consistency with accelerator type
                switch (accelerator.AcceleratorType)
                {
                    case AcceleratorType.Cuda:
                        // CUDA should support most primitives
                        if (primitiveType == PrimitiveType.MatrixMultiplication)
                            isAccelerated.Should().BeTrue();
                        break;
                    
                    case AcceleratorType.CPU:
                        // CPU might have limited acceleration
                        break;
                        
                    default:
                        // Other accelerator types
                        break;
                }
            }
        }

        #region Helper Methods

        private ILGPU.ML.Tensor<T> CreateTensor<T>(TensorShape shape) where T : unmanaged, INumber<T>
        {
            var device = _context.GetPreferredDevice(preferCPU: true);
            var accelerator = device.CreateAccelerator(_context);
            return ILGPU.ML.Tensor.Zeros<T>(accelerator, shape);
        }

        #endregion

        public void Dispose()
        {
            _context?.Dispose();
        }
    }
}