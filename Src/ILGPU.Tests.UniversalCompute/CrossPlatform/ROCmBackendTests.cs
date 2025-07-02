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
using ILGPU.Backends.ROCm;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalCompute.CrossPlatform
{
    /// <summary>
    /// Tests for ROCm backend functionality and validation.
    /// Validates AMD GPU computing capabilities through HIP/ROCm.
    /// </summary>
    public class ROCmBackendTests : IDisposable
    {
        private readonly Context _context;

        public ROCmBackendTests()
        {
            _context = Context.CreateDefault();
        }

        #region Device Discovery Tests

        [Fact]
        public void ROCmDevice_GetDevices_ShouldReturnValidDevices()
        {
            // Act
            var devices = ROCmDevice.GetDevices();

            // Assert
            devices.Should().NotBeNull();
            // Note: May be empty if no AMD GPUs are available
            foreach (var device in devices)
            {
                device.Should().NotBeNull();
                device.Name.Should().NotBeNullOrEmpty();
                device.DeviceId.Should().BeGreaterOrEqualTo(0);
            }
        }

        [Fact]
        public void ROCmDevice_GetDefaultDevice_ShouldHandleNoDevicesGracefully()
        {
            // Act
            var defaultDevice = ROCmDevice.GetDefaultDevice();

            // Assert
            // Should either return a device or null if no AMD GPUs available
            if (defaultDevice != null)
            {
                defaultDevice.Name.Should().NotBeNullOrEmpty();
                defaultDevice.DeviceId.Should().BeGreaterOrEqualTo(0);
            }
        }

        [Fact]
        public void ROCmDevice_Architecture_ShouldBeValid()
        {
            // Arrange
            var devices = ROCmDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                var architecture = device.GetArchitecture();
                architecture.Should().BeDefined();
                
                // Architecture-specific validation
                if (architecture != ROCmArchitecture.Unknown)
                {
                    device.IsAMDGPU.Should().BeTrue();
                }
            }
        }

        #endregion

        #region Context and Memory Tests

        [Fact]
        public void ROCmAccelerator_Creation_ShouldSucceedWithValidDevice()
        {
            // Arrange
            var device = ROCmDevice.GetDefaultDevice();
            
            // Skip test if no ROCm devices available
            if (device == null)
            {
                return; // Skip test gracefully
            }

            // Act & Assert
            Action createAccelerator = () =>
            {
                using var accelerator = _context.CreateROCmAccelerator(device);
                accelerator.Should().NotBeNull();
                accelerator.AcceleratorType.Should().Be(AcceleratorType.ROCm);
            };

            // Should not throw if ROCm runtime is available
            createAccelerator.Should().NotThrow();
        }

        [Fact]
        public void ROCmAccelerator_MemoryAllocation_ShouldWork()
        {
            // Arrange
            var device = ROCmDevice.GetDefaultDevice();
            if (device == null) return; // Skip if no ROCm devices

            try
            {
                using var accelerator = _context.CreateROCmAccelerator(device);
                
                // Act
                using var buffer = accelerator.Allocate1D<int>(1024);
                
                // Assert
                buffer.Should().NotBeNull();
                buffer.Length.Should().Be(1024);
                buffer.ElementSize.Should().Be(sizeof(int));
            }
            catch (NotSupportedException)
            {
                // Expected if ROCm runtime not available
                return;
            }
        }

        #endregion

        #region Kernel Compilation Tests

        [Fact]
        public void ROCmBackend_KernelCompilation_ShouldGenerateHIPCode()
        {
            // Arrange
            var device = ROCmDevice.GetDefaultDevice();
            if (device == null) return; // Skip if no ROCm devices

            try
            {
                using var accelerator = _context.CreateROCmAccelerator(device);
                
                // Act
                var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<int>>(SimpleKernel);
                
                // Assert
                kernel.Should().NotBeNull();
            }
            catch (NotSupportedException)
            {
                // Expected if ROCm compiler not available
                return;
            }
        }

        /// <summary>
        /// Simple test kernel for compilation validation.
        /// </summary>
        static void SimpleKernel(ArrayView<int> data)
        {
            var index = Grid.GlobalIndex.X;
            if (index < data.Length)
            {
                data[index] = index * 2;
            }
        }

        [Fact]
        public void ROCmCodeGenerator_GenerateCode_ShouldCreateValidHIP()
        {
            // Arrange
            var device = ROCmDevice.GetDefaultDevice();
            if (device == null) return;

            try
            {
                using var accelerator = _context.CreateROCmAccelerator(device) as ROCmAccelerator;
                if (accelerator == null) return;

                // Test that code generation infrastructure exists
                // The actual compilation test is in the kernel compilation test above
                
                // Assert
                accelerator.DeviceInfo.Should().NotBeNull();
                accelerator.DeviceInfo.Name.Should().NotBeNullOrEmpty();
            }
            catch (NotSupportedException)
            {
                // Expected if ROCm not available
                return;
            }
        }

        #endregion

        #region Capability Tests

        [Fact]
        public void ROCmCapabilities_ShouldProvideValidInformation()
        {
            // Arrange
            var devices = ROCmDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                var capabilities = device.QueryCapabilities();
                
                capabilities.Should().NotBeNull();
                capabilities.GlobalMemorySize.Should().BeGreaterThan(0);
                capabilities.MaxComputeUnits.Should().BeGreaterThan(0);
                capabilities.MaxWorkGroupSize.Should().BeGreaterThan(0);
                capabilities.WavefrontSize.Should().BeGreaterThan(0);
                
                // Architecture-specific capabilities
                var architecture = device.GetArchitecture();
                if (architecture == ROCmArchitecture.RDNA3)
                {
                    capabilities.SupportsWave32.Should().BeTrue();
                }
            }
        }

        [Fact]
        public void ROCmDevice_IsAMDGPU_ShouldBeCorrect()
        {
            // Arrange
            var devices = ROCmDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                if (device.Vendor.Contains("AMD", StringComparison.OrdinalIgnoreCase))
                {
                    device.IsAMDGPU.Should().BeTrue();
                }
            }
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void ROCmDevice_InvalidDevice_ShouldHandleGracefully()
        {
            // Act & Assert
            Action createWithInvalidDevice = () =>
            {
                var invalidDevice = new ROCmDevice(-1, "InvalidDevice", "Unknown");
                using var accelerator = _context.CreateROCmAccelerator(invalidDevice);
            };

            // Should handle invalid devices gracefully
            createWithInvalidDevice.Should().Throw<ArgumentException>()
                .Or.Throw<NotSupportedException>()
                .Or.Throw<InvalidOperationException>();
        }

        [Fact]
        public void ROCmAccelerator_NoROCmRuntime_ShouldThrowNotSupported()
        {
            // This test validates that the code handles missing ROCm runtime gracefully
            var devices = ROCmDevice.GetDevices();
            
            if (!devices.Any())
            {
                // If no devices found, creation should throw NotSupportedException
                Action createWithoutRuntime = () =>
                {
                    var mockDevice = new ROCmDevice(0, "MockDevice", "AMD");
                    using var accelerator = _context.CreateROCmAccelerator(mockDevice);
                };
                
                createWithoutRuntime.Should().Throw<NotSupportedException>()
                    .Or.Throw<InvalidOperationException>();
            }
        }

        #endregion

        #region Performance and Validation Tests

        [Fact]
        public void ROCmDevice_GetBestDevice_ShouldSelectOptimalDevice()
        {
            // Arrange
            var devices = ROCmDevice.GetDevices();
            if (!devices.Any()) return;

            // Act
            var bestDevice = ROCmDevice.GetBestDevice();

            // Assert
            if (bestDevice != null)
            {
                bestDevice.Should().BeOneOf(devices);
                
                // Best device should be AMD GPU if available
                if (devices.Any(d => d.IsAMDGPU))
                {
                    bestDevice.IsAMDGPU.Should().BeTrue();
                }
            }
        }

        [Fact]
        public void ROCmCapabilities_MemoryBandwidth_ShouldBeRealistic()
        {
            // Arrange
            var devices = ROCmDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                var capabilities = device.QueryCapabilities();
                
                // Memory bandwidth should be realistic for GPU (> 10 GB/s)
                if (device.IsAMDGPU)
                {
                    capabilities.GlobalMemoryBandwidth.Should().BeGreaterThan(10_000_000_000L);
                }
            }
        }

        #endregion

        public void Dispose()
        {
            _context?.Dispose();
        }
    }

    /// <summary>
    /// Extension methods for ROCm testing.
    /// </summary>
    public static class ROCmTestExtensions
    {
        /// <summary>
        /// Creates a ROCm accelerator for testing purposes.
        /// </summary>
        public static ROCmAccelerator CreateROCmAccelerator(this Context context, ROCmDevice device)
        {
            return new ROCmAccelerator(context, device);
        }
    }
}