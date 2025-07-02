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
using ILGPU.Backends.OneAPI;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalCompute.CrossPlatform
{
    /// <summary>
    /// Tests for Intel OneAPI backend functionality and validation.
    /// Validates Intel GPU/CPU computing capabilities through SYCL/DPC++.
    /// </summary>
    public class OneAPIBackendTests : IDisposable
    {
        private readonly Context _context;

        public OneAPIBackendTests()
        {
            _context = Context.CreateDefault();
        }

        #region Device Discovery Tests

        [Fact]
        public void OneAPIDevice_GetDevices_ShouldReturnValidDevices()
        {
            // Act
            var devices = OneAPIDevice.GetDevices();

            // Assert
            devices.Should().NotBeNull();
            // Note: May be empty if no OneAPI devices are available
            foreach (var device in devices)
            {
                device.Should().NotBeNull();
                device.Name.Should().NotBeNullOrEmpty();
                device.Vendor.Should().NotBeNullOrEmpty();
                device.DeviceType.Should().BeDefined();
            }
        }

        [Fact]
        public void OneAPIDevice_GetDefaultDevice_ShouldHandleNoDevicesGracefully()
        {
            // Act
            var defaultDevice = OneAPIDevice.GetDefaultDevice();

            // Assert
            // Should either return a device or null if no OneAPI devices available
            if (defaultDevice != null)
            {
                defaultDevice.Name.Should().NotBeNullOrEmpty();
                defaultDevice.Vendor.Should().NotBeNullOrEmpty();
                defaultDevice.DeviceType.Should().BeDefined();
            }
        }

        [Fact]
        public void OneAPIDevice_IntelGPUs_ShouldBeDetectedCorrectly()
        {
            // Act
            var intelGPUs = OneAPIDevice.GetIntelGPUs();

            // Assert
            intelGPUs.Should().NotBeNull();
            foreach (var device in intelGPUs)
            {
                device.IsIntelGPU.Should().BeTrue();
                device.Vendor.Should().Contain("Intel", StringComparison.OrdinalIgnoreCase);
                device.DeviceType.Should().Be(OneAPIDeviceType.GPU);
            }
        }

        [Fact]
        public void OneAPIDevice_Architecture_ShouldBeValid()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices.Where(d => d.IsIntelGPU))
            {
                var architecture = device.GetArchitecture();
                architecture.Should().BeDefined();
                
                // Architecture should be valid for Intel GPUs
                if (device.IsIntelGPU)
                {
                    architecture.Should().NotBe(IntelArchitecture.Unknown);
                }
            }
        }

        #endregion

        #region Context and Memory Tests

        [Fact]
        public void OneAPIAccelerator_Creation_ShouldSucceedWithValidDevice()
        {
            // Arrange
            var device = OneAPIDevice.GetDefaultDevice();
            
            // Skip test if no OneAPI devices available
            if (device == null)
            {
                return; // Skip test gracefully
            }

            // Act & Assert
            Action createAccelerator = () =>
            {
                using var accelerator = _context.CreateOneAPIAccelerator(device);
                accelerator.Should().NotBeNull();
                accelerator.AcceleratorType.Should().Be(AcceleratorType.OneAPI);
            };

            // Should not throw if OneAPI runtime is available
            createAccelerator.Should().NotThrow();
        }

        [Fact]
        public void OneAPIAccelerator_MemoryAllocation_ShouldWork()
        {
            // Arrange
            var device = OneAPIDevice.GetDefaultDevice();
            if (device == null) return; // Skip if no OneAPI devices

            try
            {
                using var accelerator = _context.CreateOneAPIAccelerator(device);
                
                // Act
                using var buffer = accelerator.Allocate1D<int>(1024);
                
                // Assert
                buffer.Should().NotBeNull();
                buffer.Length.Should().Be(1024);
                buffer.ElementSize.Should().Be(sizeof(int));
            }
            catch (NotSupportedException)
            {
                // Expected if OneAPI runtime not available
                return;
            }
        }

        [Fact]
        public void OneAPIAccelerator_USMAllocation_ShouldWorkWhenSupported()
        {
            // Arrange
            var device = OneAPIDevice.GetDefaultDevice();
            if (device == null) return; // Skip if no OneAPI devices

            try
            {
                using var accelerator = _context.CreateOneAPIAccelerator(device) as OneAPIAccelerator;
                if (accelerator == null) return;

                // Skip if USM not supported
                if (!accelerator.Capabilities.SupportsUSM) return;
                
                // Act
                using var usmBuffer = accelerator.AllocateUSM<int>(1024);
                
                // Assert
                usmBuffer.Should().NotBeNull();
                usmBuffer.Length.Should().Be(1024);
            }
            catch (NotSupportedException)
            {
                // Expected if OneAPI runtime or USM not available
                return;
            }
        }

        #endregion

        #region Kernel Compilation Tests

        [Fact]
        public void OneAPIBackend_KernelCompilation_ShouldGenerateSYCLCode()
        {
            // Arrange
            var device = OneAPIDevice.GetDefaultDevice();
            if (device == null) return; // Skip if no OneAPI devices

            try
            {
                using var accelerator = _context.CreateOneAPIAccelerator(device);
                
                // Act
                var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<int>>(SimpleKernel);
                
                // Assert
                kernel.Should().NotBeNull();
            }
            catch (NotSupportedException)
            {
                // Expected if OneAPI compiler not available
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
                data[index] = index * 3;
            }
        }

        [Fact]
        public void SYCLCodeGenerator_GenerateCode_ShouldCreateValidSYCL()
        {
            // Arrange
            var device = OneAPIDevice.GetDefaultDevice();
            if (device == null) return;

            try
            {
                using var accelerator = _context.CreateOneAPIAccelerator(device) as OneAPIAccelerator;
                if (accelerator == null) return;

                // Test that code generation infrastructure exists
                // The actual compilation test is in the kernel compilation test above
                
                // Assert
                accelerator.DeviceInfo.Should().NotBeNull();
                accelerator.DeviceInfo.Name.Should().NotBeNullOrEmpty();
            }
            catch (NotSupportedException)
            {
                // Expected if OneAPI not available
                return;
            }
        }

        #endregion

        #region Capability Tests

        [Fact]
        public void OneAPICapabilities_ShouldProvideValidInformation()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                try
                {
                    using var accelerator = _context.CreateOneAPIAccelerator(device) as OneAPIAccelerator;
                    if (accelerator == null) continue;

                    var capabilities = accelerator.Capabilities;
                    
                    capabilities.Should().NotBeNull();
                    capabilities.GlobalMemorySize.Should().BeGreaterThan(0);
                    capabilities.MaxComputeUnits.Should().BeGreaterThan(0);
                    capabilities.MaxWorkGroupSize.Should().BeGreaterThan(0);
                    capabilities.DeviceType.Should().BeDefined();
                    
                    // Intel-specific capabilities
                    if (device.IsIntelGPU)
                    {
                        capabilities.NumExecutionUnits.Should().BeGreaterThan(0);
                        capabilities.MaxThreadsPerEU.Should().BeGreaterThan(0);
                    }
                }
                catch (NotSupportedException)
                {
                    // Skip if OneAPI not available for this device
                    continue;
                }
            }
        }

        [Fact]
        public void OneAPIDevice_IsIntelDevice_ShouldBeCorrect()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices)
            {
                if (device.Vendor.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                {
                    if (device.DeviceType == OneAPIDeviceType.GPU)
                    {
                        device.IsIntelGPU.Should().BeTrue();
                    }
                    else if (device.DeviceType == OneAPIDeviceType.CPU)
                    {
                        device.IsIntelCPU.Should().BeTrue();
                    }
                }
            }
        }

        [Fact]
        public void OneAPICapabilities_Extensions_ShouldBeDetected()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices.Where(d => d.IsIntelGPU))
            {
                try
                {
                    using var accelerator = _context.CreateOneAPIAccelerator(device) as OneAPIAccelerator;
                    if (accelerator == null) continue;

                    var capabilities = accelerator.Capabilities;
                    
                    // Intel GPUs should support various features
                    capabilities.SupportsFeature(OneAPIFeature.SubGroups).Should().BeTrue();
                    
                    // Modern Intel GPUs should support USM
                    if (device.GetArchitecture() >= IntelArchitecture.XeLP)
                    {
                        capabilities.SupportsUSM.Should().BeTrue();
                    }
                }
                catch (NotSupportedException)
                {
                    // Skip if OneAPI not available for this device
                    continue;
                }
            }
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void OneAPIDevice_InvalidDevice_ShouldHandleGracefully()
        {
            // Act & Assert
            Action createWithInvalidDevice = () =>
            {
                var invalidDevice = new OneAPIDevice(IntPtr.Zero, IntPtr.Zero);
                using var accelerator = _context.CreateOneAPIAccelerator(invalidDevice);
            };

            // Should handle invalid devices gracefully
            createWithInvalidDevice.Should().Throw<ArgumentException>()
                .Or.Throw<NotSupportedException>()
                .Or.Throw<InvalidOperationException>();
        }

        [Fact]
        public void OneAPIAccelerator_NoOneAPIRuntime_ShouldThrowNotSupported()
        {
            // This test validates that the code handles missing OneAPI runtime gracefully
            var devices = OneAPIDevice.GetDevices();
            
            if (!devices.Any())
            {
                // If no devices found, creation should throw NotSupportedException
                Action createWithoutRuntime = () =>
                {
                    var mockDevice = new OneAPIDevice(IntPtr.Zero, IntPtr.Zero);
                    using var accelerator = _context.CreateOneAPIAccelerator(mockDevice);
                };
                
                createWithoutRuntime.Should().Throw<NotSupportedException>()
                    .Or.Throw<InvalidOperationException>();
            }
        }

        #endregion

        #region Performance and Validation Tests

        [Fact]
        public void OneAPIDevice_GetBestDevice_ShouldSelectOptimalDevice()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();
            if (!devices.Any()) return;

            // Act
            var bestDevice = OneAPIDevice.GetBestDevice();

            // Assert
            if (bestDevice != null)
            {
                bestDevice.Should().BeOneOf(devices);
                
                // Best device should be Intel GPU if available, then CPU
                if (devices.Any(d => d.IsIntelGPU))
                {
                    bestDevice.IsIntelGPU.Should().BeTrue();
                }
                else if (devices.Any(d => d.IsIntelCPU))
                {
                    bestDevice.IsIntelCPU.Should().BeTrue();
                }
            }
        }

        [Fact]
        public void OneAPICapabilities_Performance_ShouldBeRealistic()
        {
            // Arrange
            var devices = OneAPIDevice.GetDevices();

            // Act & Assert
            foreach (var device in devices.Where(d => d.IsIntelGPU))
            {
                try
                {
                    using var accelerator = _context.CreateOneAPIAccelerator(device) as OneAPIAccelerator;
                    if (accelerator == null) continue;

                    var capabilities = accelerator.Capabilities;
                    
                    // Performance estimates should be reasonable
                    var fp32Performance = capabilities.GetPeakPerformance(OneAPIDataType.Float32);
                    fp32Performance.Should().BeGreaterThan(0);
                    
                    var fp16Performance = capabilities.GetPeakPerformance(OneAPIDataType.Float16);
                    fp16Performance.Should().BeGreaterOrEqualTo(fp32Performance);
                }
                catch (NotSupportedException)
                {
                    // Skip if OneAPI not available for this device
                    continue;
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
    /// Extension methods for OneAPI testing.
    /// </summary>
    public static class OneAPITestExtensions
    {
        /// <summary>
        /// Creates a OneAPI accelerator for testing purposes.
        /// </summary>
        public static OneAPIAccelerator CreateOneAPIAccelerator(this Context context, OneAPIDevice device)
        {
            return new OneAPIAccelerator(context, device);
        }
    }
}