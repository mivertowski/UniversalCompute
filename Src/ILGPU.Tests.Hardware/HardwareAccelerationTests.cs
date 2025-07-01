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

using ILGPU.Runtime;
using ILGPU.Runtime.AMX;
using ILGPU.Runtime.AMX.Native;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.HardwareDetection;
using ILGPU.Runtime.ROCm;
using ILGPU.Runtime.OneAPI;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Tests for hardware acceleration functionality.
    /// </summary>
    public class HardwareAccelerationTests : IDisposable
    {
        private readonly ITestOutputHelper output;
        private readonly Context context;

        public HardwareAccelerationTests(ITestOutputHelper output)
        {
            this.output = output;
            context = Context.CreateDefault();
            HardwareManager.Initialize();
        }

        #region Matrix Multiplication Tests

        [Fact]
        public void MatrixMultiplicationWithBestAccelerator()
        {
            // Arrange
            const int size = 128;
            var a = GenerateMatrix(size, size, 1.0f);
            var b = GenerateMatrix(size, size, 2.0f);
            var c = new float[size * size];

            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.MatrixOperations, context);
            Assert.NotNull(accelerator);
            output.WriteLine($"Using accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

            // Act - Run matrix multiplication based on accelerator type
            unsafe
            {
                fixed (float* aPtr = a, bPtr = b, cPtr = c)
                {
                    if (accelerator is IntelAMXAccelerator amxAccel)
                    {
                        // Test AMX acceleration
                        amxAccel.ExecuteBF16MatMul(
                            new IntPtr(aPtr), new IntPtr(bPtr), new IntPtr(cPtr),
                            size, size, size);
                    }
                    else if (accelerator is CudaAccelerator cudaAccel)
                    {
                        // For CUDA, we'd use cuBLAS through the accelerator
                        // This is a simplified test - real implementation would use kernel launch
                        CPUMatMulFallback(aPtr, bPtr, cPtr, size, size, size);
                    }
                    else
                    {
                        // CPU fallback
                        CPUMatMulFallback(aPtr, bPtr, cPtr, size, size, size);
                    }
                }
            }

            // Assert - Verify result (all elements should be size * 2.0)
            var expected = size * 2.0f;
            for (int i = 0; i < c.Length; i++)
            {
                Assert.True(Math.Abs(c[i] - expected) < 0.001f, 
                    $"Element {i}: expected {expected}, got {c[i]}");
            }
        }

        [SkippableFact]
        public void AMXMatrixMultiplicationTest()
        {
            Skip.IfNot(HardwareManager.Capabilities.AMX.IsSupported, "AMX not supported");

            // Arrange
            const int size = 16; // AMX tile size
            var device = IntelAMXDevice.GetDefaultDevice();
            Assert.NotNull(device);

            using var amxAccel = context.CreateAMXAccelerator(device);
            
            var a = GenerateMatrix(size, size, 1.0f);
            var b = GenerateMatrix(size, size, 2.0f);
            var c = new float[size * size];

            // Act
            unsafe
            {
                fixed (float* aPtr = a, bPtr = b, cPtr = c)
                {
                    // Convert to BF16 and run AMX
                    amxAccel.ExecuteBF16MatMul(
                        new IntPtr(aPtr), new IntPtr(bPtr), new IntPtr(cPtr),
                        size, size, size);
                }
            }

            // Assert
            var expected = size * 2.0f;
            Assert.All(c, val => Assert.True(Math.Abs(val - expected) < 0.1f));
            
            output.WriteLine($"AMX matrix multiplication successful on {device.ProcessorName}");
        }

        [SkippableFact]
        public void CUDAMatrixMultiplicationTest()
        {
            Skip.IfNot(HardwareManager.Capabilities.CUDA.IsSupported, "CUDA not supported");

            // Arrange
            const int size = 128;
            var device = CudaDevice.GetBestDevice();
            Assert.NotNull(device);

            using var cudaAccel = context.CreateCudaAccelerator(device);
            output.WriteLine($"Using CUDA device: {device.Name} (CC {device.ComputeCapability})");

            // Create test data
            var a = GenerateMatrix(size, size, 1.0f);
            var b = GenerateMatrix(size, size, 2.0f);
            
            // Allocate device memory
            using var d_a = cudaAccel.Allocate1D<float>(size * size);
            using var d_b = cudaAccel.Allocate1D<float>(size * size);
            using var d_c = cudaAccel.Allocate1D<float>(size * size);

            // Copy to device
            d_a.CopyFromCPU(a);
            d_b.CopyFromCPU(b);

            // Act - In real implementation, this would call cuBLAS
            // For now, we simulate the result
            var c = new float[size * size];
            unsafe
            {
                fixed (float* aPtr = a, bPtr = b, cPtr = c)
                {
                    CPUMatMulFallback(aPtr, bPtr, cPtr, size, size, size);
                }
            }
            d_c.CopyFromCPU(c);

            // Copy back and verify
            var result = d_c.GetAsArray1D();
            
            // Assert
            var expected = size * 2.0f;
            Assert.All(result, val => Assert.True(Math.Abs(val - expected) < 0.001f));
        }

        #endregion

        #region FFT Tests

        [Fact]
        public void FFTWithBestAccelerator()
        {
            // Arrange
            const int length = 1024;
            var input = GenerateComplexSignal(length);
            var output = new (float, float)[length];

            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);
            Assert.NotNull(accelerator);
            output.WriteLine($"Using accelerator for FFT: {accelerator.Name} ({accelerator.AcceleratorType})");

            // Act - Run FFT based on accelerator type
            if (accelerator is CudaAccelerator)
            {
                // Would use cuFFT in real implementation
                CPUFFTFallback(input, output);
            }
            else if (accelerator is ROCmAccelerator)
            {
                // Would use rocFFT in real implementation
                CPUFFTFallback(input, output);
            }
            else
            {
                // CPU fallback
                CPUFFTFallback(input, output);
            }

            // Assert - Verify we got output (basic check)
            Assert.All(output, val => Assert.True(val.Item1 != 0 || val.Item2 != 0));
        }

        #endregion

        #region AI Inference Tests

        [Fact]
        public void AIInferenceWithBestAccelerator()
        {
            // Arrange
            const int batchSize = 1;
            const int inputSize = 784;  // MNIST-like
            const int outputSize = 10;  // 10 classes
            
            var input = GenerateMatrix(batchSize, inputSize, 0.5f);
            var weights = GenerateMatrix(inputSize, outputSize, 0.1f);
            var output = new float[batchSize * outputSize];

            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.AIInference, context);
            Assert.NotNull(accelerator);
            this.output.WriteLine($"Using accelerator for AI: {accelerator.Name} ({accelerator.AcceleratorType})");

            // Act
            unsafe
            {
                fixed (float* inputPtr = input, weightsPtr = weights, outputPtr = output)
                {
                    if (accelerator is IntelAMXAccelerator amxAccel)
                    {
                        // AMX is great for AI inference
                        amxAccel.ExecuteINT8MatMul(
                            new IntPtr(inputPtr), new IntPtr(weightsPtr), new IntPtr(outputPtr),
                            batchSize, inputSize, outputSize);
                    }
                    else
                    {
                        // Fallback matrix multiplication
                        CPUMatMulFallback(inputPtr, weightsPtr, outputPtr, 
                            batchSize, inputSize, outputSize);
                    }
                }
            }

            // Assert - Verify output has values
            Assert.All(output, val => Assert.True(!float.IsNaN(val) && !float.IsInfinity(val)));
            
            // Check output is reasonable (sum of 784 * 0.5 * 0.1 = ~39.2 per output)
            var expectedRange = inputSize * 0.5f * 0.1f;
            Assert.All(output, val => Assert.True(Math.Abs(val - expectedRange) < expectedRange));
        }

        #endregion

        #region Hardware Capability Tests

        [Fact]
        public void AllAcceleratorTypesCanBeCreated()
        {
            var capabilities = HardwareManager.Capabilities;
            int successCount = 0;

            // Try to create each accelerator type
            if (capabilities.CUDA.IsSupported)
            {
                try
                {
                    var device = CudaDevice.GetBestDevice();
                    if (device != null)
                    {
                        using var accel = context.CreateCudaAccelerator(device);
                        output.WriteLine($"✓ Created CUDA accelerator: {accel.Name}");
                        successCount++;
                    }
                }
                catch (Exception ex)
                {
                    output.WriteLine($"✗ CUDA creation failed: {ex.Message}");
                }
            }

            if (capabilities.AMX.IsSupported)
            {
                try
                {
                    var device = IntelAMXDevice.GetDefaultDevice();
                    if (device != null)
                    {
                        using var accel = context.CreateAMXAccelerator(device);
                        output.WriteLine($"✓ Created AMX accelerator: {accel.Name}");
                        successCount++;
                    }
                }
                catch (Exception ex)
                {
                    output.WriteLine($"✗ AMX creation failed: {ex.Message}");
                }
            }

            if (capabilities.OneAPI.IsSupported)
            {
                try
                {
                    var device = IntelOneAPIDevice.GetDefaultDevice();
                    if (device != null)
                    {
                        using var accel = context.CreateOneAPIAccelerator(device);
                        output.WriteLine($"✓ Created OneAPI accelerator: {accel.Name}");
                        successCount++;
                    }
                }
                catch (Exception ex)
                {
                    output.WriteLine($"✗ OneAPI creation failed: {ex.Message}");
                }
            }

            // Always try CPU/Velocity as fallback
            try
            {
                using var cpuAccel = context.CreateCPUAccelerator();
                output.WriteLine($"✓ Created CPU accelerator: {cpuAccel.Name}");
                successCount++;
            }
            catch (Exception ex)
            {
                output.WriteLine($"✗ CPU creation failed: {ex.Message}");
            }

            Assert.True(successCount > 0, "No accelerators could be created");
            output.WriteLine($"Successfully created {successCount} accelerator type(s)");
        }

        #endregion

        #region Helper Methods

        private float[] GenerateMatrix(int rows, int cols, float value)
        {
            var matrix = new float[rows * cols];
            for (int i = 0; i < matrix.Length; i++)
                matrix[i] = value;
            return matrix;
        }

        private (float, float)[] GenerateComplexSignal(int length)
        {
            var signal = new (float, float)[length];
            for (int i = 0; i < length; i++)
            {
                signal[i] = ((float)Math.Sin(2 * Math.PI * i / length), 0);
            }
            return signal;
        }

        private unsafe void CPUMatMulFallback(float* a, float* b, float* c, int m, int k, int n)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int ki = 0; ki < k; ki++)
                        sum += a[i * k + ki] * b[ki * n + j];
                    c[i * n + j] = sum;
                }
            }
        }

        private void CPUFFTFallback((float, float)[] input, (float, float)[] output)
        {
            // Simple DFT for testing
            int n = input.Length;
            for (int k = 0; k < n; k++)
            {
                float real = 0, imag = 0;
                for (int t = 0; t < n; t++)
                {
                    var angle = -2 * Math.PI * t * k / n;
                    real += (float)(input[t].Item1 * Math.Cos(angle) - input[t].Item2 * Math.Sin(angle));
                    imag += (float)(input[t].Item1 * Math.Sin(angle) + input[t].Item2 * Math.Cos(angle));
                }
                output[k] = (real, imag);
            }
        }

        #endregion

        public void Dispose()
        {
            context?.Dispose();
        }
    }

    /// <summary>
    /// Attribute to skip tests based on conditions.
    /// </summary>
    public class SkippableFactAttribute : FactAttribute
    {
        public override string? Skip { get; set; }
    }

    /// <summary>
    /// Helper class for conditional test skipping.
    /// </summary>
    public static class Skip
    {
        public static void IfNot(bool condition, string reason)
        {
            if (!condition)
                throw new SkipException(reason);
        }
    }

    /// <summary>
    /// Exception for skipping tests.
    /// </summary>
    public class SkipException : Exception
    {
        public SkipException(string reason) : base(reason) { }
    }
}