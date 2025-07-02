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

using ILGPU.Algorithms.FFT;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Numerics;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.FFT
{
    /// <summary>
    /// Tests for Fast Fourier Transform algorithms.
    /// </summary>
    public class FFTTests : TestBase
    {
        #region 1D FFT Tests

        [Fact]
        public void TestFFT1D_PowerOfTwo()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 8;
            var fft = new FFT<float>(accelerator!);
            
            // Create test signal: sin wave
            var input = new Complex[size];
            for (int i = 0; i < size; i++)
            {
                input[i] = new Complex(Math.Sin(2 * Math.PI * i / size), 0);
            }
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var outputBuffer = accelerator!.Allocate1D<Complex>(size);
            
            // Forward FFT
            fft.Forward1D(inputBuffer.View, outputBuffer.View);
            
            var result = outputBuffer.GetAsArray1D();
            
            // Verify FFT properties
            Assert.Equal(size, result.Length);
            
            // For a sine wave, most energy should be concentrated in specific frequency bins
            var totalEnergy = 0.0;
            for (int i = 0; i < size; i++)
            {
                totalEnergy += result[i].Magnitude * result[i].Magnitude;
            }
            
            Assert.True(totalEnergy > 0, "FFT should produce non-zero output for sine wave input");
        }

        [Fact]
        public void TestFFT1D_Inverse()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 16;
            var fft = new FFT<float>(accelerator!);
            
            // Create test signal
            var original = new Complex[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
            {
                original[i] = new Complex(random.NextDouble(), random.NextDouble());
            }
            
            using var originalBuffer = accelerator!.Allocate1D(original);
            using var transformBuffer = accelerator!.Allocate1D<Complex>(size);
            using var reconstructedBuffer = accelerator!.Allocate1D<Complex>(size);
            
            // Forward FFT
            fft.Forward1D(originalBuffer.View, transformBuffer.View);
            
            // Inverse FFT
            fft.Inverse1D(transformBuffer.View, reconstructedBuffer.View);
            
            var reconstructed = reconstructedBuffer.GetAsArray1D();
            
            // Verify reconstruction (allowing for floating-point errors)
            for (int i = 0; i < size; i++)
            {
                Assert.True(
                    Math.Abs(original[i].Real - reconstructed[i].Real) < 1e-6,
                    $"Real part mismatch at index {i}");
                Assert.True(
                    Math.Abs(original[i].Imaginary - reconstructed[i].Imaginary) < 1e-6,
                    $"Imaginary part mismatch at index {i}");
            }
        }

        [Theory]
        [InlineData(4)]
        [InlineData(8)]
        [InlineData(16)]
        [InlineData(32)]
        [InlineData(64)]
        public void TestFFT1D_DifferentSizes(int size)
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var fft = new FFT<float>(accelerator!);
            
            // Create impulse signal
            var input = new Complex[size];
            input[0] = new Complex(1, 0); // Impulse at index 0
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var outputBuffer = accelerator!.Allocate1D<Complex>(size);
            
            fft.Forward1D(inputBuffer.View, outputBuffer.View);
            
            var result = outputBuffer.GetAsArray1D();
            
            // FFT of impulse should be all ones (approximately)
            for (int i = 0; i < size; i++)
            {
                Assert.True(
                    Math.Abs(result[i].Real - 1.0) < 1e-6,
                    $"Real part should be 1.0 at index {i}");
                Assert.True(
                    Math.Abs(result[i].Imaginary) < 1e-6,
                    $"Imaginary part should be 0.0 at index {i}");
            }
        }

        #endregion

        #region 2D FFT Tests

        [Fact]
        public void TestFFT2D_Basic()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 8;
            const int height = 8;
            var fft = new FFT<float>(accelerator!);
            
            // Create 2D test pattern
            var input = new Complex[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var value = Math.Sin(2 * Math.PI * x / width) * Math.Cos(2 * Math.PI * y / height);
                    input[y * width + x] = new Complex(value, 0);
                }
            }
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var outputBuffer = accelerator!.Allocate1D<Complex>(width * height);
            
            fft.Forward2D(inputBuffer.View, outputBuffer.View, width, height);
            
            var result = outputBuffer.GetAsArray1D();
            
            // Verify output has correct dimensions
            Assert.Equal(width * height, result.Length);
            
            // Verify energy conservation (Parseval's theorem)
            var inputEnergy = 0.0;
            var outputEnergy = 0.0;
            
            for (int i = 0; i < width * height; i++)
            {
                inputEnergy += input[i].Magnitude * input[i].Magnitude;
                outputEnergy += result[i].Magnitude * result[i].Magnitude;
            }
            
            // Energy should be conserved (within floating-point precision)
            Assert.True(
                Math.Abs(inputEnergy - outputEnergy / (width * height)) < 1e-5,
                "Energy should be conserved in FFT");
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestFFT_Performance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 1024;
            var fft = new FFT<float>(accelerator!);
            
            // Create test data
            var input = new Complex[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
            {
                input[i] = new Complex(random.NextDouble(), random.NextDouble());
            }
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var outputBuffer = accelerator!.Allocate1D<Complex>(size);
            
            // Warm up
            fft.Forward1D(inputBuffer.View, outputBuffer.View);
            accelerator!.Synchronize();
            
            // Measure performance
            var time = MeasureTime(() =>
            {
                fft.Forward1D(inputBuffer.View, outputBuffer.View);
                accelerator!.Synchronize();
            });
            
            // Verify reasonable performance (should complete in reasonable time)
            Assert.True(time < 1000, $"FFT took {time}ms, expected < 1000ms");
            
            // Verify correctness
            var result = outputBuffer.GetAsArray1D();
            Assert.True(result.Length == size, "Output should have correct size");
            
            // Check that output is not all zeros
            var hasNonZero = false;
            for (int i = 0; i < size; i++)
            {
                if (result[i].Magnitude > 1e-10)
                {
                    hasNonZero = true;
                    break;
                }
            }
            Assert.True(hasNonZero, "FFT output should contain non-zero values");
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestFFT_InvalidSize()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var fft = new FFT<float>(accelerator!);
            
            // Test with non-power-of-two size (should handle gracefully)
            const int size = 7; // Not a power of 2
            var input = new Complex[size];
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var outputBuffer = accelerator!.Allocate1D<Complex>(size);
            
            // This should either work (using Bluestein algorithm) or throw meaningful exception
            try
            {
                fft.Forward1D(inputBuffer.View, outputBuffer.View);
                // If it succeeds, verify output
                var result = outputBuffer.GetAsArray1D();
                Assert.Equal(size, result.Length);
            }
            catch (NotSupportedException)
            {
                // Acceptable if non-power-of-two sizes are not supported
            }
            catch (ArgumentException)
            {
                // Also acceptable
            }
        }

        #endregion
    }
}