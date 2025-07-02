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

using ILGPU.FFT;
using ILGPU.Runtime;
using System;
using System.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.CPU
{
    public class FFTTests : TestBase
    {
        public FFTTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        private static readonly int[] TestSizes = { 8, 16, 32, 64, 128 };

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void FFT1DComplexTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator); // CPU fallback

            foreach (var size in TestSizes)
            {
                if (!fftAccelerator.IsSizeSupported(size))
                    continue;

                // Create test data: simple sinusoid
                var inputData = new Complex[size];
                for (int i = 0; i < size; i++)
                {
                    inputData[i] = new Complex(Math.Sin(2 * Math.PI * i / size), 0);
                }

                using var inputBuffer = accelerator.Allocate1D<Complex>(size);
                using var outputBuffer = accelerator.Allocate1D<Complex>(size);

                inputBuffer.CopyFromCPU(inputData);
                
                // Perform FFT
                fftAccelerator.FFT1D(inputBuffer.View, outputBuffer.View);
                
                var result = outputBuffer.GetAsArray();
                
                // Verify result: should have peak at frequency bin 1
                Assert.True(result.Length == size);
                Assert.True(result[1].Magnitude > result[0].Magnitude);
                Assert.True(result[1].Magnitude > result[2].Magnitude);
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void FFT1DRealTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator);

            foreach (var size in TestSizes)
            {
                if (!fftAccelerator.IsSizeSupported(size))
                    continue;

                // Create real test data
                var inputData = new float[size];
                for (int i = 0; i < size; i++)
                {
                    inputData[i] = (float)Math.Sin(2 * Math.PI * i / size);
                }

                using var inputBuffer = accelerator.Allocate1D<float>(size);
                using var outputBuffer = accelerator.Allocate1D<Complex>(size / 2 + 1);

                inputBuffer.CopyFromCPU(inputData);
                
                // Perform real FFT
                fftAccelerator.FFT1DReal(inputBuffer.View, outputBuffer.View);
                
                var result = outputBuffer.GetAsArray();
                
                // Verify result size
                Assert.True(result.Length == size / 2 + 1);
                Assert.True(result[1].Magnitude > result[0].Magnitude);
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void IFFT1DRealTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator);

            foreach (var size in TestSizes)
            {
                if (!fftAccelerator.IsSizeSupported(size))
                    continue;

                // Create frequency domain data
                var inputData = new Complex[size / 2 + 1];
                inputData[0] = new Complex(0, 0);
                inputData[1] = new Complex(size / 2, 0); // Peak at frequency 1
                for (int i = 2; i < inputData.Length; i++)
                {
                    inputData[i] = new Complex(0, 0);
                }

                using var inputBuffer = accelerator.Allocate1D<Complex>(inputData.Length);
                using var outputBuffer = accelerator.Allocate1D<float>(size);

                inputBuffer.CopyFromCPU(inputData);
                
                // Perform inverse real FFT
                fftAccelerator.IFFT1DReal(inputBuffer.View, outputBuffer.View);
                
                var result = outputBuffer.GetAsArray();
                
                // Verify result size and basic properties
                Assert.True(result.Length == size);
                
                // Result should be approximately sinusoidal
                var maxValue = float.MinValue;
                for (int i = 0; i < result.Length; i++)
                {
                    maxValue = Math.Max(maxValue, Math.Abs(result[i]));
                }
                Assert.True(maxValue > 0.1f); // Should have significant amplitude
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void FFT2DTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator);

            var size = 8; // Small 2D FFT
            if (!fftAccelerator.IsSizeSupported(size))
                return;

            // Create 2D test data
            var inputData = new Complex[size * size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    // Simple pattern: impulse at center
                    inputData[i * size + j] = (i == size / 2 && j == size / 2) ? 
                        new Complex(1, 0) : new Complex(0, 0);
                }
            }

            using var inputBuffer = accelerator.Allocate2DDenseX<Complex>(new Index2D(size, size));
            using var outputBuffer = accelerator.Allocate2DDenseX<Complex>(new Index2D(size, size));

            inputBuffer.CopyFromCPU(inputData);
            
            // Perform 2D FFT
            fftAccelerator.FFT2D(inputBuffer.View, outputBuffer.View);
            
            var result = outputBuffer.GetAsArray();
            
            // Verify result size
            Assert.True(result.Length == size * size);
            
            // All frequency components should have equal magnitude for center impulse
            var expectedMagnitude = result[0].Magnitude;
            for (int i = 1; i < result.Length; i++)
            {
                Assert.True(Math.Abs(result[i].Magnitude - expectedMagnitude) < 0.1,
                    $"Expected magnitude {expectedMagnitude}, got {result[i].Magnitude}");
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void FFTBatchTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator);

            var size = 16;
            var batchCount = 4;
            
            if (!fftAccelerator.IsSizeSupported(size))
                return;

            // Create batch test data
            var inputs = new ArrayView<Complex>[batchCount];
            var outputs = new ArrayView<Complex>[batchCount];
            var buffers = new MemoryBuffer1D<Complex, Stride1D.Dense>[batchCount * 2];

            try
            {
                for (int b = 0; b < batchCount; b++)
                {
                    var inputData = new Complex[size];
                    for (int i = 0; i < size; i++)
                    {
                        // Different frequency for each batch
                        inputData[i] = new Complex(Math.Sin(2 * Math.PI * (b + 1) * i / size), 0);
                    }

                    buffers[b * 2] = accelerator.Allocate1D<Complex>(size);
                    buffers[b * 2 + 1] = accelerator.Allocate1D<Complex>(size);
                    
                    buffers[b * 2].CopyFromCPU(inputData);
                    inputs[b] = buffers[b * 2].View;
                    outputs[b] = buffers[b * 2 + 1].View;
                }

                // Perform batch FFT
                fftAccelerator.FFTBatch(inputs, outputs);

                // Verify each batch result
                for (int b = 0; b < batchCount; b++)
                {
                    var result = buffers[b * 2 + 1].GetAsArray();
                    
                    // Each batch should have peak at different frequency
                    var peakIndex = b + 1;
                    if (peakIndex < result.Length)
                    {
                        Assert.True(result[peakIndex].Magnitude > result[0].Magnitude);
                    }
                }
            }
            finally
            {
                foreach (var buffer in buffers)
                {
                    buffer?.Dispose();
                }
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void IPPFFTTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new IPPFFTAccelerator(accelerator);

            var size = 32;
            if (!fftAccelerator.IsSizeSupported(size))
                return;

            // Test Intel IPP FFT implementation
            var inputData = new Complex[size];
            for (int i = 0; i < size; i++)
            {
                inputData[i] = new Complex(Math.Cos(2 * Math.PI * 2 * i / size), 0);
            }

            using var inputBuffer = accelerator.Allocate1D<Complex>(size);
            using var outputBuffer = accelerator.Allocate1D<Complex>(size);

            inputBuffer.CopyFromCPU(inputData);
            
            // Perform IPP FFT
            fftAccelerator.FFT1D(inputBuffer.View, outputBuffer.View);
            
            var result = outputBuffer.GetAsArray();
            
            // Verify result
            Assert.True(result.Length == size);
            Assert.True(result[2].Magnitude > result[0].Magnitude); // Peak at frequency 2
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void FFTRoundTripTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);
            var fftAccelerator = new CudaFFTAccelerator(accelerator);

            var size = 64;
            if (!fftAccelerator.IsSizeSupported(size))
                return;

            // Test FFT -> IFFT round trip
            var originalData = new float[size];
            for (int i = 0; i < size; i++)
            {
                originalData[i] = (float)(Math.Sin(2 * Math.PI * i / size) + 
                                  0.5 * Math.Cos(2 * Math.PI * 3 * i / size));
            }

            using var realBuffer = accelerator.Allocate1D<float>(size);
            using var complexBuffer = accelerator.Allocate1D<Complex>(size / 2 + 1);
            using var resultBuffer = accelerator.Allocate1D<float>(size);

            realBuffer.CopyFromCPU(originalData);
            
            // Forward FFT
            fftAccelerator.FFT1DReal(realBuffer.View, complexBuffer.View);
            
            // Inverse FFT
            fftAccelerator.IFFT1DReal(complexBuffer.View, resultBuffer.View);
            
            var result = resultBuffer.GetAsArray();
            
            // Verify round-trip accuracy
            for (int i = 0; i < size; i++)
            {
                var error = Math.Abs(result[i] - originalData[i]);
                Assert.True(error < 0.1f, 
                    $"Round-trip error too large at index {i}: {error}");
            }
        }
    }
}