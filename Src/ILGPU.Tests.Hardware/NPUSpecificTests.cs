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

using ILGPU.Intel.NPU;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Specific tests for Intel NPU (Neural Processing Unit) accelerator.
    /// Tests AI workloads, quantization, and NPU-specific features.
    /// </summary>
    public class NPUSpecificTests : TestBase
    {
        public NPUSpecificTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        [Fact]
        public void NPUGenerationDetection()
        {
            if (!NPUCapabilities.DetectNPU())
            {
                Output.WriteLine("Intel NPU not supported - skipping test");
                return;
            }

            var capabilities = NPUCapabilities.Query();
            
            Assert.True(capabilities.Generation != NPUGeneration.None, "NPU generation should be detected");
            Assert.True(capabilities.Generation != NPUGeneration.Unknown, "NPU generation should be known");
            
            Output.WriteLine($"Detected NPU Generation: {capabilities.Generation}");
            
            switch (capabilities.Generation)
            {
                case NPUGeneration.NPU2:
                    Output.WriteLine("  Meteor Lake NPU (10 TOPS)");
                    Assert.True(capabilities.MaxTOPS >= 8.0 && capabilities.MaxTOPS <= 12.0);
                    break;
                case NPUGeneration.NPU3:
                    Output.WriteLine("  Lunar Lake NPU (40 TOPS)");
                    Assert.True(capabilities.MaxTOPS >= 35.0 && capabilities.MaxTOPS <= 45.0);
                    break;
                case NPUGeneration.NPU4:
                    Output.WriteLine("  Arrow Lake NPU (45+ TOPS)");
                    Assert.True(capabilities.MaxTOPS >= 40.0);
                    break;
            }
        }

        [Fact]
        [KernelMethod(nameof(NPUConvolutionKernel))]
        public void NPUConvolutionOperation()
        {
            if (!NPUCapabilities.DetectNPU())
            {
                Output.WriteLine("Intel NPU not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<IntelNPUDevice>().EnableNPU();
            using var accelerator = context.CreateNPUAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int batchSize = 1;
            const int inputChannels = 3;
            const int outputChannels = 16;
            const int height = 32;
            const int width = 32;
            const int kernelSize = 3;

            var inputSize = batchSize * inputChannels * height * width;
            var weightSize = outputChannels * inputChannels * kernelSize * kernelSize;
            var outputHeight = height - kernelSize + 1; // No padding
            var outputWidth = width - kernelSize + 1;
            var outputSize = batchSize * outputChannels * outputHeight * outputWidth;

            var input = new float[inputSize];
            var weights = new float[weightSize];
            var bias = new float[outputChannels];
            
            // Initialize with realistic values
            var random = new Random(42);
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)(random.NextDouble() * 2.0 - 1.0); // [-1, 1]
            
            for (int i = 0; i < weightSize; i++)
                weights[i] = (float)(random.NextGaussian() * 0.1); // Small weights
            
            for (int i = 0; i < outputChannels; i++)
                bias[i] = (float)(random.NextDouble() * 0.1);

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var weightsBuffer = accelerator.Allocate1D<float>(weights);
            using var biasBuffer = accelerator.Allocate1D<float>(bias);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                int, int, int, int, int, int>(NPUConvolutionKernel);

            kernel(stream, new Index3D(outputChannels, outputHeight, outputWidth),
                   inputBuffer.View, weightsBuffer.View, biasBuffer.View, outputBuffer.View,
                   inputChannels, height, width, kernelSize, outputHeight, outputWidth);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == outputSize);
            Assert.True(result.Any(x => Math.Abs(x) > 1e-6f), "Should have non-trivial output");
            
            Output.WriteLine($"NPU convolution: {inputChannels}→{outputChannels} channels, {height}x{width}→{outputHeight}x{outputWidth}");
            Output.WriteLine($"Output range: [{result.Min():F4}, {result.Max():F4}]");
        }

        static void NPUConvolutionKernel(
            Index3D index,
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> bias,
            ArrayView<float> output,
            int inputChannels,
            int inputHeight,
            int inputWidth,
            int kernelSize,
            int outputHeight,
            int outputWidth)
        {
            var outChannel = index.X;
            var outY = index.Y;
            var outX = index.Z;
            
            if (outChannel >= weights.Length / (inputChannels * kernelSize * kernelSize) ||
                outY >= outputHeight || outX >= outputWidth)
                return;

            float sum = bias[outChannel];
            
            for (int inChannel = 0; inChannel < inputChannels; inChannel++)
            {
                for (int ky = 0; ky < kernelSize; ky++)
                {
                    for (int kx = 0; kx < kernelSize; kx++)
                    {
                        int inY = outY + ky;
                        int inX = outX + kx;
                        
                        if (inY < inputHeight && inX < inputWidth)
                        {
                            var inputIdx = inChannel * inputHeight * inputWidth + inY * inputWidth + inX;
                            var weightIdx = outChannel * inputChannels * kernelSize * kernelSize +
                                          inChannel * kernelSize * kernelSize + ky * kernelSize + kx;
                            
                            sum += input[inputIdx] * weights[weightIdx];
                        }
                    }
                }
            }
            
            var outputIdx = outChannel * outputHeight * outputWidth + outY * outputWidth + outX;
            output[outputIdx] = Math.Max(0.0f, sum); // ReLU activation
        }

        [Fact]
        [KernelMethod(nameof(NPUQuantizedInferenceKernel))]
        public void NPUQuantizedInference()
        {
            var capabilities = NPUCapabilities.Query();
            if (!capabilities.SupportsInt8)
            {
                Output.WriteLine("NPU Int8 quantization not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<IntelNPUDevice>().EnableNPU();
            using var accelerator = context.CreateNPUAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int inputSize = 224 * 224 * 3; // Typical image
            const int outputSize = 1000; // ImageNet classes
            
            // Simulate quantized weights (Int8)
            var quantizedWeights = new sbyte[outputSize * inputSize];
            var scales = new float[outputSize];
            var zeroPoints = new sbyte[outputSize];
            
            var random = new Random(42);
            for (int i = 0; i < quantizedWeights.Length; i++)
                quantizedWeights[i] = (sbyte)random.Next(-128, 128);
            
            for (int i = 0; i < outputSize; i++)
            {
                scales[i] = (float)(random.NextDouble() * 0.01 + 0.001); // Small scales
                zeroPoints[i] = (sbyte)random.Next(-10, 10);
            }

            var input = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)random.NextDouble();

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var weightsBuffer = accelerator.Allocate1D<sbyte>(quantizedWeights);
            using var scalesBuffer = accelerator.Allocate1D<float>(scales);
            using var zeroPointsBuffer = accelerator.Allocate1D<sbyte>(zeroPoints);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<sbyte>, ArrayView<float>, ArrayView<sbyte>,
                ArrayView<float>, int, int>(NPUQuantizedInferenceKernel);

            kernel(stream, outputSize, inputBuffer.View, weightsBuffer.View, scalesBuffer.View,
                   zeroPointsBuffer.View, outputBuffer.View, inputSize, outputSize);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == outputSize);
            Assert.True(result.Any(x => x != 0), "Should have non-zero outputs");
            
            Output.WriteLine($"NPU quantized inference: Int8 weights, {inputSize}→{outputSize}");
            Output.WriteLine($"Output range: [{result.Min():F4}, {result.Max():F4}]");
        }

        static void NPUQuantizedInferenceKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<sbyte> quantizedWeights,
            ArrayView<float> scales,
            ArrayView<sbyte> zeroPoints,
            ArrayView<float> output,
            int inputSize,
            int outputSize)
        {
            var i = index.X;
            if (i >= outputSize)
                return;

            int sum = 0;
            for (int j = 0; j < inputSize; j++)
            {
                var quantizedInput = (sbyte)Math.Clamp((int)(input[j] * 127.0f), -128, 127);
                sum += quantizedInput * quantizedWeights[i * inputSize + j];
            }
            
            // Dequantize: (quantized - zero_point) * scale
            output[i] = (sum - zeroPoints[i]) * scales[i];
        }

        [Fact]
        [KernelMethod(nameof(NPUAttentionKernel))]
        public void NPUAttentionMechanism()
        {
            var capabilities = NPUCapabilities.Query();
            if (!capabilities.SupportsAttention)
            {
                Output.WriteLine("NPU attention not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<IntelNPUDevice>().EnableNPU();
            using var accelerator = context.CreateNPUAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int seqLength = 128;
            const int hiddenSize = 512;
            const int numHeads = 8;
            const int headDim = hiddenSize / numHeads;

            var querySize = seqLength * hiddenSize;
            var query = new float[querySize];
            var key = new float[querySize];
            var value = new float[querySize];
            
            var random = new Random(42);
            for (int i = 0; i < querySize; i++)
            {
                query[i] = (float)(random.NextGaussian() * 0.1);
                key[i] = (float)(random.NextGaussian() * 0.1);
                value[i] = (float)(random.NextGaussian() * 0.1);
            }

            using var queryBuffer = accelerator.Allocate1D<float>(query);
            using var keyBuffer = accelerator.Allocate1D<float>(key);
            using var valueBuffer = accelerator.Allocate1D<float>(value);
            using var outputBuffer = accelerator.Allocate1D<float>(querySize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                int, int, int, int>(NPUAttentionKernel);

            kernel(stream, new Index2D(seqLength, hiddenSize),
                   queryBuffer.View, keyBuffer.View, valueBuffer.View, outputBuffer.View,
                   seqLength, hiddenSize, numHeads, headDim);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == querySize);
            Assert.True(result.Any(x => Math.Abs(x) > 1e-6f), "Should have non-trivial attention output");
            
            Output.WriteLine($"NPU attention: {seqLength} seq × {hiddenSize} hidden × {numHeads} heads");
            Output.WriteLine($"Output range: [{result.Min():F4}, {result.Max():F4}]");
        }

        static void NPUAttentionKernel(
            Index2D index,
            ArrayView<float> query,
            ArrayView<float> key,
            ArrayView<float> value,
            ArrayView<float> output,
            int seqLength,
            int hiddenSize,
            int numHeads,
            int headDim)
        {
            var seqPos = index.X;
            var hiddenPos = index.Y;
            
            if (seqPos >= seqLength || hiddenPos >= hiddenSize)
                return;

            var headIdx = hiddenPos / headDim;
            var dimIdx = hiddenPos % headDim;
            
            if (headIdx >= numHeads)
                return;

            // Simplified attention computation
            float sum = 0.0f;
            float normalization = 0.0f;
            
            for (int otherSeq = 0; otherSeq < seqLength; otherSeq++)
            {
                // Compute attention score (simplified dot product)
                float score = 0.0f;
                for (int d = 0; d < headDim; d++)
                {
                    var queryIdx = seqPos * hiddenSize + headIdx * headDim + d;
                    var keyIdx = otherSeq * hiddenSize + headIdx * headDim + d;
                    score += query[queryIdx] * key[keyIdx];
                }
                
                // Apply softmax (simplified)
                var weight = Math.Exp(score / Math.Sqrt(headDim));
                normalization += weight;
                
                // Accumulate weighted value
                var valueIdx = otherSeq * hiddenSize + headIdx * headDim + dimIdx;
                sum += weight * value[valueIdx];
            }
            
            output[seqPos * hiddenSize + hiddenPos] = sum / Math.Max(normalization, 1e-8f);
        }

        [Fact]
        public void NPUModelFormatSupport()
        {
            var capabilities = NPUCapabilities.Query();
            
            Assert.True(capabilities.SupportsModelFormat(ModelFormat.ONNX), "All NPUs should support ONNX");
            Assert.True(capabilities.SupportsModelFormat(ModelFormat.OpenVINO), "All NPUs should support OpenVINO");
            
            Output.WriteLine("NPU Model Format Support:");
            Output.WriteLine($"  ONNX: {capabilities.SupportsModelFormat(ModelFormat.ONNX)}");
            Output.WriteLine($"  OpenVINO: {capabilities.SupportsModelFormat(ModelFormat.OpenVINO)}");
            Output.WriteLine($"  TensorFlow: {capabilities.SupportsModelFormat(ModelFormat.TensorFlow)}");
            Output.WriteLine($"  PyTorch: {capabilities.SupportsModelFormat(ModelFormat.PyTorch)}");
            
            // Advanced formats should be supported on newer NPUs
            if (capabilities.Generation >= NPUGeneration.NPU3)
            {
                Assert.True(capabilities.SupportsModelFormat(ModelFormat.TensorFlow));
                Assert.True(capabilities.SupportsModelFormat(ModelFormat.PyTorch));
            }
        }

        [Fact]
        public void NPUBatchSizeOptimization()
        {
            var capabilities = NPUCapabilities.Query();
            
            // Test optimal batch sizes for different model sizes
            var modelSizes = new[] { 10.0, 50.0, 100.0, 500.0, 1000.0 }; // MB
            
            Output.WriteLine("NPU Optimal Batch Sizes:");
            
            foreach (var modelSize in modelSizes)
            {
                var optimalBatch = capabilities.GetOptimalBatchSize(modelSize);
                Assert.True(optimalBatch > 0, "Optimal batch size should be positive");
                
                Output.WriteLine($"  {modelSize:F0}MB model: batch size {optimalBatch}");
                
                // Larger models should generally have smaller optimal batch sizes
                if (modelSize > 100.0)
                    Assert.True(optimalBatch <= 8, "Large models should have smaller batch sizes");
            }
        }

        [Fact]
        public void NPUPowerAndThermal()
        {
            var capabilities = NPUCapabilities.Query();
            
            // Test power estimation at different utilization levels
            var utilizations = new[] { 10.0, 25.0, 50.0, 75.0, 100.0 };
            
            Output.WriteLine("NPU Power Consumption Estimates:");
            
            foreach (var util in utilizations)
            {
                var power = capabilities.GetEstimatedPower(util);
                Assert.True(power >= 0, "Power consumption should be non-negative");
                
                Output.WriteLine($"  {util:F0}% utilization: {power:F2}W");
            }
            
            // Power should increase with utilization
            var lowPower = capabilities.GetEstimatedPower(25.0);
            var highPower = capabilities.GetEstimatedPower(100.0);
            Assert.True(highPower >= lowPower, "Higher utilization should consume more power");
            
            Output.WriteLine($"NPU Generation: {capabilities.Generation}");
            Output.WriteLine($"Max TOPS: {capabilities.MaxTOPS:F1}");
            Output.WriteLine($"Memory Bandwidth: {capabilities.MemoryBandwidth:F1} GB/s");
        }

        [Fact]
        [KernelMethod(nameof(NPUSparsityKernel))]
        public void NPUSparsitySupport()
        {
            var capabilities = NPUCapabilities.Query();
            if (!capabilities.SupportsSparsity)
            {
                Output.WriteLine("NPU sparsity not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<IntelNPUDevice>().EnableNPU();
            using var accelerator = context.CreateNPUAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int matrixSize = 512;
            const float sparsityRatio = 0.9f; // 90% sparse
            
            var matrix = new float[matrixSize * matrixSize];
            var indices = new int[matrixSize * matrixSize];
            var values = new float[matrixSize * matrixSize];
            var vector = new float[matrixSize];
            
            var random = new Random(42);
            int nonZeroCount = 0;
            
            // Create sparse matrix
            for (int i = 0; i < matrixSize * matrixSize; i++)
            {
                if (random.NextDouble() > sparsityRatio)
                {
                    var value = (float)(random.NextGaussian() * 0.1);
                    matrix[i] = value;
                    indices[nonZeroCount] = i;
                    values[nonZeroCount] = value;
                    nonZeroCount++;
                }
            }
            
            // Initialize input vector
            for (int i = 0; i < matrixSize; i++)
                vector[i] = (float)random.NextDouble();

            using var matrixBuffer = accelerator.Allocate1D<float>(matrix);
            using var indicesBuffer = accelerator.Allocate1D<int>(indices.AsSpan(0, nonZeroCount).ToArray());
            using var valuesBuffer = accelerator.Allocate1D<float>(values.AsSpan(0, nonZeroCount).ToArray());
            using var vectorBuffer = accelerator.Allocate1D<float>(vector);
            using var outputBuffer = accelerator.Allocate1D<float>(matrixSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                int, int, int>(NPUSparsityKernel);

            kernel(stream, matrixSize, indicesBuffer.View, valuesBuffer.View, vectorBuffer.View,
                   outputBuffer.View, matrixSize, nonZeroCount, matrixSize);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == matrixSize);
            
            Output.WriteLine($"NPU sparse matrix multiplication: {matrixSize}x{matrixSize}, {sparsityRatio*100:F1}% sparse");
            Output.WriteLine($"Non-zero elements: {nonZeroCount}/{matrixSize * matrixSize}");
            Output.WriteLine($"Compression ratio: {(float)(matrixSize * matrixSize) / nonZeroCount:F1}x");
        }

        static void NPUSparsityKernel(
            Index1D index,
            ArrayView<int> sparseIndices,
            ArrayView<float> sparseValues,
            ArrayView<float> vector,
            ArrayView<float> output,
            int matrixSize,
            int nonZeroCount,
            int vectorSize)
        {
            var row = index.X;
            if (row >= matrixSize)
                return;

            float sum = 0.0f;
            
            // Process sparse matrix row
            for (int i = 0; i < nonZeroCount; i++)
            {
                var globalIdx = sparseIndices[i];
                var matrixRow = globalIdx / matrixSize;
                var matrixCol = globalIdx % matrixSize;
                
                if (matrixRow == row && matrixCol < vectorSize)
                {
                    sum += sparseValues[i] * vector[matrixCol];
                }
            }
            
            output[row] = sum;
        }
    }

    /// <summary>
    /// Extension methods for Random to generate Gaussian distributed numbers.
    /// </summary>
    public static class RandomExtensions
    {
        public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
        {
            // Box-Muller transform
            var u1 = 1.0 - random.NextDouble();
            var u2 = 1.0 - random.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}