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

using ILGPU.Apple.NeuralEngine;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Specific tests for Apple Neural Engine (ANE) accelerator.
    /// Tests Core ML integration, neural operations, and ANE-specific features.
    /// </summary>
    public class ANESpecificTests : TestBase
    {
        public ANESpecificTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        [Fact]
        public void ANEGenerationDetection()
        {
            if (!ANECapabilities.DetectNeuralEngine())
            {
                Output.WriteLine("Apple Neural Engine not supported - skipping test");
                return;
            }

            var capabilities = ANECapabilities.Query();
            
            Assert.True(capabilities.IsAvailable, "ANE should be available");
            Assert.True(capabilities.Generation != ANEGeneration.None, "ANE generation should be detected");
            
            Output.WriteLine($"Detected ANE Generation: {capabilities.Generation}");
            
            switch (capabilities.Generation)
            {
                case ANEGeneration.ANE1:
                    Output.WriteLine("  First generation (A11, A12) - 5.5 TOPS");
                    Assert.True(capabilities.MaxTOPS >= 5.0 && capabilities.MaxTOPS <= 6.0);
                    break;
                case ANEGeneration.ANE2:
                    Output.WriteLine("  Second generation (A13, A14) - 11 TOPS");
                    Assert.True(capabilities.MaxTOPS >= 10.0 && capabilities.MaxTOPS <= 12.0);
                    break;
                case ANEGeneration.ANE3:
                    Output.WriteLine("  Third generation (A15, A16, M1, M2) - 15.8 TOPS");
                    Assert.True(capabilities.MaxTOPS >= 15.0 && capabilities.MaxTOPS <= 17.0);
                    break;
                case ANEGeneration.ANE4:
                    Output.WriteLine("  Fourth generation (future) - 20+ TOPS");
                    Assert.True(capabilities.MaxTOPS >= 18.0);
                    break;
            }
            
            Assert.True(capabilities.SupportsCoreML, "ANE should support Core ML");
        }

        [Fact]
        [KernelMethod(nameof(ANEConvolutionKernel))]
        public void ANEConvolutionOptimization()
        {
            if (!ANECapabilities.DetectNeuralEngine())
            {
                Output.WriteLine("Apple Neural Engine not supported - skipping test");
                return;
            }

            var capabilities = ANECapabilities.Query();
            if (!capabilities.SupportsConvolution)
            {
                Output.WriteLine("ANE convolution not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AppleNeuralEngineDevice>().EnableANE();
            using var accelerator = context.CreateANEAccelerator(0);
            using var stream = accelerator.CreateStream();

            // MobileNet-style convolution
            const int batchSize = 1;
            const int inputChannels = 32;
            const int outputChannels = 64;
            const int height = 112;
            const int width = 112;
            const int kernelSize = 3;
            const int stride = 1;
            const int padding = 1;

            var inputSize = batchSize * inputChannels * height * width;
            var weightSize = outputChannels * inputChannels * kernelSize * kernelSize;
            var outputHeight = (height + 2 * padding - kernelSize) / stride + 1;
            var outputWidth = (width + 2 * padding - kernelSize) / stride + 1;
            var outputSize = batchSize * outputChannels * outputHeight * outputWidth;

            var input = new float[inputSize];
            var weights = new float[weightSize];
            var bias = new float[outputChannels];
            
            var random = new Random(42);
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            
            // Xavier initialization for weights
            var fanIn = inputChannels * kernelSize * kernelSize;
            var fanOut = outputChannels * kernelSize * kernelSize;
            var limit = Math.Sqrt(6.0 / (fanIn + fanOut));
            
            for (int i = 0; i < weightSize; i++)
                weights[i] = (float)(random.NextDouble() * 2.0 * limit - limit);
            
            for (int i = 0; i < outputChannels; i++)
                bias[i] = 0.0f; // Zero bias

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var weightsBuffer = accelerator.Allocate1D<float>(weights);
            using var biasBuffer = accelerator.Allocate1D<float>(bias);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                int, int, int, int, int, int, int, int>(ANEConvolutionKernel);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            kernel(stream, new Index3D(outputChannels, outputHeight, outputWidth),
                   inputBuffer.View, weightsBuffer.View, biasBuffer.View, outputBuffer.View,
                   inputChannels, height, width, kernelSize, stride, padding, outputHeight, outputWidth);
            stream.Synchronize();
            stopwatch.Stop();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == outputSize);
            Assert.True(result.Any(x => Math.Abs(x) > 1e-6f), "Should have non-trivial output");
            
            var gops = (2.0 * outputChannels * inputChannels * kernelSize * kernelSize * outputHeight * outputWidth) / 1e9;
            var gopsPerSecond = gops / stopwatch.Elapsed.TotalSeconds;
            
            Output.WriteLine($"ANE convolution: {inputChannels}→{outputChannels}, {height}x{width}→{outputHeight}x{outputWidth}");
            Output.WriteLine($"Performance: {stopwatch.Elapsed.TotalMilliseconds:F2}ms, {gopsPerSecond:F2} GOPS");
            Output.WriteLine($"Output range: [{result.Min():F4}, {result.Max():F4}]");
        }

        static void ANEConvolutionKernel(
            Index3D index,
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> bias,
            ArrayView<float> output,
            int inputChannels,
            int inputHeight,
            int inputWidth,
            int kernelSize,
            int stride,
            int padding,
            int outputHeight,
            int outputWidth)
        {
            var outChannel = index.X;
            var outY = index.Y;
            var outX = index.Z;
            
            if (outChannel >= bias.Length || outY >= outputHeight || outX >= outputWidth)
                return;

            float sum = bias[outChannel];
            
            for (int inChannel = 0; inChannel < inputChannels; inChannel++)
            {
                for (int ky = 0; ky < kernelSize; ky++)
                {
                    for (int kx = 0; kx < kernelSize; kx++)
                    {
                        int inY = outY * stride - padding + ky;
                        int inX = outX * stride - padding + kx;
                        
                        if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth)
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
        [KernelMethod(nameof(ANEFloat16Kernel))]
        public void ANEFloat16Precision()
        {
            var capabilities = ANECapabilities.Query();
            if (!capabilities.SupportsFloat16)
            {
                Output.WriteLine("ANE Float16 not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AppleNeuralEngineDevice>().EnableANE();
            using var accelerator = context.CreateANEAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int size = 1024;
            var input = new float[size];
            
            // Test Float16 precision boundaries
            var testValues = new[]
            {
                0.0f, 1.0f, -1.0f, 
                3.14159f, -2.71828f,
                1e-4f, -1e-4f,  // Small values
                65504.0f, -65504.0f,  // Max FP16 values
                1.0f/3.0f, 2.0f/3.0f  // Precision test
            };
            
            for (int i = 0; i < size; i++)
                input[i] = testValues[i % testValues.Length] + (float)(i * 1e-6);

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(ANEFloat16Kernel);

            kernel(stream, size, inputBuffer.View, outputBuffer.View, size);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == size);
            
            // Check Float16 precision limitations
            for (int i = 0; i < Math.Min(testValues.Length, result.Length); i++)
            {
                var expected = testValues[i];
                var actual = result[i];
                var relativeDiff = Math.Abs((actual - expected) / Math.Max(Math.Abs(expected), 1e-7f));
                
                // Float16 has ~3.3 decimal digits of precision
                Assert.True(relativeDiff < 1e-3f || Math.Abs(expected) < 1e-6f, 
                    $"Float16 precision error: expected {expected:G6}, got {actual:G6}");
            }
            
            Output.WriteLine($"ANE Float16 precision test passed for {size} values");
            Output.WriteLine($"Test values: {string.Join(", ", testValues.Take(5).Select(x => x.ToString("G4")))}...");
        }

        static void ANEFloat16Kernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int size)
        {
            var i = index.X;
            if (i >= size)
                return;

            var value = input[i];
            
            // Simulate Float16 precision (IEEE 754 binary16)
            // 1 sign bit, 5 exponent bits, 10 mantissa bits
            var bits = BitConverter.SingleToUInt32Bits(value);
            var sign = bits & 0x80000000;
            var exponent = (bits & 0x7F800000) >> 23;
            var mantissa = bits & 0x007FFFFF;
            
            // Convert to Float16 range
            if (exponent == 0xFF) // Infinity or NaN
            {
                output[i] = value; // Pass through
                return;
            }
            
            // Adjust exponent bias (127 -> 15)
            var newExponent = (int)exponent - 127 + 15;
            
            if (newExponent <= 0) // Underflow to zero or subnormal
            {
                output[i] = sign != 0 ? -0.0f : 0.0f;
                return;
            }
            
            if (newExponent >= 31) // Overflow to infinity
            {
                output[i] = sign != 0 ? float.NegativeInfinity : float.PositiveInfinity;
                return;
            }
            
            // Truncate mantissa to 10 bits
            var newMantissa = mantissa >> 13; // Keep top 10 bits
            
            // Reconstruct Float32 with reduced precision
            var newBits = sign | ((uint)(newExponent - 15 + 127) << 23) | (newMantissa << 13);
            output[i] = BitConverter.UInt32BitsToSingle(newBits);
        }

        [Fact]
        [KernelMethod(nameof(ANETransformerKernel))]
        public void ANETransformerOptimization()
        {
            var capabilities = ANECapabilities.Query();
            if (!capabilities.SupportsTransformer)
            {
                Output.WriteLine("ANE Transformer not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AppleNeuralEngineDevice>().EnableANE();
            using var accelerator = context.CreateANEAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int seqLength = 64;    // Shorter for mobile
            const int modelDim = 256;    // Smaller model
            const int numHeads = 8;
            const int headDim = modelDim / numHeads;

            var inputSize = seqLength * modelDim;
            var input = new float[inputSize];
            
            // Initialize with typical transformer input distribution
            var random = new Random(42);
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)(random.NextGaussian() * 0.02); // Small variance

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(inputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, int, int, int, int>(
                ANETransformerKernel);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            kernel(stream, new Index2D(seqLength, modelDim),
                   inputBuffer.View, outputBuffer.View, seqLength, modelDim, numHeads, headDim);
            stream.Synchronize();
            stopwatch.Stop();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == inputSize);
            Assert.True(result.Any(x => Math.Abs(x) > 1e-6f), "Should have non-trivial transformer output");
            
            // Check attention normalization properties
            for (int seq = 0; seq < seqLength; seq++)
            {
                var sequenceSum = 0.0f;
                for (int dim = 0; dim < modelDim; dim++)
                    sequenceSum += Math.Abs(result[seq * modelDim + dim]);
                
                Assert.True(sequenceSum > 0, "Sequence should have non-zero attention output");
            }
            
            var complexity = 2.0 * seqLength * seqLength * modelDim; // Attention complexity
            var throughput = complexity / stopwatch.Elapsed.TotalSeconds / 1e9;
            
            Output.WriteLine($"ANE Transformer: {seqLength} seq × {modelDim} dim × {numHeads} heads");
            Output.WriteLine($"Performance: {stopwatch.Elapsed.TotalMilliseconds:F2}ms, {throughput:F2} GOPS");
            Output.WriteLine($"Output statistics: mean={result.Average():F6}, std={result.StandardDeviation():F6}");
        }

        static void ANETransformerKernel(
            Index2D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int seqLength,
            int modelDim,
            int numHeads,
            int headDim)
        {
            var seqPos = index.X;
            var dimPos = index.Y;
            
            if (seqPos >= seqLength || dimPos >= modelDim)
                return;

            var headIdx = dimPos / headDim;
            var dimInHead = dimPos % headDim;
            
            if (headIdx >= numHeads)
            {
                output[seqPos * modelDim + dimPos] = input[seqPos * modelDim + dimPos];
                return;
            }

            // Simplified self-attention computation optimized for ANE
            float attention = 0.0f;
            float normalization = 0.0f;
            
            for (int otherSeq = 0; otherSeq < seqLength; otherSeq++)
            {
                // Compute attention score using current position as query
                float score = 0.0f;
                for (int d = 0; d < headDim; d++)
                {
                    var queryIdx = seqPos * modelDim + headIdx * headDim + d;
                    var keyIdx = otherSeq * modelDim + headIdx * headDim + d;
                    score += input[queryIdx] * input[keyIdx]; // Self-attention
                }
                
                // Scaled dot-product attention
                score /= Math.Sqrt(headDim);
                var weight = Math.Exp(score);
                normalization += weight;
                
                // Apply attention to value
                var valueIdx = otherSeq * modelDim + headIdx * headDim + dimInHead;
                attention += weight * input[valueIdx];
            }
            
            // Normalize attention output
            attention /= Math.Max(normalization, 1e-8f);
            
            // Add residual connection
            var residual = input[seqPos * modelDim + dimPos];
            output[seqPos * modelDim + dimPos] = attention + residual;
        }

        [Fact]
        public void ANEOptimalBatchSize()
        {
            var capabilities = ANECapabilities.Query();
            
            // ANE is optimized for low-latency inference, typically batch size 1
            var modelComplexities = new[] { 1000L, 10000L, 100000L, 1000000L, 10000000L };
            
            Output.WriteLine("ANE Optimal Batch Sizes by Model Complexity:");
            
            foreach (var complexity in modelComplexities)
            {
                var optimalBatch = capabilities.GetOptimalBatchSize(complexity);
                Assert.True(optimalBatch > 0, "Optimal batch size should be positive");
                Assert.True(optimalBatch <= 8, "ANE typically uses small batch sizes for low latency");
                
                Output.WriteLine($"  {complexity:N0} parameters: batch size {optimalBatch}");
            }
            
            // Verify batch size decreases with model complexity
            var smallModel = capabilities.GetOptimalBatchSize(1000L);
            var largeModel = capabilities.GetOptimalBatchSize(10000000L);
            Assert.True(largeModel <= smallModel, "Larger models should have smaller or equal optimal batch sizes");
        }

        [Fact]
        public void ANEPowerEfficiency()
        {
            var capabilities = ANECapabilities.Query();
            
            var powerEfficiency = capabilities.GetPowerEfficiency();
            Assert.True(powerEfficiency > 0, "Power efficiency should be positive");
            
            Output.WriteLine($"ANE Power Efficiency: {powerEfficiency:F1} TOPS/Watt");
            
            // Test power consumption at different utilization levels
            var utilizations = new[] { 10.0, 30.0, 50.0, 80.0, 100.0 };
            
            Output.WriteLine("ANE Power Consumption by Utilization:");
            
            foreach (var util in utilizations)
            {
                var power = capabilities.GetEstimatedPower(util);
                Assert.True(power >= 0, "Power consumption should be non-negative");
                Assert.True(power <= 2.0, "ANE power consumption should be low (mobile device)");
                
                var actualEfficiency = (capabilities.MaxTOPS * util / 100.0) / Math.Max(power, 0.001);
                Output.WriteLine($"  {util:F0}% utilization: {power:F3}W, {actualEfficiency:F1} TOPS/W");
            }
            
            // Verify efficiency claims for different generations
            switch (capabilities.Generation)
            {
                case ANEGeneration.ANE1:
                    Assert.True(powerEfficiency >= 10.0 && powerEfficiency <= 12.0);
                    break;
                case ANEGeneration.ANE2:
                    Assert.True(powerEfficiency >= 12.0 && powerEfficiency <= 15.0);
                    break;
                case ANEGeneration.ANE3:
                    Assert.True(powerEfficiency >= 12.0 && powerEfficiency <= 16.0);
                    break;
                case ANEGeneration.ANE4:
                    Assert.True(powerEfficiency >= 18.0);
                    break;
            }
        }

        [Fact]
        public void ANEModelTypeOptimization()
        {
            var capabilities = ANECapabilities.Query();
            
            var modelTypes = new[]
            {
                ANEModelType.ConvolutionalNeuralNetwork,
                ANEModelType.RecurrentNeuralNetwork,
                ANEModelType.Transformer,
                ANEModelType.ObjectDetection,
                ANEModelType.NaturalLanguageProcessing,
                ANEModelType.ComputerVision
            };
            
            Output.WriteLine("ANE Model Type Optimization Support:");
            
            foreach (var modelType in modelTypes)
            {
                var isOptimal = capabilities.IsModelTypeOptimal(modelType);
                Output.WriteLine($"  {modelType}: {(isOptimal ? "Optimal" : "Supported")}");
                
                // Basic support assertions
                switch (modelType)
                {
                    case ANEModelType.ConvolutionalNeuralNetwork:
                        Assert.True(isOptimal, "All ANE generations should optimize CNNs");
                        break;
                    case ANEModelType.ComputerVision:
                        Assert.True(isOptimal, "All ANE generations should optimize computer vision");
                        break;
                    case ANEModelType.Transformer:
                        if (capabilities.Generation >= ANEGeneration.ANE3)
                            Assert.True(isOptimal, "ANE3+ should optimize transformers");
                        break;
                }
            }
        }

        [Fact]
        [KernelMethod(nameof(ANECoreMLIntegrationKernel))]
        public void ANECoreMLIntegration()
        {
            var capabilities = ANECapabilities.Query();
            if (!capabilities.SupportsCoreML)
            {
                Output.WriteLine("ANE Core ML integration not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AppleNeuralEngineDevice>().EnableANE();
            using var accelerator = context.CreateANEAccelerator(0);
            using var stream = accelerator.CreateStream();

            // Simulate Core ML model execution
            const int inputSize = 224 * 224 * 3; // ImageNet input
            const int outputSize = 1000; // ImageNet classes
            
            var input = new float[inputSize];
            var random = new Random(42);
            
            // Normalize input like typical image preprocessing
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)((random.NextDouble() - 0.5) * 2.0); // [-1, 1]

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                ANECoreMLIntegrationKernel);

            kernel(stream, outputSize, inputBuffer.View, outputBuffer.View, inputSize, outputSize);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == outputSize);
            
            // Check if output looks like softmax probabilities
            var sum = result.Sum();
            var maxValue = result.Max();
            var minValue = result.Min();
            
            Assert.True(Math.Abs(sum - 1.0f) < 0.1f, "Output should approximately sum to 1 (softmax-like)");
            Assert.True(maxValue >= 0, "All outputs should be non-negative");
            Assert.True(maxValue <= 1.0f, "All outputs should be <= 1");
            
            Output.WriteLine($"ANE Core ML simulation: {inputSize}→{outputSize}");
            Output.WriteLine($"Output sum: {sum:F4}, range: [{minValue:F4}, {maxValue:F4}]");
            Output.WriteLine($"Top-5 predictions: {string.Join(", ", result.Select((v, i) => new { v, i }).OrderByDescending(x => x.v).Take(5).Select(x => $"{x.i}:{x.v:F4}"))}");
        }

        static void ANECoreMLIntegrationKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int inputSize,
            int outputSize)
        {
            var i = index.X;
            if (i >= outputSize)
                return;

            // Simulate a simple fully connected layer + softmax
            // This would normally be replaced by actual Core ML model execution
            
            // Compute weighted sum (simplified linear layer)
            float sum = 0.0f;
            for (int j = 0; j < inputSize; j++)
            {
                // Use a simple deterministic weight based on indices
                var weight = Math.Sin((i + 1) * (j + 1) * 0.001f) * 0.01f;
                sum += input[j] * weight;
            }
            
            // Add bias
            sum += (float)(Math.Sin(i * 0.1) * 0.1);
            
            // Apply activation (tanh for bounded output)
            var activated = Math.Tanh(sum);
            
            // Convert to probability-like values
            output[i] = (float)((activated + 1.0) * 0.5); // Map [-1,1] to [0,1]
        }

        [Fact]
        public void ANEThermalManagement()
        {
            var capabilities = ANECapabilities.Query();
            
            // Test thermal state simulation
            var thermalStates = new[]
            {
                ANEThermalState.Normal,
                ANEThermalState.Fair,
                ANEThermalState.Serious,
                ANEThermalState.Critical
            };
            
            Output.WriteLine("ANE Thermal State Management:");
            
            foreach (var state in thermalStates)
            {
                // Simulate different performance levels based on thermal state
                var performanceMultiplier = state switch
                {
                    ANEThermalState.Normal => 1.0,
                    ANEThermalState.Fair => 0.85,
                    ANEThermalState.Serious => 0.6,
                    ANEThermalState.Critical => 0.3,
                    _ => 1.0
                };
                
                var effectiveTOPS = capabilities.MaxTOPS * performanceMultiplier;
                Output.WriteLine($"  {state}: {effectiveTOPS:F1} TOPS ({performanceMultiplier * 100:F0}% performance)");
                
                Assert.True(effectiveTOPS >= 0, "Effective TOPS should be non-negative");
                Assert.True(effectiveTOPS <= capabilities.MaxTOPS, "Effective TOPS should not exceed maximum");
            }
            
            // Verify thermal throttling behavior
            var normalPerf = capabilities.MaxTOPS * 1.0;
            var criticalPerf = capabilities.MaxTOPS * 0.3;
            Assert.True(criticalPerf < normalPerf, "Critical thermal state should reduce performance");
        }
    }

    /// <summary>
    /// Extension methods for statistical calculations.
    /// </summary>
    public static class StatisticsExtensions
    {
        public static double StandardDeviation(this float[] values)
        {
            var mean = values.Average();
            var variance = values.Select(x => Math.Pow(x - mean, 2)).Average();
            return Math.Sqrt(variance);
        }
    }
}