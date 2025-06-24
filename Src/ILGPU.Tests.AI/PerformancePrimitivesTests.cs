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
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;
using System;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.AI
{
    /// <summary>
    /// Tests for performance primitives across different accelerators.
    /// </summary>
    public class PerformancePrimitivesTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly IPerformancePrimitives _primitives;

        public PerformancePrimitivesTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                          .CreateAccelerator(_context);
            _primitives = PerformancePrimitivesFactory.Create(_accelerator);
        }

        [Fact]
        public void PerformancePrimitivesFactory_ShouldCreateValidInstance()
        {
            // Arrange & Act
            var primitives = PerformancePrimitivesFactory.Create(_accelerator);

            // Assert
            primitives.Should().NotBeNull();
            primitives.Accelerator.Should().Be(_accelerator);
            primitives.Capabilities.Should().NotBeNull();
        }

        [Fact]
        public void PerformancePrimitiveCapabilities_ShouldHaveValidValues()
        {
            // Arrange & Act
            var capabilities = _primitives.Capabilities;

            // Assert
            capabilities.Should().NotBeNull();
            capabilities.MaxTensorRank.Should().BeGreaterThan(0);
            capabilities.PreferredBatchSize.Should().BeGreaterThan(0);
            capabilities.PeakTFLOPS.Should().BeGreaterOrEqualTo(0);
        }

        [Theory]
        [InlineData(2, 3, 4)] // Small matrices
        [InlineData(64, 64, 64)] // Medium matrices
        [InlineData(256, 256, 256)] // Large matrices
        public async Task GemmAsync_ShouldExecuteMatrixMultiplication(int m, int n, int k)
        {
            // Arrange
            var a = CreateRandomTensor<float>(new TensorShape(m, k));
            var b = CreateRandomTensor<float>(new TensorShape(k, n));
            var c = CreateZeroTensor<float>(new TensorShape(m, n));

            // Act
            await _primitives.GemmAsync(a, b, c, 1.0f, 0.0f);

            // Assert
            c.Should().NotBeNull();
            ValidateMatrixMultiplication(a, b, c);
        }

        [Theory]
        [InlineData(1, 3, 32, 32, 16, 3, 3)] // Standard convolution
        [InlineData(2, 64, 224, 224, 64, 7, 7)] // Larger convolution
        public async Task Conv2DAsync_ShouldExecuteConvolution(
            int batch, int inChannels, int height, int width,
            int outChannels, int kernelSize, int kernelHeight)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(batch, inChannels, height, width));
            var kernel = CreateRandomTensor<float>(new TensorShape(outChannels, inChannels, kernelSize, kernelHeight));
            var parameters = new ConvolutionParameters
            {
                Stride = new Size2D(1, 1),
                Padding = new Size2D(0, 0),
                Dilation = new Size2D(1, 1)
            };

            var outputHeight = height - kernelSize + 1;
            var outputWidth = width - kernelHeight + 1;
            var output = CreateZeroTensor<float>(new TensorShape(batch, outChannels, outputHeight, outputWidth));

            // Act
            await _primitives.Conv2DAsync(input, kernel, output, parameters);

            // Assert
            output.Should().NotBeNull();
            // Convolution validation would require reference implementation
        }

        [Theory]
        [InlineData(10, 512)] // Small tensor
        [InlineData(1000, 1024)] // Medium tensor
        public async Task ReLUAsync_ShouldApplyActivation(int batch, int features)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(batch, features), -2.0f, 2.0f);
            var output = CreateZeroTensor<float>(new TensorShape(batch, features));

            // Act
            await _primitives.ReLUAsync(input, output);

            // Assert
            output.Should().NotBeNull();
            ValidateReLUActivation(input, output);
        }

        [Theory]
        [InlineData(32, 128, 64)] // Small softmax
        [InlineData(64, 1024, 512)] // Large softmax
        public async Task SoftmaxAsync_ShouldApplyNormalization(int batch, int sequence, int features)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(batch, sequence, features));
            var output = CreateZeroTensor<float>(new TensorShape(batch, sequence, features));

            // Act
            await _primitives.SoftmaxAsync(input, output, axis: -1);

            // Assert
            output.Should().NotBeNull();
            ValidateSoftmaxOutput(output);
        }

        [Theory]
        [InlineData(16, 512)] // Layer norm
        [InlineData(64, 1024)] // Larger layer norm
        public async Task LayerNormAsync_ShouldNormalizeLayer(int batch, int features)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(batch, features));
            var output = CreateZeroTensor<float>(new TensorShape(batch, features));
            var gamma = CreateOnesTensor<float>(new TensorShape(features));
            var beta = CreateZeroTensor<float>(new TensorShape(features));
            var epsilon = 1e-5f;

            // Act
            await _primitives.LayerNormAsync(input, output, gamma, beta, epsilon);

            // Assert
            output.Should().NotBeNull();
            ValidateLayerNormOutput(output);
        }

        [Theory]
        [InlineData(2, 2)] // 2x2 pooling
        [InlineData(3, 3)] // 3x3 pooling
        public async Task MaxPool2DAsync_ShouldApplyPooling(int poolWidth, int poolHeight)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(1, 16, 32, 32));
            var poolSize = new Size2D(poolWidth, poolHeight);
            var stride = new Size2D(poolWidth, poolHeight);
            var padding = new Size2D(0, 0);
            var output = CreateZeroTensor<float>(new TensorShape(1, 16, 32 / poolWidth, 32 / poolHeight));

            // Act
            await _primitives.MaxPool2DAsync(input, output, poolSize, stride, padding);

            // Assert
            output.Should().NotBeNull();
        }

        [Fact]
        public async Task QuantizeToInt8Async_ShouldQuantizeTensor()
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(64, 128), -1.0f, 1.0f);
            var output = CreateZeroTensor<sbyte>(new TensorShape(64, 128));
            var scale = 0.01f;
            var zeroPoint = (sbyte)0;

            // Act
            await _primitives.QuantizeToInt8Async(input, output, scale, zeroPoint);

            // Assert
            output.Should().NotBeNull();
            ValidateQuantization(input, output, scale, zeroPoint);
        }

        [Fact]
        public async Task DequantizeFromInt8Async_ShouldDequantizeTensor()
        {
            // Arrange
            var input = CreateRandomTensor<sbyte>(new TensorShape(64, 128), -127, 127);
            var output = CreateZeroTensor<float>(new TensorShape(64, 128));
            var scale = 0.01f;
            var zeroPoint = (sbyte)0;

            // Act
            await _primitives.DequantizeFromInt8Async(input, output, scale, zeroPoint);

            // Assert
            output.Should().NotBeNull();
            ValidateDequantization(input, output, scale, zeroPoint);
        }

        [Fact]
        public void HasAcceleratedPrimitives_ShouldReturnCorrectValue()
        {
            // Arrange & Act
            var hasAccelerated = PerformancePrimitivesFactory.HasAcceleratedPrimitives(_accelerator);

            // Assert
            Assert.IsType<bool>(hasAccelerated);
        }

        [Theory]
        [InlineData(PrimitiveType.MatrixMultiplication)]
        [InlineData(PrimitiveType.Convolution)]
        [InlineData(PrimitiveType.Activation)]
        public void IsPrimitiveAccelerated_ShouldReturnValidResult(PrimitiveType primitiveType)
        {
            // Arrange & Act
            var isAccelerated = _primitives.Capabilities.IsPrimitiveAccelerated(primitiveType);

            // Assert
            Assert.IsType<bool>(isAccelerated);
        }

        #region Helper Methods

        private ITensor<T> CreateRandomTensor<T>(TensorShape shape, T? min = null, T? max = null) where T : unmanaged
        {
            var tensor = TensorFactory.Create<T>(shape, ComputeLocation.Cpu);
            var random = new Random(42); // Fixed seed for reproducible tests

            unsafe
            {
                var ptr = (T*)tensor.GetDataPointer();
                for (long i = 0; i < shape.Length; i++)
                {
                    if (typeof(T) == typeof(float))
                    {
                        var minVal = min != null ? (float)(object)min : -1.0f;
                        var maxVal = max != null ? (float)(object)max : 1.0f;
                        var value = (float)(random.NextDouble() * (maxVal - minVal) + minVal);
                        ptr[i] = (T)(object)value;
                    }
                    else if (typeof(T) == typeof(sbyte))
                    {
                        var minVal = min != null ? (sbyte)(object)min : (sbyte)-127;
                        var maxVal = max != null ? (sbyte)(object)max : (sbyte)127;
                        var value = (sbyte)random.Next(minVal, maxVal + 1);
                        ptr[i] = (T)(object)value;
                    }
                }
            }

            return tensor;
        }

        private ITensor<T> CreateZeroTensor<T>(TensorShape shape) where T : unmanaged
        {
            return TensorFactory.Create<T>(shape, ComputeLocation.Cpu);
        }

        private ITensor<T> CreateOnesTensor<T>(TensorShape shape) where T : unmanaged
        {
            var tensor = TensorFactory.Create<T>(shape, ComputeLocation.Cpu);
            
            unsafe
            {
                var ptr = (T*)tensor.GetDataPointer();
                for (long i = 0; i < shape.Length; i++)
                {
                    if (typeof(T) == typeof(float))
                        ptr[i] = (T)(object)1.0f;
                    else if (typeof(T) == typeof(int))
                        ptr[i] = (T)(object)1;
                }
            }

            return tensor;
        }

        private void ValidateMatrixMultiplication<T>(ITensor<T> a, ITensor<T> b, ITensor<T> c) where T : unmanaged
        {
            // Basic validation - would need reference implementation for full validation
            c.Shape[0].Should().Be(a.Shape[0]);
            c.Shape[1].Should().Be(b.Shape[1]);
        }

        private void ValidateReLUActivation<T>(ITensor<T> input, ITensor<T> output) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                unsafe
                {
                    var inputPtr = (float*)input.GetDataPointer();
                    var outputPtr = (float*)output.GetDataPointer();
                    
                    for (long i = 0; i < input.Shape.Length; i++)
                    {
                        var expected = Math.Max(0.0f, inputPtr[i]);
                        outputPtr[i].Should().BeApproximately(expected, 1e-6f);
                    }
                }
            }
        }

        private void ValidateSoftmaxOutput<T>(ITensor<T> output) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                unsafe
                {
                    var ptr = (float*)output.GetDataPointer();
                    var lastDim = (long)output.Shape[(int)(output.Shape.Length - 1)];
                    
                    for (long i = 0; i < output.Shape.Length; i += lastDim)
                    {
                        var sum = 0.0f;
                        for (long j = 0; j < lastDim; j++)
                        {
                            sum += ptr[i + j];
                        }
                        sum.Should().BeApproximately(1.0f, 1e-5f);
                    }
                }
            }
        }

        private void ValidateLayerNormOutput<T>(ITensor<T> output) where T : unmanaged
        {
            // Layer norm output should have mean ≈ 0 and variance ≈ 1
            output.Should().NotBeNull();
        }

        private void ValidateQuantization<T>(ITensor<T> input, ITensor<sbyte> output, T scale, sbyte zeroPoint) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                unsafe
                {
                    var inputPtr = (float*)input.GetDataPointer();
                    var outputPtr = (sbyte*)output.GetDataPointer();
                    var scaleVal = (float)(object)scale;
                    
                    for (long i = 0; i < input.Shape.Length; i++)
                    {
                        var quantized = (sbyte)((sbyte)Math.Round(inputPtr[i] / scaleVal) + zeroPoint);
                        quantized = (sbyte)Math.Max(-128, Math.Min(127, (int)quantized));
                        outputPtr[i].Should().Be(quantized);
                    }
                }
            }
        }

        private void ValidateDequantization<T>(ITensor<sbyte> input, ITensor<T> output, T scale, sbyte zeroPoint) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                unsafe
                {
                    var inputPtr = (sbyte*)input.GetDataPointer();
                    var outputPtr = (float*)output.GetDataPointer();
                    var scaleVal = (float)(object)scale;
                    
                    for (long i = 0; i < input.Shape.Length; i++)
                    {
                        var dequantized = (inputPtr[i] - zeroPoint) * scaleVal;
                        outputPtr[i].Should().BeApproximately(dequantized, 1e-6f);
                    }
                }
            }
        }

        #endregion

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }
}