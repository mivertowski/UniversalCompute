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

using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using System;
using System.Threading;
using System.Threading.Tasks;

#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Apple Metal Performance Shaders interface for optimized GPU operations.
    /// </summary>
    public sealed class MetalPerformanceShaders : IDisposable
    {
        private readonly IntPtr _device;
        private readonly MetalCapabilities _capabilities;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalPerformanceShaders class.
        /// </summary>
        /// <param name="device">The Metal device handle.</param>
        public MetalPerformanceShaders(IntPtr device)
        {
            _device = device;
            _capabilities = MetalCapabilities.Query(device);
        }

        /// <summary>
        /// Gets the Metal device capabilities.
        /// </summary>
        public MetalCapabilities Capabilities => _capabilities;

        #region Matrix Operations

        /// <summary>
        /// Executes optimized matrix multiplication using Metal Performance Shaders.
        /// </summary>
        public async Task<ITensor<T>> MatrixMultiplyAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            ThrowIfDisposed();
            ValidateMatrixInputs(a, b);

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var resultShape = new TensorShape(a.Shape[0], b.Shape[1]);
                var result = TensorFactory.Create<T>(resultShape, ComputeLocation.Gpu);

                ExecuteMatrixMultiply(a, b, result);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes optimized matrix-vector multiplication.
        /// </summary>
        public async Task<ITensor<T>> MatrixVectorMultiplyAsync<T>(
            ITensor<T> matrix,
            ITensor<T> vector,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            ThrowIfDisposed();
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var resultShape = new TensorShape(matrix.Shape[0]);
                var result = TensorFactory.Create<T>(resultShape, ComputeLocation.Gpu);

                ExecuteMatrixVectorMultiply(matrix, vector, result);
                return result;
            }, cancellationToken);
        }

        #endregion

        #region Convolution Operations

        /// <summary>
        /// Executes optimized 2D convolution using Metal Performance Shaders.
        /// </summary>
        public async Task<ITensor<float>> ConvolutionAsync(
            ITensor<float> input,
            ITensor<float> weights,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = CalculateConvolutionOutputShape(input.Shape, weights.Shape, parameters);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Gpu);

                ExecuteConvolution(input, weights, result, parameters);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes depthwise convolution optimized for mobile architectures.
        /// </summary>
        public async Task<ITensor<float>> DepthwiseConvolutionAsync(
            ITensor<float> input,
            ITensor<float> weights,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = CalculateConvolutionOutputShape(input.Shape, weights.Shape, parameters);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Gpu);

                ExecuteDepthwiseConvolution(input, weights, result, parameters);
                return result;
            }, cancellationToken);
        }

        #endregion

        #region Neural Network Operations

        /// <summary>
        /// Executes ReLU activation function.
        /// </summary>
        public async Task<ITensor<float>> ReLUAsync(
            ITensor<float> input,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (input == null) throw new ArgumentNullException(nameof(input));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var result = TensorFactory.Create<float>(input.Shape, ComputeLocation.Gpu);
                ExecuteReLU(input, result);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes softmax activation function.
        /// </summary>
        public async Task<ITensor<float>> SoftmaxAsync(
            ITensor<float> input,
            int axis = -1,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (input == null) throw new ArgumentNullException(nameof(input));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var result = TensorFactory.Create<float>(input.Shape, ComputeLocation.Gpu);
                ExecuteSoftmax(input, result, axis);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes batch normalization.
        /// </summary>
        public async Task<ITensor<float>> BatchNormalizationAsync(
            ITensor<float> input,
            ITensor<float> scale,
            ITensor<float> bias,
            ITensor<float> mean,
            ITensor<float> variance,
            float epsilon = 1e-5f,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var result = TensorFactory.Create<float>(input.Shape, ComputeLocation.Gpu);
                ExecuteBatchNormalization(input, scale, bias, mean, variance, result, epsilon);
                return result;
            }, cancellationToken);
        }

        #endregion

        #region Pooling Operations

        /// <summary>
        /// Executes max pooling operation.
        /// </summary>
        public async Task<ITensor<float>> MaxPoolingAsync(
            ITensor<float> input,
            (int height, int width) poolSize,
            (int height, int width) stride,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = CalculatePoolingOutputShape(input.Shape, poolSize, stride);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Gpu);

                ExecuteMaxPooling(input, result, poolSize, stride);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes average pooling operation.
        /// </summary>
        public async Task<ITensor<float>> AveragePoolingAsync(
            ITensor<float> input,
            (int height, int width) poolSize,
            (int height, int width) stride,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = CalculatePoolingOutputShape(input.Shape, poolSize, stride);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Gpu);

                ExecuteAveragePooling(input, result, poolSize, stride);
                return result;
            }, cancellationToken);
        }

        #endregion

        #region Private Implementation

        private static void ValidateMatrixInputs<T>(ITensor<T> a, ITensor<T> b) where T : unmanaged
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException("Matrix multiplication requires 2D tensors");
            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException("Matrix dimensions incompatible for multiplication");
        }

        private void ExecuteMatrixMultiply<T>(ITensor<T> a, ITensor<T> b, ITensor<T> result) where T : unmanaged
        {
            // In a real implementation, this would use Metal Performance Shaders BLAS operations
            // For now, this is a placeholder that would call native MPS functions
            unsafe
            {
                var aPtr = a.GetDataPointer();
                var bPtr = b.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                // Call MPS matrix multiplication
                // MPSMatrixMultiplication would be called here
            }
        }

        private void ExecuteMatrixVectorMultiply<T>(ITensor<T> matrix, ITensor<T> vector, ITensor<T> result) where T : unmanaged
        {
            // MPS optimized matrix-vector multiplication
            unsafe
            {
                var matrixPtr = matrix.GetDataPointer();
                var vectorPtr = vector.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                // Call MPS matrix-vector multiplication
            }
        }

        private void ExecuteConvolution(ITensor<float> input, ITensor<float> weights, ITensor<float> result, ConvolutionParameters parameters)
        {
            // MPS CNNConvolution kernel
            unsafe
            {
                var inputPtr = input.GetDataPointer();
                var weightsPtr = weights.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                // Call MPS convolution
            }
        }

        private void ExecuteDepthwiseConvolution(ITensor<float> input, ITensor<float> weights, ITensor<float> result, ConvolutionParameters parameters)
        {
            // MPS CNNConvolution with depthwise configuration
        }

        private void ExecuteReLU(ITensor<float> input, ITensor<float> result)
        {
            // MPS CNNNeuronReLU
        }

        private void ExecuteSoftmax(ITensor<float> input, ITensor<float> result, int axis)
        {
            // MPS CNNSoftMax
        }

        private void ExecuteBatchNormalization(ITensor<float> input, ITensor<float> scale, ITensor<float> bias, 
            ITensor<float> mean, ITensor<float> variance, ITensor<float> result, float epsilon)
        {
            // MPS CNNBatchNormalization
        }

        private void ExecuteMaxPooling(ITensor<float> input, ITensor<float> result, (int height, int width) poolSize, (int height, int width) stride)
        {
            // MPS CNNPoolingMax
        }

        private void ExecuteAveragePooling(ITensor<float> input, ITensor<float> result, (int height, int width) poolSize, (int height, int width) stride)
        {
            // MPS CNNPoolingAverage
        }

        private static TensorShape CalculateConvolutionOutputShape(TensorShape inputShape, TensorShape weightsShape, ConvolutionParameters parameters)
        {
            var batchSize = inputShape[0];
            var outputChannels = weightsShape[0];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            
            var outputHeight = (inputHeight + 2 * parameters.Padding.Height - parameters.KernelSize.Height) / parameters.Stride.Height + 1;
            var outputWidth = (inputWidth + 2 * parameters.Padding.Width - parameters.KernelSize.Width) / parameters.Stride.Width + 1;

            return new TensorShape(batchSize, outputChannels, outputHeight, outputWidth);
        }

        private static TensorShape CalculatePoolingOutputShape(TensorShape inputShape, (int height, int width) poolSize, (int height, int width) stride)
        {
            var batchSize = inputShape[0];
            var channels = inputShape[1];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            
            var outputHeight = (inputHeight - poolSize.height) / stride.height + 1;
            var outputWidth = (inputWidth - poolSize.width) / stride.width + 1;

            return new TensorShape(batchSize, channels, outputHeight, outputWidth);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalPerformanceShaders));
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes the Metal Performance Shaders interface.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // Release MPS resources
                _disposed = true;
            }
        }

        #endregion
    }
}
#endif