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

using ILGPU.Apple.NeuralEngine.Native;
using ILGPU.Backends.Metal;
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using System;
using System.Threading;
using System.Threading.Tasks;

#if ENABLE_ANE_ACCELERATOR
namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Apple Neural Engine accelerator for AI workloads on Apple Silicon.
    /// </summary>
    public sealed class AppleNeuralEngine : IDisposable
    {
        private readonly MetalDevice _device;
        private readonly ANECapabilities _capabilities;
        private readonly IntPtr _aneContext;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AppleNeuralEngine class.
        /// </summary>
        /// <param name="device">The Metal device associated with the ANE.</param>
        public AppleNeuralEngine(MetalDevice device)
        {
            _device = device ?? throw new ArgumentNullException(nameof(device));
            _capabilities = ANECapabilities.Query();
            
            if (_capabilities.IsAvailable)
            {
                _aneContext = ANENative.CreateContext();
                if (_aneContext == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to create Apple Neural Engine context");
            }
            else
            {
                throw new NotSupportedException("Apple Neural Engine not available on this device");
            }
        }

        /// <summary>
        /// Gets the Neural Engine capabilities.
        /// </summary>
        public ANECapabilities Capabilities => _capabilities;

        /// <summary>
        /// Gets whether the Neural Engine is available.
        /// </summary>
        public bool IsAvailable => _capabilities.IsAvailable;

        #region Neural Network Operations

        /// <summary>
        /// Executes neural network inference on the Apple Neural Engine.
        /// </summary>
        public async Task<ITensor<T>> ExecuteAsync<T>(
            NeuralOperation operation,
            ITensor<T> input,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            ThrowIfDisposed();
            if (operation == null)
                throw new ArgumentNullException(nameof(operation));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = operation.CalculateOutputShape(input.Shape);
                var result = TensorFactory.Create<T>(outputShape, ComputeLocation.Npu);

                ExecuteNeuralOperation(operation, input, result);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes Core ML model inference on the Neural Engine.
        /// </summary>
        public async Task<ITensor<float>> ExecuteCoreMLAsync(
            CoreMLModel model,
            ITensor<float> input,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = model.GetOutputShape(input.Shape);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Npu);

                ExecuteCoreMLInference(model, input, result);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes optimized convolution using Neural Engine.
        /// </summary>
        public async Task<ITensor<float>> ConvolutionAsync(
            ITensor<float> input,
            ITensor<float> weights,
            ITensor<float> bias,
            ANEConvolutionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (!_capabilities.SupportsConvolution)
                throw new NotSupportedException("Convolution not supported on this Neural Engine generation");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = CalculateConvolutionOutputShape(input.Shape, weights.Shape, parameters);
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Npu);

                ExecuteANEConvolution(input, weights, bias, result, parameters);
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes transformer attention optimized for Neural Engine.
        /// </summary>
        public async Task<ITensor<float>> AttentionAsync(
            ITensor<float> query,
            ITensor<float> key,
            ITensor<float> value,
            ANEAttentionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (!_capabilities.SupportsAttention)
                throw new NotSupportedException("Attention operations not supported on this Neural Engine generation");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var outputShape = query.Shape; // Attention preserves sequence length
                var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Npu);

                ExecuteANEAttention(query, key, value, result, parameters);
                return result;
            }, cancellationToken);
        }

        #endregion

        #region Model Management

        /// <summary>
        /// Loads a Core ML model for Neural Engine execution.
        /// </summary>
        public async Task<CoreMLModel> LoadModelAsync(
            string modelPath,
            ANEOptimizationOptions options = null,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var model = new CoreMLModel(modelPath, _capabilities);
                
                if (options != null)
                {
                    model.OptimizeForNeuralEngine(options);
                }

                return model;
            }, cancellationToken);
        }

        /// <summary>
        /// Compiles a model specifically for Neural Engine execution.
        /// </summary>
        public async Task<CoreMLModel> CompileModelAsync(
            NeuralNetwork network,
            ANECompilationOptions options,
            CancellationToken cancellationToken = default)
        {
            if (network == null)
                throw new ArgumentNullException(nameof(network));
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var compiler = new ANEModelCompiler(_capabilities);
                return compiler.CompileForNeuralEngine(network, options);
            }, cancellationToken);
        }

        #endregion

        #region Performance Monitoring

        /// <summary>
        /// Gets Neural Engine performance metrics.
        /// </summary>
        public ANEPerformanceMetrics GetPerformanceMetrics()
        {
            ThrowIfDisposed();
            if (!IsAvailable)
                throw new NotSupportedException("Neural Engine not available");

            return ANENative.GetPerformanceMetrics(_aneContext);
        }

        /// <summary>
        /// Gets Neural Engine power consumption information.
        /// </summary>
        public ANEPowerInfo GetPowerInfo()
        {
            ThrowIfDisposed();
            if (!IsAvailable)
                throw new NotSupportedException("Neural Engine not available");

            return ANENative.GetPowerInfo(_aneContext);
        }

        /// <summary>
        /// Gets Neural Engine thermal state.
        /// </summary>
        public ANEThermalState GetThermalState()
        {
            ThrowIfDisposed();
            if (!IsAvailable)
                throw new NotSupportedException("Neural Engine not available");

            return ANENative.GetThermalState(_aneContext);
        }

        #endregion

        #region Private Implementation

        private void ExecuteNeuralOperation<T>(NeuralOperation operation, ITensor<T> input, ITensor<T> result)
            where T : unmanaged
        {
            unsafe
            {
                var inputPtr = input.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                // Execute based on operation type
                switch (operation.Type)
                {
                    case NeuralOperationType.Convolution:
                        if (operation is ConvolutionOperation convOp)
                        {
                            ANEKernels.ExecuteConvolution(
                                (float*)inputPtr, (float*)resultPtr,
                                input.Shape, result.Shape,
                                convOp.Parameters, _aneContext);
                        }
                        break;

                    case NeuralOperationType.MatMul:
                        ANEKernels.ExecuteMatMul(
                            (float*)inputPtr, (float*)resultPtr,
                            input.Shape, result.Shape,
                            _aneContext);
                        break;

                    case NeuralOperationType.Attention:
                        if (operation is AttentionOperation attOp)
                        {
                            ANEKernels.ExecuteAttention(
                                (float*)inputPtr, (float*)resultPtr,
                                input.Shape, result.Shape,
                                attOp.Parameters, _aneContext);
                        }
                        break;

                    default:
                        throw new NotSupportedException($"Neural operation {operation.Type} not supported on ANE");
                }
            }
        }

        private void ExecuteCoreMLInference(CoreMLModel model, ITensor<float> input, ITensor<float> result)
        {
            unsafe
            {
                var inputPtr = input.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                ANEKernels.ExecuteCoreMLInference(
                    (float*)inputPtr, (float*)resultPtr,
                    input.Shape, result.Shape,
                    model.NativeHandle, _aneContext);
            }
        }

        private void ExecuteANEConvolution(ITensor<float> input, ITensor<float> weights, ITensor<float> bias, 
            ITensor<float> result, ANEConvolutionParameters parameters)
        {
            unsafe
            {
                ANEKernels.ExecuteConvolutionWithBias(
                    (float*)input.GetDataPointer(),
                    (float*)weights.GetDataPointer(),
                    (float*)bias.GetDataPointer(),
                    (float*)result.GetDataPointer(),
                    input.Shape, weights.Shape, result.Shape,
                    parameters, _aneContext);
            }
        }

        private void ExecuteANEAttention(ITensor<float> query, ITensor<float> key, ITensor<float> value, 
            ITensor<float> result, ANEAttentionParameters parameters)
        {
            unsafe
            {
                ANEKernels.ExecuteMultiHeadAttention(
                    (float*)query.GetDataPointer(),
                    (float*)key.GetDataPointer(),
                    (float*)value.GetDataPointer(),
                    (float*)result.GetDataPointer(),
                    query.Shape, key.Shape, value.Shape,
                    parameters, _aneContext);
            }
        }

        private static TensorShape CalculateConvolutionOutputShape(TensorShape inputShape, TensorShape weightsShape, 
            ANEConvolutionParameters parameters)
        {
            var batchSize = inputShape[0];
            var outputChannels = weightsShape[0];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            
            var outputHeight = (inputHeight + 2 * parameters.Padding.Height - parameters.KernelSize.Height) / parameters.Stride.Height + 1;
            var outputWidth = (inputWidth + 2 * parameters.Padding.Width - parameters.KernelSize.Width) / parameters.Stride.Width + 1;

            return new TensorShape(batchSize, outputChannels, outputHeight, outputWidth);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AppleNeuralEngine));
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes the Apple Neural Engine.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_aneContext != IntPtr.Zero)
                {
                    ANENative.ReleaseContext(_aneContext);
                }
                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// Convolution operation for Neural operations.
    /// </summary>
    public sealed class ConvolutionOperation : NeuralOperation
    {
        public ConvolutionOperation(ConvolutionParameters parameters)
        {
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        }

        public override string Name => "Convolution";
        public override NeuralOperationType Type => NeuralOperationType.Convolution;
        public override TensorShape InputShape { get; }
        public ConvolutionParameters Parameters { get; }

        public override TensorShape CalculateOutputShape(TensorShape inputShape)
        {
            var batchSize = inputShape[0];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            
            var outputHeight = (inputHeight + 2 * Parameters.Padding.Height - Parameters.KernelSize.Height) / Parameters.Stride.Height + 1;
            var outputWidth = (inputWidth + 2 * Parameters.Padding.Width - Parameters.KernelSize.Width) / Parameters.Stride.Width + 1;

            return new TensorShape(batchSize, 64, outputHeight, outputWidth); // Placeholder output channels
        }
    }

    /// <summary>
    /// Attention operation for Neural operations.
    /// </summary>
    public sealed class AttentionOperation : NeuralOperation
    {
        public AttentionOperation(AttentionParameters parameters)
        {
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        }

        public override string Name => "Attention";
        public override NeuralOperationType Type => NeuralOperationType.Attention;
        public override TensorShape InputShape { get; }
        public AttentionParameters Parameters { get; }

        public override TensorShape CalculateOutputShape(TensorShape inputShape)
        {
            return inputShape; // Attention preserves input shape
        }
    }
}
#endif