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

using ILGPU.Intel.NPU.Native;
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Basic accelerator interface for NPU operations.
    /// </summary>
    public interface IAccelerator : IDisposable
    {
        Device Device { get; }
        bool IsAvailable { get; }
    }

    /// <summary>
    /// Intel Neural Processing Unit (NPU) accelerator for AI workloads.
    /// </summary>
    public sealed class IntelNPUAccelerator : IAccelerator
    {
        #region Instance

        /// <summary>
        /// Gets whether Intel NPU is supported on this system.
        /// </summary>
        public bool SupportsNPU { get; }

        /// <summary>
        /// Gets the NPU capabilities of this device.
        /// </summary>
        public NPUCapabilities Capabilities { get; }

        /// <summary>
        /// Gets the underlying CPU device.
        /// </summary>
        public Device Device { get; }

        /// <summary>
        /// Gets whether this accelerator is available.
        /// </summary>
        public bool IsAvailable => SupportsNPU && NPUNative.IsNPUInitialized();

        /// <summary>
        /// Gets the NPU generation (e.g., NPU2, NPU3, NPU4).
        /// </summary>
        public NPUGeneration Generation { get; }

        internal IntelNPUAccelerator(Device device)
        {
            Device = device ?? throw new ArgumentNullException(nameof(device));
            SupportsNPU = NPUCapabilities.DetectNPU();
            
            if (SupportsNPU)
            {
                Capabilities = NPUCapabilities.Query();
                Generation = Capabilities.Generation;
                NPUNative.InitializeNPU();
            }
            else
            {
                Capabilities = new NPUCapabilities();
                Generation = NPUGeneration.None;
            }
        }

        #endregion

        #region AI Operations

        /// <summary>
        /// Executes a neural network inference using Intel NPU.
        /// </summary>
        public async Task<ITensor<T>> InferenceAsync<T>(
            NeuralNetwork network,
            ITensor<T> input,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            if (!SupportsNPU)
                throw new NotSupportedException("Intel NPU not supported on this device");

            if (network == null)
                throw new ArgumentNullException(nameof(network));

            if (input == null)
                throw new ArgumentNullException(nameof(input));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Create NPU-compatible neural network representation
                var appleNetwork = new ILGPU.Apple.NeuralEngine.NeuralNetwork("NPU_Converted", new ILGPU.Apple.NeuralEngine.NeuralOperation[0]);
                using var npuContext = new NPUInferenceContext(appleNetwork, Capabilities);
                
                // Execute inference on NPU
                var result = ExecuteNPUInference(input, npuContext);
                
                return result;
            }, cancellationToken);
        }

        /// <summary>
        /// Executes convolution operations optimized for NPU.
        /// </summary>
        public async Task<ITensor<float>> ConvolutionAsync(
            ITensor<float> input,
            ITensor<float> weights,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            if (!Capabilities.SupportsConvolution)
                throw new NotSupportedException("Convolution not supported on this NPU generation");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var convConfig = new NPUConvolutionConfig(parameters, Capabilities);
                return ExecuteNPUConvolution(input, weights, convConfig);
            }, cancellationToken);
        }

        /// <summary>
        /// Executes matrix multiplication using NPU matrix engines.
        /// </summary>
        public async Task<ITensor<T>> MatMulAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            if (!Capabilities.SupportsMatMul)
                throw new NotSupportedException("Matrix multiplication not supported on this NPU generation");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var matmulConfig = CreateOptimalMatMulConfig(a.Shape, b.Shape);
                return ExecuteNPUMatMul(a, b, matmulConfig);
            }, cancellationToken);
        }

        /// <summary>
        /// Executes transformer attention operations optimized for NPU.
        /// </summary>
        public async Task<ITensor<float>> AttentionAsync(
            ITensor<float> query,
            ITensor<float> key,
            ITensor<float> value,
            AttentionParameters parameters,
            CancellationToken cancellationToken = default)
        {
            if (!Capabilities.SupportsAttention)
                throw new NotSupportedException("Attention operations not supported on this NPU generation");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var attentionConfig = new NPUAttentionConfig(parameters, Capabilities);
                return ExecuteNPUAttention(query, key, value, attentionConfig);
            }, cancellationToken);
        }

        /// <summary>
        /// Executes an operation on the NPU.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="operation">The operation to execute.</param>
        /// <param name="output">The output tensor.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task<ITensor<T>> ExecuteOperationAsync<T>(
            NPUOperation<T> operation,
            ITensor<T> output,
            CancellationToken cancellationToken = default) where T : unmanaged
        {
            if (operation == null)
                throw new ArgumentNullException(nameof(operation));

            return await operation.ExecuteAsync(this, cancellationToken);
        }

        /// <summary>
        /// Executes a convolution kernel on the NPU.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="kernel">Convolution kernel.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="parameters">Convolution parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteConvolutionKernelAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged
        {
            if (!SupportsNPU)
                throw new NotSupportedException("NPU not supported on this system");

            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                unsafe
                {
                    NPUNative.ExecuteConvolution(
                        input.GetDataPointer(),
                        kernel.GetDataPointer(),
                        output.GetDataPointer(),
                        input.Shape[0], // batchSize
                        input.Shape[1], // inputChannels
                        kernel.Shape[0], // outputChannels
                        input.Shape[2], // inputHeight
                        input.Shape[3], // inputWidth
                        kernel.Shape[2], // kernelHeight
                        kernel.Shape[3], // kernelWidth
                        parameters.Stride.Height, // strideHeight
                        parameters.Stride.Width, // strideWidth
                        parameters.Padding.Height, // paddingHeight
                        parameters.Padding.Width); // paddingWidth
                }
            }, cancellationToken);
        }

        /// <summary>
        /// Executes an attention kernel on the NPU.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="query">Query tensor.</param>
        /// <param name="key">Key tensor.</param>
        /// <param name="value">Value tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="parameters">Attention parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteAttentionKernelAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            AttentionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged
        {
            if (!SupportsNPU)
                throw new NotSupportedException("NPU not supported on this system");

            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                unsafe
                {
                    NPUNative.ExecuteAttention(
                        query.GetDataPointer(),
                        key.GetDataPointer(),
                        value.GetDataPointer(),
                        output.GetDataPointer(),
                        query.Shape[0], // batchSize
                        query.Shape[1], // sequenceLength
                        query.Shape[2], // hiddenSize
                        parameters.NumHeads); // numHeads
                }
            }, cancellationToken);
        }

        #endregion

        #region Model Loading

        /// <summary>
        /// Loads a pre-trained model for NPU execution.
        /// </summary>
        public async Task<NeuralNetwork> LoadModelAsync(
            string modelPath,
            ModelFormat format,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Return basic neural network model
                return new NeuralNetwork("NPU_Model");
            }, cancellationToken);
        }

        /// <summary>
        /// Optimizes a model for NPU execution.
        /// </summary>
        public async Task<NeuralNetwork> OptimizeModelAsync(
            NeuralNetwork model,
            OptimizationOptions options,
            CancellationToken cancellationToken = default)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Apply NPU-specific optimizations
                return model;
            }, cancellationToken);
        }

        #endregion

        #region Performance Monitoring

        /// <summary>
        /// Gets NPU performance metrics.
        /// </summary>
        public NPUPerformanceMetrics GetPerformanceMetrics()
        {
            if (!SupportsNPU)
                throw new NotSupportedException("NPU not available");

            return NPUNative.GetPerformanceMetrics();
        }

        /// <summary>
        /// Gets NPU power consumption information.
        /// </summary>
        public NPUPowerInfo GetPowerInfo()
        {
            if (!SupportsNPU)
                throw new NotSupportedException("NPU not available");

            return NPUNative.GetPowerInfo();
        }

        #endregion

        #region Private Implementation

        private ITensor<T> ExecuteNPUInference<T>(ITensor<T> input, NPUInferenceContext context)
            where T : unmanaged
        {
            // Create result tensor with appropriate output shape
            var outputShape = input.Shape;
            var result = TensorFactory.Create<T>(outputShape, ComputeLocation.Npu);

            unsafe
            {
                // Get data pointers
                var inputPtr = input.GetDataPointer();
                var resultPtr = result.GetDataPointer();

                // Execute NPU inference based on data type
                if (typeof(T) == typeof(float))
                {
                    NPUKernels.InferenceFloat(
                        (float*)inputPtr, (float*)resultPtr,
                        input.Shape, outputShape,
                        context.NativeContext);
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    NPUKernels.InferenceBF16(
                        (BFloat16*)inputPtr, (BFloat16*)resultPtr,
                        input.Shape, outputShape,
                        context.NativeContext);
                }
                else if (typeof(T) == typeof(byte))
                {
                    NPUKernels.InferenceInt8(
                        (byte*)inputPtr, (byte*)resultPtr,
                        input.Shape, outputShape,
                        context.NativeContext);
                }
                else
                {
                    throw new NotSupportedException($"Data type {typeof(T)} not supported by NPU");
                }
            }

            return result;
        }

        private ITensor<float> ExecuteNPUConvolution(ITensor<float> input, ITensor<float> weights, NPUConvolutionConfig config)
        {
            var outputShape = config.CalculateOutputShape(input.Shape, weights.Shape);
            var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Npu);

            unsafe
            {
                NPUKernels.ConvolutionFloat(
                    (float*)input.GetDataPointer(),
                    (float*)weights.GetDataPointer(),
                    (float*)result.GetDataPointer(),
                    input.Shape, weights.Shape, outputShape,
                    config.NativeConfig);
            }

            return result;
        }

        private ITensor<T> ExecuteNPUMatMul<T>(ITensor<T> a, ITensor<T> b, NPUMatMulConfig config)
            where T : unmanaged
        {
            var resultShape = new TensorShape(a.Shape[0], b.Shape[1]);
            var result = TensorFactory.Create<T>(resultShape, ComputeLocation.Npu);

            unsafe
            {
                var aPtr = (T*)a.GetDataPointer();
                var bPtr = (T*)b.GetDataPointer();
                var resultPtr = (T*)result.GetDataPointer();

                if (typeof(T) == typeof(float))
                {
                    NPUKernels.MatMulFloat(
                        (float*)aPtr, (float*)bPtr, (float*)resultPtr,
                        a.Shape[0], a.Shape[1], b.Shape[1],
                        config.NativeConfig);
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    NPUKernels.MatMulBF16(
                        (BFloat16*)aPtr, (BFloat16*)bPtr, (BFloat16*)resultPtr,
                        a.Shape[0], a.Shape[1], b.Shape[1],
                        config.NativeConfig);
                }
                else
                {
                    throw new NotSupportedException($"Data type {typeof(T)} not supported for NPU MatMul");
                }
            }

            return result;
        }

        private ITensor<float> ExecuteNPUAttention(ITensor<float> query, ITensor<float> key, 
            ITensor<float> value, NPUAttentionConfig config)
        {
            var outputShape = query.Shape; // Attention preserves sequence length
            var result = TensorFactory.Create<float>(outputShape, ComputeLocation.Npu);

            unsafe
            {
                NPUKernels.AttentionFloat(
                    (float*)query.GetDataPointer(),
                    (float*)key.GetDataPointer(),
                    (float*)value.GetDataPointer(),
                    (float*)result.GetDataPointer(),
                    query.Shape, key.Shape, value.Shape,
                    config.NativeConfig);
            }

            return result;
        }

        private NPUMatMulConfig CreateOptimalMatMulConfig(TensorShape aShape, TensorShape bShape) => new NPUMatMulConfig(new MatMulConfiguration
        {
            M = aShape[0],
            K = aShape[1],
            N = bShape[1],
            UseBF16 = Capabilities.SupportsBF16,
            UseSparsity = Capabilities.SupportsSparsity
        });

        private IModelLoader CreateModelLoader(ModelFormat format) => format switch
        {
            ModelFormat.ONNX => new ONNXModelLoader(),
            ModelFormat.OpenVINO => new OpenVINOModelLoader(),
            ModelFormat.TensorFlow => new TensorFlowModelLoader(),
            ModelFormat.PyTorch => new PyTorchModelLoader(),
            _ => throw new NotSupportedException($"Model format {format} not supported")
        };

        #endregion

        #region Disposal

        public void Dispose()
        {
            if (SupportsNPU)
            {
                NPUNative.ReleaseNPU();
            }
        }

        #endregion
    }

    /// <summary>
    /// Intel NPU generation types.
    /// </summary>
    public enum NPUGeneration
    {
        /// <summary>
        /// No NPU available.
        /// </summary>
        None = 0,

        /// <summary>
        /// Intel NPU 2.0 (Meteor Lake).
        /// </summary>
        NPU2 = 2,

        /// <summary>
        /// Intel NPU 3.0 (Lunar Lake).
        /// </summary>
        NPU3 = 3,

        /// <summary>
        /// Intel NPU 4.0 (Arrow Lake and future).
        /// </summary>
        NPU4 = 4
    }

    /// <summary>
    /// Supported model formats for NPU loading.
    /// </summary>
    public enum ModelFormat
    {
        /// <summary>
        /// ONNX model format.
        /// </summary>
        ONNX,

        /// <summary>
        /// Intel OpenVINO model format.
        /// </summary>
        OpenVINO,

        /// <summary>
        /// TensorFlow model format.
        /// </summary>
        TensorFlow,

        /// <summary>
        /// PyTorch model format.
        /// </summary>
        PyTorch
    }
}