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

namespace ILGPU.Runtime.AI
{
    /// <summary>
    /// Unified interface for high-performance AI/ML primitives across all accelerators.
    /// </summary>
    public interface IPerformancePrimitives
    {
        /// <summary>
        /// Gets the accelerator associated with these primitives.
        /// </summary>
        Accelerator Accelerator { get; }

        /// <summary>
        /// Gets the capabilities of the performance primitives.
        /// </summary>
        PerformancePrimitiveCapabilities Capabilities { get; }

        #region Matrix Operations

        /// <summary>
        /// Performs general matrix multiplication (GEMM).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="alpha">Scalar multiplier for A*B.</param>
        /// <param name="beta">Scalar multiplier for C.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task GemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            T alpha,
            T beta,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Performs batched matrix multiplication.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">Batch of matrices A.</param>
        /// <param name="b">Batch of matrices B.</param>
        /// <param name="c">Result batch of matrices C.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task BatchedGemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Convolution Operations

        /// <summary>
        /// Performs 2D convolution.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="kernel">Convolution kernel.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="parameters">Convolution parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task Conv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Performs depthwise convolution.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="kernel">Depthwise kernel.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="parameters">Convolution parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task DepthwiseConv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Attention Operations

        /// <summary>
        /// Performs multi-head attention.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="query">Query tensor.</param>
        /// <param name="key">Key tensor.</param>
        /// <param name="value">Value tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="parameters">Attention parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task MultiHeadAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            AttentionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Performs scaled dot-product attention.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="query">Query tensor.</param>
        /// <param name="key">Key tensor.</param>
        /// <param name="value">Value tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="scale">Scale factor.</param>
        /// <param name="mask">Optional attention mask.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task ScaledDotProductAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            T scale,
            ITensor<bool>? mask = null,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Activation Functions

        /// <summary>
        /// Applies ReLU activation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task ReLUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Applies GELU activation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task GELUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Applies Softmax activation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="axis">The axis along which to apply softmax.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task SoftmaxAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            int axis = -1,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Normalization Operations

        /// <summary>
        /// Applies layer normalization.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="gamma">Scale parameter.</param>
        /// <param name="beta">Shift parameter.</param>
        /// <param name="epsilon">Small value for numerical stability.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task LayerNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Applies batch normalization.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="mean">Running mean.</param>
        /// <param name="variance">Running variance.</param>
        /// <param name="gamma">Scale parameter.</param>
        /// <param name="beta">Shift parameter.</param>
        /// <param name="epsilon">Small value for numerical stability.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task BatchNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> mean,
            ITensor<T> variance,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Pooling Operations

        /// <summary>
        /// Performs max pooling.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="poolSize">Pool size.</param>
        /// <param name="stride">Stride.</param>
        /// <param name="padding">Padding.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task MaxPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Performs average pooling.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="poolSize">Pool size.</param>
        /// <param name="stride">Stride.</param>
        /// <param name="padding">Padding.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task AvgPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion

        #region Quantization Operations

        /// <summary>
        /// Quantizes a tensor to INT8.
        /// </summary>
        /// <typeparam name="T">The input element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Output INT8 tensor.</param>
        /// <param name="scale">Scale factor.</param>
        /// <param name="zeroPoint">Zero point.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task QuantizeToInt8Async<T>(
            ITensor<T> input,
            ITensor<sbyte> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) where T : unmanaged;

        /// <summary>
        /// Dequantizes an INT8 tensor.
        /// </summary>
        /// <typeparam name="T">The output element type.</typeparam>
        /// <param name="input">Input INT8 tensor.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="scale">Scale factor.</param>
        /// <param name="zeroPoint">Zero point.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task DequantizeFromInt8Async<T>(
            ITensor<sbyte> input,
            ITensor<T> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion
    }

    /// <summary>
    /// Capabilities of performance primitives.
    /// </summary>
    public sealed class PerformancePrimitiveCapabilities
    {
        /// <summary>
        /// Gets whether matrix multiplication is hardware-accelerated.
        /// </summary>
        public bool SupportsAcceleratedGemm { get; set; }

        /// <summary>
        /// Gets whether convolution is hardware-accelerated.
        /// </summary>
        public bool SupportsAcceleratedConvolution { get; set; }

        /// <summary>
        /// Gets whether attention mechanisms are hardware-accelerated.
        /// </summary>
        public bool SupportsAcceleratedAttention { get; set; }

        /// <summary>
        /// Gets whether FP16 operations are supported.
        /// </summary>
        public bool SupportsFP16 { get; set; }

        /// <summary>
        /// Gets whether BFloat16 operations are supported.
        /// </summary>
        public bool SupportsBFloat16 { get; set; }

        /// <summary>
        /// Gets whether INT8 operations are supported.
        /// </summary>
        public bool SupportsInt8 { get; set; }

        /// <summary>
        /// Gets whether tensor cores or matrix units are available.
        /// </summary>
        public bool HasTensorCores { get; set; }

        /// <summary>
        /// Gets the preferred batch size for optimal performance.
        /// </summary>
        public int PreferredBatchSize { get; set; }

        /// <summary>
        /// Gets the maximum supported tensor rank.
        /// </summary>
        public int MaxTensorRank { get; set; }

        /// <summary>
        /// Gets whether unified memory is supported.
        /// </summary>
        public bool SupportsUnifiedMemory { get; set; }

        /// <summary>
        /// Gets the estimated peak performance in TFLOPS.
        /// </summary>
        public double PeakTFLOPS { get; set; }

        /// <summary>
        /// Checks if a specific primitive is accelerated.
        /// </summary>
        /// <param name="primitive">The primitive type.</param>
        /// <returns>True if accelerated; otherwise, false.</returns>
        public bool IsPrimitiveAccelerated(PrimitiveType primitive) => primitive switch
        {
            PrimitiveType.MatrixMultiplication => SupportsAcceleratedGemm,
            PrimitiveType.Convolution => SupportsAcceleratedConvolution,
            PrimitiveType.Attention => SupportsAcceleratedAttention,
            _ => false
        };
    }

    /// <summary>
    /// Types of performance primitives.
    /// </summary>
    public enum PrimitiveType
    {
        /// <summary>
        /// Matrix multiplication operations.
        /// </summary>
        MatrixMultiplication,

        /// <summary>
        /// Convolution operations.
        /// </summary>
        Convolution,

        /// <summary>
        /// Attention mechanisms.
        /// </summary>
        Attention,

        /// <summary>
        /// Activation functions.
        /// </summary>
        Activation,

        /// <summary>
        /// Normalization operations.
        /// </summary>
        Normalization,

        /// <summary>
        /// Pooling operations.
        /// </summary>
        Pooling,

        /// <summary>
        /// Quantization operations.
        /// </summary>
        Quantization
    }

    /// <summary>
    /// Factory for creating performance primitive implementations.
    /// </summary>
    public static class PerformancePrimitivesFactory
    {
        /// <summary>
        /// Creates performance primitives for the given accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <returns>Performance primitives implementation.</returns>
        public static IPerformancePrimitives Create(Accelerator accelerator)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            return accelerator.AcceleratorType switch
            {
                AcceleratorType.Cuda => new CudaPerformancePrimitives(accelerator),
                AcceleratorType.OpenCL => new OpenCLPerformancePrimitives(accelerator),
                AcceleratorType.CPU => new CPUPerformancePrimitives(accelerator),
#if ENABLE_METAL_ACCELERATOR
                AcceleratorType.Metal => new MetalPerformancePrimitives(accelerator),
#endif
#if ENABLE_ONEAPI_ACCELERATOR
                AcceleratorType.OneAPI => new OneAPIPerformancePrimitives(accelerator),
#endif
#if ENABLE_AMX_ACCELERATOR
                AcceleratorType.IntelAMX => new AMXPerformancePrimitives(accelerator),
#endif
#if ENABLE_NPU_ACCELERATOR
                AcceleratorType.IntelNPU => new NPUPerformancePrimitives(accelerator),
#endif
#if ENABLE_ANE_ACCELERATOR
                AcceleratorType.AppleNeuralEngine => new ANEPerformancePrimitives(accelerator),
#endif
                _ => new GenericPerformancePrimitives(accelerator)
            };
        }

        /// <summary>
        /// Checks if hardware-accelerated primitives are available for the accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <returns>True if accelerated primitives are available; otherwise, false.</returns>
        public static bool HasAcceleratedPrimitives(Accelerator accelerator)
        {
            if (accelerator == null)
                return false;

            return accelerator.AcceleratorType switch
            {
                AcceleratorType.Cuda => true, // cuBLAS, cuDNN
                AcceleratorType.Metal => true, // Metal Performance Shaders
                AcceleratorType.OneAPI => true, // oneMKL, oneDNN
                AcceleratorType.IntelAMX => true, // AMX instructions
                AcceleratorType.IntelNPU => true, // NPU acceleration
                AcceleratorType.AppleNeuralEngine => true, // ANE acceleration
                _ => false
            };
        }
    }

    /// <summary>
    /// Base implementation of performance primitives.
    /// </summary>
    public abstract class PerformancePrimitivesBase : IPerformancePrimitives
    {
        /// <summary>
        /// Initializes a new instance of the PerformancePrimitivesBase class.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        protected PerformancePrimitivesBase(Accelerator accelerator)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Capabilities = InitializeCapabilities();
        }

        /// <summary>
        /// Gets the accelerator.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Gets the capabilities.
        /// </summary>
        public PerformancePrimitiveCapabilities Capabilities { get; }

        /// <summary>
        /// Initializes the capabilities for this accelerator.
        /// </summary>
        /// <returns>The capabilities.</returns>
        protected abstract PerformancePrimitiveCapabilities InitializeCapabilities();

        #region Abstract Methods

        public abstract Task GemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            T alpha,
            T beta,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task BatchedGemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task Conv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task DepthwiseConv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task MultiHeadAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            AttentionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task ScaledDotProductAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            T scale,
            ITensor<bool>? mask = null,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task ReLUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task GELUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task SoftmaxAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            int axis = -1,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task LayerNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task BatchNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> mean,
            ITensor<T> variance,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task MaxPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task AvgPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task QuantizeToInt8Async<T>(
            ITensor<T> input,
            ITensor<sbyte> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) where T : unmanaged;

        public abstract Task DequantizeFromInt8Async<T>(
            ITensor<sbyte> input,
            ITensor<T> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) where T : unmanaged;

        #endregion
    }

    /// <summary>
    /// Generic fallback implementation of performance primitives.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the GenericPerformancePrimitives class.
    /// </remarks>
    /// <param name="accelerator">The accelerator.</param>
    public sealed class GenericPerformancePrimitives(Accelerator accelerator) : PerformancePrimitivesBase(accelerator)
    {
        protected override PerformancePrimitiveCapabilities InitializeCapabilities() => new PerformancePrimitiveCapabilities
        {
            SupportsAcceleratedGemm = false,
            SupportsAcceleratedConvolution = false,
            SupportsAcceleratedAttention = false,
            SupportsFP16 = false,
            SupportsBFloat16 = false,
            SupportsInt8 = false,
            HasTensorCores = false,
            PreferredBatchSize = 1,
            MaxTensorRank = 8,
            SupportsUnifiedMemory = Accelerator.SupportsUnifiedMemory,
            PeakTFLOPS = 0.1 // Fallback implementation
        };

        public override Task GemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            T alpha,
            T beta,
            CancellationToken cancellationToken = default) =>
            // Generic CPU-based implementation
            Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Basic matrix multiplication C = alpha * A * B + beta * C
                var M = a.Shape[0];
                var K = a.Shape[1];
                var N = b.Shape[1];

                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        var sum = GetZero<T>();
                        for (int k = 0; k < K; k++)
                        {
                            var aVal = a[i, k];
                            var bVal = b[k, j];
                            sum = Add(sum, Multiply(aVal, bVal));
                        }
                        var result = Add(Multiply(alpha, sum), Multiply(beta, c[i, j]));
                        c[i, j] = result;
                    }
                }
            }, cancellationToken);

        public override Task BatchedGemmAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T> c,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Batched matrix multiplication for tensors with batch dimension
                                                                           var batchSize = a.Shape[0];
                                                                           var M = a.Shape[1];
                                                                           var K = a.Shape[2];
                                                                           var N = b.Shape[2];

                                                                           for (int batch = 0; batch < batchSize; batch++)
                                                                           {
                                                                               for (int i = 0; i < M; i++)
                                                                               {
                                                                                   for (int j = 0; j < N; j++)
                                                                                   {
                                                                                       var sum = GetZero<T>();
                                                                                       for (int k = 0; k < K; k++)
                                                                                       {
                                                                                           var aVal = a[batch, i, k];
                                                                                           var bVal = b[batch, k, j];
                                                                                           sum = Add(sum, Multiply(aVal, bVal));
                                                                                       }
                                                                                       c[batch, i, j] = sum;
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task Conv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Basic 2D convolution implementation
                                                                           var batchSize = input.Shape[0];
                                                                           var inputChannels = input.Shape[1];
                                                                           var inputHeight = input.Shape[2];
                                                                           var inputWidth = input.Shape[3];

                                                                           var outputChannels = kernel.Shape[0];
                                                                           var kernelHeight = kernel.Shape[2];
                                                                           var kernelWidth = kernel.Shape[3];

                                                                           var outputHeight = (inputHeight + 2 * parameters.Padding.Height - kernelHeight) / parameters.Stride.Height + 1;
                                                                           var outputWidth = (inputWidth + 2 * parameters.Padding.Width - kernelWidth) / parameters.Stride.Width + 1;

                                                                           for (int b = 0; b < batchSize; b++)
                                                                           {
                                                                               for (int oc = 0; oc < outputChannels; oc++)
                                                                               {
                                                                                   for (int oh = 0; oh < outputHeight; oh++)
                                                                                   {
                                                                                       for (int ow = 0; ow < outputWidth; ow++)
                                                                                       {
                                                                                           var sum = GetZero<T>();

                                                                                           for (int ic = 0; ic < inputChannels; ic++)
                                                                                           {
                                                                                               for (int kh = 0; kh < kernelHeight; kh++)
                                                                                               {
                                                                                                   for (int kw = 0; kw < kernelWidth; kw++)
                                                                                                   {
                                                                                                       var ih = oh * parameters.Stride.Height - parameters.Padding.Height + kh;
                                                                                                       var iw = ow * parameters.Stride.Width - parameters.Padding.Width + kw;

                                                                                                       if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                                                                                       {
                                                                                                           var inputVal = input[b, ic, ih, iw];
                                                                                                           var kernelVal = kernel[oc, ic, kh, kw];
                                                                                                           sum = Add(sum, Multiply(inputVal, kernelVal));
                                                                                                       }
                                                                                                   }
                                                                                               }
                                                                                           }

                                                                                           output[b, oc, oh, ow] = sum;
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task DepthwiseConv2DAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ITensor<T> output,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Generic depthwise convolution implementation
                                                                           // Depthwise conv applies one filter per input channel
                                                                           var batchSize = input.Shape[0];
                                                                           var inChannels = input.Shape[1];
                                                                           var inHeight = input.Shape[2];
                                                                           var inWidth = input.Shape[3];

                                                                           var kernelHeight = kernel.Shape[2];
                                                                           var kernelWidth = kernel.Shape[3];

                                                                           var outHeight = (inHeight + 2 * parameters.Padding.Height - kernelHeight) / parameters.Stride.Height + 1;
                                                                           var outWidth = (inWidth + 2 * parameters.Padding.Width - kernelWidth) / parameters.Stride.Width + 1;

                                                                           unsafe
                                                                           {
                                                                               var inputPtr = (T*)input.GetDataPointer();
                                                                               var kernelPtr = (T*)kernel.GetDataPointer();
                                                                               var outputPtr = (T*)output.GetDataPointer();

                                                                               for (int b = 0; b < batchSize; b++)
                                                                               {
                                                                                   for (int c = 0; c < inChannels; c++)
                                                                                   {
                                                                                       for (int oh = 0; oh < outHeight; oh++)
                                                                                       {
                                                                                           for (int ow = 0; ow < outWidth; ow++)
                                                                                           {
                                                                                               T sum = default(T);

                                                                                               for (int kh = 0; kh < kernelHeight; kh++)
                                                                                               {
                                                                                                   for (int kw = 0; kw < kernelWidth; kw++)
                                                                                                   {
                                                                                                       int ih = oh * parameters.Stride.Height + kh - parameters.Padding.Height;
                                                                                                       int iw = ow * parameters.Stride.Width + kw - parameters.Padding.Width;

                                                                                                       if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                                                                                       {
                                                                                                           var inputIdx = b * inChannels * inHeight * inWidth +
                                                                                                                        c * inHeight * inWidth + ih * inWidth + iw;
                                                                                                           var kernelIdx = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;

                                                                                                           if (typeof(T) == typeof(float))
                                                                                                           {
                                                                                                               var inputVal = (float)(object)inputPtr[inputIdx];
                                                                                                               var kernelVal = (float)(object)kernelPtr[kernelIdx];
                                                                                                               sum = (T)(object)((float)(object)sum + inputVal * kernelVal);
                                                                                                           }
                                                                                                           else if (typeof(T) == typeof(double))
                                                                                                           {
                                                                                                               var inputVal = (double)(object)inputPtr[inputIdx];
                                                                                                               var kernelVal = (double)(object)kernelPtr[kernelIdx];
                                                                                                               sum = (T)(object)((double)(object)sum + inputVal * kernelVal);
                                                                                                           }
                                                                                                       }
                                                                                                   }
                                                                                               }

                                                                                               var outputIdx = b * inChannels * outHeight * outWidth +
                                                                                                             c * outHeight * outWidth + oh * outWidth + ow;
                                                                                               outputPtr[outputIdx] = sum;
                                                                                           }
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task MultiHeadAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            AttentionParameters parameters,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Simplified multi-head attention implementation
                                                                           var batchSize = query.Shape[0];
                                                                           var seqLength = query.Shape[1];
                                                                           var headDim = query.Shape[2] / parameters.NumHeads;

                                                                           // For simplicity, implement single-head attention and replicate
                                                                           for (int b = 0; b < batchSize; b++)
                                                                           {
                                                                               for (int head = 0; head < parameters.NumHeads; head++)
                                                                               {
                                                                                   var headOffset = head * headDim;

                                                                                   // Compute attention scores for this head
                                                                                   for (int i = 0; i < seqLength; i++)
                                                                                   {
                                                                                       for (int j = 0; j < seqLength; j++)
                                                                                       {
                                                                                           var score = GetZero<T>();

                                                                                           // Dot product between query and key
                                                                                           for (int d = 0; d < headDim; d++)
                                                                                           {
                                                                                               var qVal = query[b, i, headOffset + d];
                                                                                               var kVal = key[b, j, headOffset + d];
                                                                                               score = Add(score, Multiply(qVal, kVal));
                                                                                           }

                                                                                           // Scale by sqrt(head_dim)
                                                                                           var scale = CreateScalar<T>(1.0f / MathF.Sqrt(headDim));
                                                                                           score = Multiply(score, scale);

                                                                                           // Apply to output (simplified - would normally apply softmax)
                                                                                           for (int d = 0; d < headDim; d++)
                                                                                           {
                                                                                               var vVal = value[b, j, headOffset + d];
                                                                                               var weightedVal = Multiply(score, vVal);
                                                                                               output[b, i, headOffset + d] = Add(output[b, i, headOffset + d], weightedVal);
                                                                                           }
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task ScaledDotProductAttentionAsync<T>(
            ITensor<T> query,
            ITensor<T> key,
            ITensor<T> value,
            ITensor<T> output,
            T scale,
            ITensor<bool>? mask = null,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Simplified scaled dot-product attention: output = softmax(QK^T / scale) * V
                                                                           var batchSize = query.Shape[0];
                                                                           var seqLength = query.Shape[1];
                                                                           var headDim = query.Shape[2];

                                                                           for (int b = 0; b < batchSize; b++)
                                                                           {
                                                                               // Compute attention scores: Q * K^T
                                                                               for (int i = 0; i < seqLength; i++)
                                                                               {
                                                                                   // Compute scores for position i
                                                                                   var scores = new T[seqLength];
                                                                                   var maxScore = GetZero<T>();

                                                                                   for (int j = 0; j < seqLength; j++)
                                                                                   {
                                                                                       var score = GetZero<T>();
                                                                                       for (int d = 0; d < headDim; d++)
                                                                                       {
                                                                                           var qVal = query[b, i, d];
                                                                                           var kVal = key[b, j, d];
                                                                                           score = Add(score, Multiply(qVal, kVal));
                                                                                       }
                                                                                       score = Multiply(score, scale);

                                                                                       // Apply mask if provided
                                                                                       if (mask != null && !mask[b, i, j])
                                                                                           score = CreateScalar<T>(-1e9f); // Large negative value

                                                                                       scores[j] = score;
                                                                                       if (j == 0 || IsGreaterThan(score, maxScore))
                                                                                           maxScore = score;
                                                                                   }

                                                                                   // Apply softmax
                                                                                   var sumExp = GetZero<T>();
                                                                                   for (int j = 0; j < seqLength; j++)
                                                                                   {
                                                                                       var expVal = Exp(Subtract(scores[j], maxScore));
                                                                                       scores[j] = expVal;
                                                                                       sumExp = Add(sumExp, expVal);
                                                                                   }

                                                                                   // Normalize and compute output
                                                                                   for (int d = 0; d < headDim; d++)
                                                                                   {
                                                                                       var outputVal = GetZero<T>();
                                                                                       for (int j = 0; j < seqLength; j++)
                                                                                       {
                                                                                           var weight = Divide(scores[j], sumExp);
                                                                                           var vVal = value[b, j, d];
                                                                                           outputVal = Add(outputVal, Multiply(weight, vVal));
                                                                                       }
                                                                                       output[b, i, d] = outputVal;
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task ReLUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // ReLU: output = max(0, input)
                                                                           var totalElements = input.Length;
                                                                           for (long i = 0; i < totalElements; i++)
                                                                           {
                                                                               var flatIndex = ComputeFlatIndex(input.Shape, i);
                                                                               var inputVal = GetElementAtFlatIndex(input, flatIndex);
                                                                               var zero = GetZero<T>();
                                                                               var result = IsGreaterThan(inputVal, zero) ? inputVal : zero;
                                                                               SetElementAtFlatIndex(output, flatIndex, result);
                                                                           }
                                                                       }, cancellationToken);

        public override Task GELUAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                                                                           var totalElements = input.Length;
                                                                           var sqrt2OverPi = CreateScalar<T>(0.7978845608f); // sqrt(2/π)
                                                                           var coefficient = CreateScalar<T>(0.044715f);
                                                                           var half = CreateScalar<T>(0.5f);
                                                                           var one = CreateScalar<T>(1.0f);

                                                                           for (long i = 0; i < totalElements; i++)
                                                                           {
                                                                               var flatIndex = ComputeFlatIndex(input.Shape, i);
                                                                               var x = GetElementAtFlatIndex(input, flatIndex);

                                                                               // Compute x^3
                                                                               var x3 = Multiply(Multiply(x, x), x);

                                                                               // Compute inner expression: sqrt(2/π) * (x + 0.044715 * x^3)
                                                                               var inner = Multiply(sqrt2OverPi, Add(x, Multiply(coefficient, x3)));

                                                                               // Approximate tanh using (exp(2x) - 1) / (exp(2x) + 1)
                                                                               var exp2x = Exp(Multiply(CreateScalar<T>(2.0f), inner));
                                                                               var tanh = Divide(Subtract(exp2x, one), Add(exp2x, one));

                                                                               // Final GELU computation
                                                                               var result = Multiply(x, Multiply(half, Add(one, tanh)));
                                                                               SetElementAtFlatIndex(output, flatIndex, result);
                                                                           }
                                                                       }, cancellationToken);

        public override Task SoftmaxAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            int axis = -1,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Simple 1D softmax implementation
                                                                           if (input.Rank == 1)
                                                                           {
                                                                               var length = input.Shape[0];

                                                                               // Find max for numerical stability
                                                                               var maxVal = input[0];
                                                                               for (int i = 1; i < length; i++)
                                                                               {
                                                                                   var val = input[i];
                                                                                   if (IsGreaterThan(val, maxVal))
                                                                                       maxVal = val;
                                                                               }

                                                                               // Compute exp(x - max) and sum
                                                                               var sum = GetZero<T>();
                                                                               for (int i = 0; i < length; i++)
                                                                               {
                                                                                   var expVal = Exp(Subtract(input[i], maxVal));
                                                                                   output[i] = expVal;
                                                                                   sum = Add(sum, expVal);
                                                                               }

                                                                               // Normalize
                                                                               for (int i = 0; i < length; i++)
                                                                               {
                                                                                   output[i] = Divide(output[i], sum);
                                                                               }
                                                                           }
                                                                           else
                                                                           {
                                                                               // For multi-dimensional tensors, copy input to output as placeholder
                                                                               var totalElements = input.Length;
                                                                               for (long i = 0; i < totalElements; i++)
                                                                               {
                                                                                   var flatIndex = ComputeFlatIndex(input.Shape, i);
                                                                                   var val = GetElementAtFlatIndex(input, flatIndex);
                                                                                   SetElementAtFlatIndex(output, flatIndex, val);
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task LayerNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Layer normalization across the last dimension
                                                                           var batchSize = input.Shape[0];
                                                                           var featureSize = input.Shape[input.Rank - 1];
                                                                           var numElements = (int)(input.Length / featureSize);

                                                                           for (int i = 0; i < numElements; i++)
                                                                           {
                                                                               // Compute mean
                                                                               var sum = GetZero<T>();
                                                                               for (int j = 0; j < featureSize; j++)
                                                                               {
                                                                                   var indices = ComputeIndicesFromFlat(input.Shape, i * featureSize + j);
                                                                                   sum = Add(sum, GetElementAtFlatIndex(input, indices));
                                                                               }
                                                                               var mean = Divide(sum, CreateScalar<T>(featureSize));

                                                                               // Compute variance
                                                                               var variance = GetZero<T>();
                                                                               for (int j = 0; j < featureSize; j++)
                                                                               {
                                                                                   var indices = ComputeIndicesFromFlat(input.Shape, i * featureSize + j);
                                                                                   var diff = Subtract(GetElementAtFlatIndex(input, indices), mean);
                                                                                   variance = Add(variance, Multiply(diff, diff));
                                                                               }
                                                                               variance = Divide(variance, CreateScalar<T>(featureSize));

                                                                               // Compute normalized output
                                                                               var stdDev = Sqrt(Add(variance, epsilon));
                                                                               for (int j = 0; j < featureSize; j++)
                                                                               {
                                                                                   var indices = ComputeIndicesFromFlat(input.Shape, i * featureSize + j);
                                                                                   var normalized = Divide(Subtract(GetElementAtFlatIndex(input, indices), mean), stdDev);

                                                                                   // Apply scale and shift
                                                                                   var scaled = Multiply(normalized, gamma[j]);
                                                                                   var result = Add(scaled, beta[j]);
                                                                                   SetElementAtFlatIndex(output, indices, result);
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task BatchNormAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            ITensor<T> mean,
            ITensor<T> variance,
            ITensor<T> gamma,
            ITensor<T> beta,
            T epsilon,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Generic batch normalization implementation
                                                                           // BatchNorm formula: y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
                                                                           var batchSize = input.Shape[0];
                                                                           var channels = input.Shape[1];
                                                                           var height = input.Shape[2];
                                                                           var width = input.Shape[3];

                                                                           unsafe
                                                                           {
                                                                               var inputPtr = (T*)input.GetDataPointer();
                                                                               var outputPtr = (T*)output.GetDataPointer();
                                                                               var meanPtr = (T*)mean.GetDataPointer();
                                                                               var variancePtr = (T*)variance.GetDataPointer();
                                                                               var gammaPtr = (T*)gamma.GetDataPointer();
                                                                               var betaPtr = (T*)beta.GetDataPointer();

                                                                               for (int b = 0; b < batchSize; b++)
                                                                               {
                                                                                   for (int c = 0; c < channels; c++)
                                                                                   {
                                                                                       for (int h = 0; h < height; h++)
                                                                                       {
                                                                                           for (int w = 0; w < width; w++)
                                                                                           {
                                                                                               var inputIdx = b * channels * height * width +
                                                                                                            c * height * width + h * width + w;

                                                                                               if (typeof(T) == typeof(float))
                                                                                               {
                                                                                                   var x = (float)(object)inputPtr[inputIdx];
                                                                                                   var mu = (float)(object)meanPtr[c];
                                                                                                   var sigma2 = (float)(object)variancePtr[c];
                                                                                                   var g = (float)(object)gammaPtr[c];
                                                                                                   var b_val = (float)(object)betaPtr[c];
                                                                                                   var eps = (float)(object)epsilon;

                                                                                                   var normalized = (x - mu) / MathF.Sqrt(sigma2 + eps);
                                                                                                   var result = normalized * g + b_val;
                                                                                                   outputPtr[inputIdx] = (T)(object)result;
                                                                                               }
                                                                                               else if (typeof(T) == typeof(double))
                                                                                               {
                                                                                                   var x = (double)(object)inputPtr[inputIdx];
                                                                                                   var mu = (double)(object)meanPtr[c];
                                                                                                   var sigma2 = (double)(object)variancePtr[c];
                                                                                                   var g = (double)(object)gammaPtr[c];
                                                                                                   var b_val = (double)(object)betaPtr[c];
                                                                                                   var eps = (double)(object)epsilon;

                                                                                                   var normalized = (x - mu) / Math.Sqrt(sigma2 + eps);
                                                                                                   var result = normalized * g + b_val;
                                                                                                   outputPtr[inputIdx] = (T)(object)result;
                                                                                               }
                                                                                           }
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task MaxPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           var batchSize = input.Shape[0];
                                                                           var channels = input.Shape[1];
                                                                           var inputHeight = input.Shape[2];
                                                                           var inputWidth = input.Shape[3];

                                                                           var outputHeight = (inputHeight + 2 * padding.Height - poolSize.Height) / stride.Height + 1;
                                                                           var outputWidth = (inputWidth + 2 * padding.Width - poolSize.Width) / stride.Width + 1;

                                                                           for (int b = 0; b < batchSize; b++)
                                                                           {
                                                                               for (int c = 0; c < channels; c++)
                                                                               {
                                                                                   for (int oh = 0; oh < outputHeight; oh++)
                                                                                   {
                                                                                       for (int ow = 0; ow < outputWidth; ow++)
                                                                                       {
                                                                                           var maxVal = CreateScalar<T>(-1e9f); // Large negative value
                                                                                           bool hasValidValue = false;

                                                                                           for (int ph = 0; ph < poolSize.Height; ph++)
                                                                                           {
                                                                                               for (int pw = 0; pw < poolSize.Width; pw++)
                                                                                               {
                                                                                                   var ih = oh * stride.Height - padding.Height + ph;
                                                                                                   var iw = ow * stride.Width - padding.Width + pw;

                                                                                                   if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                                                                                   {
                                                                                                       var val = input[b, c, ih, iw];
                                                                                                       if (!hasValidValue || IsGreaterThan(val, maxVal))
                                                                                                       {
                                                                                                           maxVal = val;
                                                                                                           hasValidValue = true;
                                                                                                       }
                                                                                                   }
                                                                                               }
                                                                                           }

                                                                                           output[b, c, oh, ow] = hasValidValue ? maxVal : GetZero<T>();
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task AvgPool2DAsync<T>(
            ITensor<T> input,
            ITensor<T> output,
            Size2D poolSize,
            Size2D stride,
            Size2D padding,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           var batchSize = input.Shape[0];
                                                                           var channels = input.Shape[1];
                                                                           var inputHeight = input.Shape[2];
                                                                           var inputWidth = input.Shape[3];

                                                                           var outputHeight = (inputHeight + 2 * padding.Height - poolSize.Height) / stride.Height + 1;
                                                                           var outputWidth = (inputWidth + 2 * padding.Width - poolSize.Width) / stride.Width + 1;

                                                                           for (int b = 0; b < batchSize; b++)
                                                                           {
                                                                               for (int c = 0; c < channels; c++)
                                                                               {
                                                                                   for (int oh = 0; oh < outputHeight; oh++)
                                                                                   {
                                                                                       for (int ow = 0; ow < outputWidth; ow++)
                                                                                       {
                                                                                           var sum = GetZero<T>();
                                                                                           int validCount = 0;

                                                                                           for (int ph = 0; ph < poolSize.Height; ph++)
                                                                                           {
                                                                                               for (int pw = 0; pw < poolSize.Width; pw++)
                                                                                               {
                                                                                                   var ih = oh * stride.Height - padding.Height + ph;
                                                                                                   var iw = ow * stride.Width - padding.Width + pw;

                                                                                                   if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                                                                                   {
                                                                                                       sum = Add(sum, input[b, c, ih, iw]);
                                                                                                       validCount++;
                                                                                                   }
                                                                                               }
                                                                                           }

                                                                                           output[b, c, oh, ow] = validCount > 0 ? Divide(sum, CreateScalar<T>(validCount)) : GetZero<T>();
                                                                                       }
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task QuantizeToInt8Async<T>(
            ITensor<T> input,
            ITensor<sbyte> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Generic quantization implementation
                                                                           // Quantization formula: q = round(x / scale) + zero_point
                                                                           var totalElements = input.Length;

                                                                           unsafe
                                                                           {
                                                                               var inputPtr = (T*)input.GetDataPointer();
                                                                               var outputPtr = (sbyte*)output.GetDataPointer();

                                                                               for (long i = 0; i < totalElements; i++)
                                                                               {
                                                                                   if (typeof(T) == typeof(float))
                                                                                   {
                                                                                       var x = (float)(object)inputPtr[i];
                                                                                       var scaleVal = (float)(object)scale;
                                                                                       var quantized = (int)MathF.Round(x / scaleVal) + zeroPoint;

                                                                                       // Clamp to int8 range
                                                                                       quantized = Math.Max(-128, Math.Min(127, quantized));
                                                                                       outputPtr[i] = (sbyte)quantized;
                                                                                   }
                                                                                   else if (typeof(T) == typeof(double))
                                                                                   {
                                                                                       var x = (double)(object)inputPtr[i];
                                                                                       var scaleVal = (double)(object)scale;
                                                                                       var quantized = (int)Math.Round(x / scaleVal) + zeroPoint;

                                                                                       // Clamp to int8 range
                                                                                       quantized = Math.Max(-128, Math.Min(127, quantized));
                                                                                       outputPtr[i] = (sbyte)quantized;
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        public override Task DequantizeFromInt8Async<T>(
            ITensor<sbyte> input,
            ITensor<T> output,
            T scale,
            sbyte zeroPoint,
            CancellationToken cancellationToken = default) => Task.Run(() =>
                                                                       {
                                                                           cancellationToken.ThrowIfCancellationRequested();

                                                                           // Generic dequantization implementation
                                                                           // Dequantization formula: x = (q - zero_point) * scale
                                                                           var totalElements = input.Length;

                                                                           unsafe
                                                                           {
                                                                               var inputPtr = (sbyte*)input.GetDataPointer();
                                                                               var outputPtr = (T*)output.GetDataPointer();

                                                                               for (long i = 0; i < totalElements; i++)
                                                                               {
                                                                                   var quantized = inputPtr[i];

                                                                                   if (typeof(T) == typeof(float))
                                                                                   {
                                                                                       var scaleVal = (float)(object)scale;
                                                                                       var dequantized = (quantized - zeroPoint) * scaleVal;
                                                                                       outputPtr[i] = (T)(object)dequantized;
                                                                                   }
                                                                                   else if (typeof(T) == typeof(double))
                                                                                   {
                                                                                       var scaleVal = (double)(object)scale;
                                                                                       var dequantized = (quantized - zeroPoint) * scaleVal;
                                                                                       outputPtr[i] = (T)(object)dequantized;
                                                                                   }
                                                                               }
                                                                           }
                                                                       }, cancellationToken);

        // Helper methods for generic arithmetic operations
        private static T GetZero<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)0.0f;
            if (typeof(T) == typeof(double)) return (T)(object)0.0;
            if (typeof(T) == typeof(int)) return (T)(object)0;
            if (typeof(T) == typeof(Half)) return (T)(object)Half.Zero;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Add<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)((float)(object)a + (float)(object)b);
            if (typeof(T) == typeof(double)) return (T)(object)((double)(object)a + (double)(object)b);
            if (typeof(T) == typeof(int)) return (T)(object)((int)(object)a + (int)(object)b);
            if (typeof(T) == typeof(Half)) return (T)(object)((Half)(object)a + (Half)(object)b);
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Multiply<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)((float)(object)a * (float)(object)b);
            if (typeof(T) == typeof(double)) return (T)(object)((double)(object)a * (double)(object)b);
            if (typeof(T) == typeof(int)) return (T)(object)((int)(object)a * (int)(object)b);
            if (typeof(T) == typeof(Half)) return (T)(object)((Half)(object)a * (Half)(object)b);
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Subtract<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)((float)(object)a - (float)(object)b);
            if (typeof(T) == typeof(double)) return (T)(object)((double)(object)a - (double)(object)b);
            if (typeof(T) == typeof(int)) return (T)(object)((int)(object)a - (int)(object)b);
            if (typeof(T) == typeof(Half)) return (T)(object)((Half)(object)a - (Half)(object)b);
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Divide<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)((float)(object)a / (float)(object)b);
            if (typeof(T) == typeof(double)) return (T)(object)((double)(object)a / (double)(object)b);
            if (typeof(T) == typeof(int)) return (T)(object)((int)(object)a / (int)(object)b);
            if (typeof(T) == typeof(Half)) return (T)(object)((Half)(object)a / (Half)(object)b);
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Exp<T>(T x) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)MathF.Exp((float)(object)x);
            if (typeof(T) == typeof(double)) return (T)(object)Math.Exp((double)(object)x);
            if (typeof(T) == typeof(Half)) return (T)(object)(Half)MathF.Exp((float)(Half)(object)x);
            throw new NotSupportedException($"Type {typeof(T)} not supported for Exp");
        }

        private static bool IsGreaterThan<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (float)(object)a > (float)(object)b;
            if (typeof(T) == typeof(double)) return (double)(object)a > (double)(object)b;
            if (typeof(T) == typeof(int)) return (int)(object)a > (int)(object)b;
            if (typeof(T) == typeof(Half)) return (Half)(object)a > (Half)(object)b;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static int[] ComputeFlatIndex(TensorShape shape, long flatIndex)
        {
            var indices = new int[shape.Rank];
            for (int i = shape.Rank - 1; i >= 0; i--)
            {
                indices[i] = (int)(flatIndex % shape[i]);
                flatIndex /= shape[i];
            }
            return indices;
        }

        private static T GetElementAtFlatIndex<T>(ITensor<T> tensor, int[] indices) where T : unmanaged => tensor[indices];

        private static void SetElementAtFlatIndex<T>(ITensor<T> tensor, int[] indices, T value) where T : unmanaged => tensor[indices] = value;

        private static T CreateScalar<T>(float value) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)value;
            if (typeof(T) == typeof(double)) return (T)(object)(double)value;
            if (typeof(T) == typeof(Half)) return (T)(object)(Half)value;
            if (typeof(T) == typeof(int)) return (T)(object)(int)value;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T Sqrt<T>(T x) where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)MathF.Sqrt((float)(object)x);
            if (typeof(T) == typeof(double)) return (T)(object)Math.Sqrt((double)(object)x);
            if (typeof(T) == typeof(Half)) return (T)(object)(Half)MathF.Sqrt((float)(Half)(object)x);
            throw new NotSupportedException($"Type {typeof(T)} not supported for Sqrt");
        }

        private static int[] ComputeIndicesFromFlat(TensorShape shape, long flatIndex)
        {
            var indices = new int[shape.Rank];
            for (int i = shape.Rank - 1; i >= 0; i--)
            {
                indices[i] = (int)(flatIndex % shape[i]);
                flatIndex /= shape[i];
            }
            return indices;
        }
    }
}