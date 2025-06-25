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

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Base class for NPU operations.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public abstract class NPUOperation<T> where T : unmanaged
    {
        /// <summary>
        /// Gets the operation name.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Gets the input tensor.
        /// </summary>
        public abstract ITensor<T> Input { get; }

        /// <summary>
        /// Gets the expected output shape.
        /// </summary>
        public abstract TensorShape OutputShape { get; }

        /// <summary>
        /// Executes the operation on the NPU.
        /// </summary>
        /// <param name="npu">The NPU accelerator.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The output tensor.</returns>
        public abstract Task<ITensor<T>> ExecuteAsync(IntelNPUAccelerator npu, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Convolution operation for NPU execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class ConvolutionOperation<T> : NPUOperation<T> where T : unmanaged
    {
        /// <summary>
        /// Initializes a new instance of the ConvolutionOperation class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernel">The convolution kernel.</param>
        /// <param name="parameters">The convolution parameters.</param>
        public ConvolutionOperation(ITensor<T> input, ITensor<T> kernel, ConvolutionParameters parameters)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            
            // Calculate output shape
            var outputHeight = (input.Shape[2] + 2 * parameters.Padding.Height - kernel.Shape[2]) / parameters.Stride.Height + 1;
            var outputWidth = (input.Shape[3] + 2 * parameters.Padding.Width - kernel.Shape[3]) / parameters.Stride.Width + 1;
            OutputShape = new TensorShape(input.Shape[0], kernel.Shape[0], outputHeight, outputWidth);
        }

        /// <summary>
        /// Gets the operation name.
        /// </summary>
        public override string Name => "Convolution2D";

        /// <summary>
        /// Gets the input tensor.
        /// </summary>
        public override ITensor<T> Input { get; }

        /// <summary>
        /// Gets the convolution kernel.
        /// </summary>
        public ITensor<T> Kernel { get; }

        /// <summary>
        /// Gets the convolution parameters.
        /// </summary>
        public ConvolutionParameters Parameters { get; }

        /// <summary>
        /// Gets the expected output shape.
        /// </summary>
        public override TensorShape OutputShape { get; }

        /// <summary>
        /// Executes the convolution on the NPU.
        /// </summary>
        /// <param name="npu">The NPU accelerator.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The output tensor.</returns>
        public override async Task<ITensor<T>> ExecuteAsync(IntelNPUAccelerator npu, CancellationToken cancellationToken = default)
        {
            var output = TensorFactory.Create<T>(OutputShape, ComputeLocation.Npu);
            
            // Execute NPU convolution kernel
            await npu.ExecuteConvolutionKernelAsync(Input, Kernel, output, Parameters, cancellationToken);
            
            return output;
        }
    }

    /// <summary>
    /// Attention operation for NPU execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <remarks>
    /// Initializes a new instance of the AttentionOperation class.
    /// </remarks>
    /// <param name="query">The query tensor.</param>
    /// <param name="key">The key tensor.</param>
    /// <param name="value">The value tensor.</param>
    /// <param name="parameters">The attention parameters.</param>
    public sealed class AttentionOperation<T>(ITensor<T> query, ITensor<T> key, ITensor<T> value, AttentionParameters parameters) : NPUOperation<T> where T : unmanaged
    {

        /// <summary>
        /// Gets the operation name.
        /// </summary>
        public override string Name => "MultiHeadAttention";

        /// <summary>
        /// Gets the query tensor (used as input for base class).
        /// </summary>
        public override ITensor<T> Input => Query;

        /// <summary>
        /// Gets the query tensor.
        /// </summary>
        public ITensor<T> Query { get; } = query ?? throw new ArgumentNullException(nameof(query));

        /// <summary>
        /// Gets the key tensor.
        /// </summary>
        public ITensor<T> Key { get; } = key ?? throw new ArgumentNullException(nameof(key));

        /// <summary>
        /// Gets the value tensor.
        /// </summary>
        public ITensor<T> Value { get; } = value ?? throw new ArgumentNullException(nameof(value));

        /// <summary>
        /// Gets the attention parameters.
        /// </summary>
        public AttentionParameters Parameters { get; } = parameters ?? throw new ArgumentNullException(nameof(parameters));

        /// <summary>
        /// Gets the expected output shape.
        /// </summary>
        public override TensorShape OutputShape { get; } = query.Shape;

        /// <summary>
        /// Executes the attention operation on the NPU.
        /// </summary>
        /// <param name="npu">The NPU accelerator.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The output tensor.</returns>
        public override async Task<ITensor<T>> ExecuteAsync(IntelNPUAccelerator npu, CancellationToken cancellationToken = default)
        {
            var output = TensorFactory.Create<T>(OutputShape, ComputeLocation.Npu);
            
            // Execute NPU attention kernel
            await npu.ExecuteAttentionKernelAsync(Query, Key, Value, output, Parameters, cancellationToken);
            
            return output;
        }
    }
}