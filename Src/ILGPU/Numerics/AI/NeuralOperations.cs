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

using System;
using ILGPU.Numerics;

namespace ILGPU.Numerics.AI
{
    /// <summary>
    /// Represents a neural network operation.
    /// </summary>
    public abstract class NeuralOperation
    {
        /// <summary>
        /// Gets the operation name.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Gets the operation type.
        /// </summary>
        public abstract NeuralOperationType Type { get; }

        /// <summary>
        /// Gets the input shape requirements.
        /// </summary>
        public abstract TensorShape InputShape { get; }

        /// <summary>
        /// Calculates the output shape for the given input shape.
        /// </summary>
        /// <param name="inputShape">The input tensor shape.</param>
        /// <returns>The output tensor shape.</returns>
        public abstract TensorShape CalculateOutputShape(TensorShape inputShape);
    }

    /// <summary>
    /// Represents a matrix operation.
    /// </summary>
    public abstract class MatrixOperation
    {
        /// <summary>
        /// Gets the operation name.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Gets the operation type.
        /// </summary>
        public abstract MatrixOperationType Type { get; }

        /// <summary>
        /// Calculates the output shape for the given input shapes.
        /// </summary>
        /// <param name="aShape">The first matrix shape.</param>
        /// <param name="bShape">The second matrix shape.</param>
        /// <returns>The output matrix shape.</returns>
        public abstract TensorShape CalculateOutputShape(TensorShape aShape, TensorShape bShape);
    }


    /// <summary>
    /// Convolution operation parameters.
    /// </summary>
    public sealed class ConvolutionParameters
    {
        /// <summary>
        /// Gets or sets the kernel size.
        /// </summary>
        public Size2D KernelSize { get; set; } = new Size2D(3, 3);

        /// <summary>
        /// Gets or sets the stride.
        /// </summary>
        public Size2D Stride { get; set; } = new Size2D(1, 1);

        /// <summary>
        /// Gets or sets the padding.
        /// </summary>
        public Size2D Padding { get; set; } = new Size2D(0, 0);

        /// <summary>
        /// Gets or sets the dilation.
        /// </summary>
        public Size2D Dilation { get; set; } = new Size2D(1, 1);

        /// <summary>
        /// Gets or sets the number of groups.
        /// </summary>
        public int Groups { get; set; } = 1;

        /// <summary>
        /// Gets or sets the activation function.
        /// </summary>
        public ActivationFunction Activation { get; set; } = ActivationFunction.None;
    }

    /// <summary>
    /// Attention operation parameters.
    /// </summary>
    public sealed class AttentionParameters
    {
        /// <summary>
        /// Gets or sets the number of attention heads.
        /// </summary>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the head dimension.
        /// </summary>
        public int HeadDim { get; set; } = 64;

        /// <summary>
        /// Gets or sets the dropout probability.
        /// </summary>
        public float Dropout { get; set; }

        /// <summary>
        /// Gets or sets whether to use causal masking.
        /// </summary>
        public bool CausalMask { get; set; }

        /// <summary>
        /// Gets or sets the scaling factor.
        /// </summary>
        public float Scale { get; set; } = 1.0f;
    }

    /// <summary>
    /// Matrix multiplication configuration.
    /// </summary>
    public sealed class MatMulConfiguration
    {
        /// <summary>
        /// Gets or sets the M dimension.
        /// </summary>
        public int M { get; set; }

        /// <summary>
        /// Gets or sets the K dimension.
        /// </summary>
        public int K { get; set; }

        /// <summary>
        /// Gets or sets the N dimension.
        /// </summary>
        public int N { get; set; }

        /// <summary>
        /// Gets or sets whether to use BFloat16.
        /// </summary>
        public bool UseBF16 { get; set; }

        /// <summary>
        /// Gets or sets whether to use sparsity.
        /// </summary>
        public bool UseSparsity { get; set; }

        /// <summary>
        /// Gets or sets the alpha scalar.
        /// </summary>
        public float Alpha { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the beta scalar.
        /// </summary>
        public float Beta { get; set; }
    }

    /// <summary>
    /// Model optimization options.
    /// </summary>
    public sealed class OptimizationOptions
    {
        /// <summary>
        /// Gets or sets whether to enable quantization.
        /// </summary>
        public bool EnableQuantization { get; set; } = true;

        /// <summary>
        /// Gets or sets the quantization mode.
        /// </summary>
        public QuantizationMode QuantizationMode { get; set; } = QuantizationMode.INT8;

        /// <summary>
        /// Gets or sets whether to enable pruning.
        /// </summary>
        public bool EnablePruning { get; set; }

        /// <summary>
        /// Gets or sets the pruning ratio.
        /// </summary>
        public float PruningRatio { get; set; } = 0.5f;

        /// <summary>
        /// Gets or sets whether to enable kernel fusion.
        /// </summary>
        public bool EnableKernelFusion { get; set; } = true;

        /// <summary>
        /// Gets or sets the target batch size.
        /// </summary>
        public int TargetBatchSize { get; set; } = 1;
    }

    /// <summary>
    /// Neural operation types.
    /// </summary>
    public enum NeuralOperationType
    {
        /// <summary>
        /// Convolution operation.
        /// </summary>
        Convolution,

        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        MatMul,

        /// <summary>
        /// Attention mechanism.
        /// </summary>
        Attention,

        /// <summary>
        /// Pooling operation.
        /// </summary>
        Pooling,

        /// <summary>
        /// Normalization operation.
        /// </summary>
        Normalization,

        /// <summary>
        /// Activation function.
        /// </summary>
        Activation
    }

    /// <summary>
    /// Matrix operation types.
    /// </summary>
    public enum MatrixOperationType
    {
        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        MatMul,

        /// <summary>
        /// Element-wise addition.
        /// </summary>
        Add,

        /// <summary>
        /// Element-wise multiplication.
        /// </summary>
        Mul,

        /// <summary>
        /// Matrix transpose.
        /// </summary>
        Transpose,

        /// <summary>
        /// Matrix inverse.
        /// </summary>
        Inverse
    }

    /// <summary>
    /// Activation function types.
    /// </summary>
    public enum ActivationFunction
    {
        /// <summary>
        /// No activation.
        /// </summary>
        None,

        /// <summary>
        /// ReLU activation.
        /// </summary>
        ReLU,

        /// <summary>
        /// GELU activation.
        /// </summary>
        GELU,

        /// <summary>
        /// Sigmoid activation.
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Tanh activation.
        /// </summary>
        Tanh,

        /// <summary>
        /// Swish activation.
        /// </summary>
        Swish
    }

    /// <summary>
    /// Quantization modes.
    /// </summary>
    public enum QuantizationMode
    {
        /// <summary>
        /// 8-bit integer quantization.
        /// </summary>
        INT8,

        /// <summary>
        /// 4-bit integer quantization.
        /// </summary>
        INT4,

        /// <summary>
        /// BFloat16 quantization.
        /// </summary>
        BFloat16,

        /// <summary>
        /// Mixed precision.
        /// </summary>
        Mixed
    }
}