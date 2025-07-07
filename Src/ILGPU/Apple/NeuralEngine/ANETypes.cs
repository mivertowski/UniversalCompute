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

using System;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Apple Neural Engine activation function types.
    /// </summary>
    public enum ANEActivation
    {
        /// <summary>
        /// No activation function.
        /// </summary>
        None = 0,

        /// <summary>
        /// ReLU activation.
        /// </summary>
        ReLU = 1,

        /// <summary>
        /// GELU activation.
        /// </summary>
        GELU = 2,

        /// <summary>
        /// Swish activation.
        /// </summary>
        Swish = 3,

        /// <summary>
        /// Sigmoid activation.
        /// </summary>
        Sigmoid = 4,

        /// <summary>
        /// Tanh activation.
        /// </summary>
        Tanh = 5
    }

    /// <summary>
    /// ANE precision types.
    /// </summary>
    public enum ANEPrecision
    {
        /// <summary>
        /// 32-bit floating point precision.
        /// </summary>
        Float32 = 0,

        /// <summary>
        /// 16-bit floating point precision.
        /// </summary>
        Float16 = 1,

        /// <summary>
        /// 8-bit integer precision.
        /// </summary>
        Int8 = 2,

        /// <summary>
        /// Mixed precision for optimal performance.
        /// </summary>
        Mixed = 3
    }

    /// <summary>
    /// Configuration for ANE convolution operations.
    /// </summary>
    public sealed class ANEConvolutionConfig
    {
        /// <summary>
        /// Gets or sets the input tensor shape (N, C, H, W).
        /// </summary>
        public (int N, int C, int H, int W) InputShape { get; set; }

        /// <summary>
        /// Gets or sets the kernel size (height, width).
        /// </summary>
        public (int Height, int Width) KernelSize { get; set; } = (3, 3);

        /// <summary>
        /// Gets or sets the stride (height, width).
        /// </summary>
        public (int Height, int Width) Stride { get; set; } = (1, 1);

        /// <summary>
        /// Gets or sets the padding (height, width).
        /// </summary>
        public (int Height, int Width) Padding { get; set; } = (0, 0);

        /// <summary>
        /// Gets or sets the activation function.
        /// </summary>
        public ANEActivation Activation { get; set; } = ANEActivation.None;

        /// <summary>
        /// Gets or sets the precision for the operation.
        /// </summary>
        public ANEPrecision Precision { get; set; } = ANEPrecision.Float16;

        /// <summary>
        /// Creates a default convolution configuration.
        /// </summary>
        public static ANEConvolutionConfig Default => new();
    }

    /// <summary>
    /// Configuration for ANE attention operations.
    /// </summary>
    public sealed class ANEAttentionConfig
    {
        /// <summary>
        /// Gets or sets the sequence length.
        /// </summary>
        public int SequenceLength { get; set; }

        /// <summary>
        /// Gets or sets the hidden dimension.
        /// </summary>
        public int HiddenDimension { get; set; }

        /// <summary>
        /// Gets or sets the number of attention heads.
        /// </summary>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets whether to use causal masking.
        /// </summary>
        public bool UseCausalMask { get; set; }

        /// <summary>
        /// Gets or sets the dropout probability.
        /// </summary>
        public float DropoutRate { get; set; }

        /// <summary>
        /// Gets or sets the precision for the operation.
        /// </summary>
        public ANEPrecision Precision { get; set; } = ANEPrecision.Float16;

        /// <summary>
        /// Creates a default attention configuration.
        /// </summary>
        public static ANEAttentionConfig Default => new();
    }

    /// <summary>
    /// Core ML model handle for ANE execution.
    /// </summary>
    public sealed class ANECoreMLModel : IDisposable
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new ANE Core ML model.
        /// </summary>
        /// <param name="modelHandle">Native model handle.</param>
        internal ANECoreMLModel(IntPtr modelHandle)
        {
            ModelHandle = modelHandle;
        }

        /// <summary>
        /// Gets the native model handle.
        /// </summary>
        internal IntPtr ModelHandle { get; private set; }

        /// <summary>
        /// Disposes the model.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (ModelHandle != IntPtr.Zero)
                {
                    // Release native model handle
                    ModelHandle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
