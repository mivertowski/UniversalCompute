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
using System.Runtime.InteropServices;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// NPU execution modes.
    /// </summary>
    public enum NPUExecutionMode
    {
        /// <summary>
        /// High performance mode with maximum throughput.
        /// </summary>
        HighPerformance = 0,

        /// <summary>
        /// Balanced mode optimizing performance and power.
        /// </summary>
        Balanced = 1,

        /// <summary>
        /// Low power mode for battery optimization.
        /// </summary>
        LowPower = 2,

        /// <summary>
        /// Ultra low power mode for extreme battery savings.
        /// </summary>
        UltraLowPower = 3
    }

    /// <summary>
    /// NPU thermal states.
    /// </summary>
    public enum NPUThermalState
    {
        /// <summary>
        /// Normal operating temperature.
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Elevated temperature with minor throttling.
        /// </summary>
        Warm = 1,

        /// <summary>
        /// High temperature with moderate throttling.
        /// </summary>
        Hot = 2,

        /// <summary>
        /// Critical temperature with severe throttling.
        /// </summary>
        Critical = 3
    }

    /// <summary>
    /// NPU precision types.
    /// </summary>
    public enum NPUPrecision
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
        /// Brain floating point 16-bit precision.
        /// </summary>
        BFloat16 = 2,

        /// <summary>
        /// 8-bit integer precision.
        /// </summary>
        Int8 = 3,

        /// <summary>
        /// 4-bit integer precision.
        /// </summary>
        Int4 = 4,

        /// <summary>
        /// Mixed precision for optimal performance.
        /// </summary>
        Mixed = 5
    }

    /// <summary>
    /// NPU operation types.
    /// </summary>
    public enum NPUOperationType
    {
        /// <summary>
        /// Matrix multiplication operation.
        /// </summary>
        MatrixMultiply = 0,

        /// <summary>
        /// Convolution operation.
        /// </summary>
        Convolution = 1,

        /// <summary>
        /// Pooling operation.
        /// </summary>
        Pooling = 2,

        /// <summary>
        /// Activation function.
        /// </summary>
        Activation = 3,

        /// <summary>
        /// Normalization operation.
        /// </summary>
        Normalization = 4,

        /// <summary>
        /// Attention mechanism.
        /// </summary>
        Attention = 5,

        /// <summary>
        /// Element-wise operation.
        /// </summary>
        ElementWise = 6,

        /// <summary>
        /// Reduction operation.
        /// </summary>
        Reduction = 7
    }

    /// <summary>
    /// NPU activation function types.
    /// </summary>
    public enum NPUActivation
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
        Tanh = 5,

        /// <summary>
        /// Leaky ReLU activation.
        /// </summary>
        LeakyReLU = 6
    }

    /// <summary>
    /// NPU memory layout types.
    /// </summary>
    public enum NPUMemoryLayout
    {
        /// <summary>
        /// Row major layout.
        /// </summary>
        RowMajor = 0,

        /// <summary>
        /// Column major layout.
        /// </summary>
        ColumnMajor = 1,

        /// <summary>
        /// Blocked layout for optimal cache utilization.
        /// </summary>
        Blocked = 2,

        /// <summary>
        /// Tiled layout for tensor operations.
        /// </summary>
        Tiled = 3
    }

    /// <summary>
    /// Configuration for NPU quantization operations.
    /// </summary>
    public sealed class NPUQuantizationConfig
    {
        /// <summary>
        /// Gets or sets the quantization precision.
        /// </summary>
        public NPUPrecision Precision { get; set; } = NPUPrecision.Int8;

        /// <summary>
        /// Gets or sets whether to use symmetric quantization.
        /// </summary>
        public bool UseSymmetricQuantization { get; set; } = true;

        /// <summary>
        /// Gets or sets the quantization scale factor.
        /// </summary>
        public float Scale { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the zero point for asymmetric quantization.
        /// </summary>
        public int ZeroPoint { get; set; } = 0;

        /// <summary>
        /// Creates a default quantization configuration.
        /// </summary>
        public static NPUQuantizationConfig Default => new();
    }

    /// <summary>
    /// OpenVINO model handle for NPU execution.
    /// </summary>
    public sealed class NPUOpenVINOModel : IDisposable
    {
        private IntPtr _modelHandle;
        private bool _disposed;

        /// <summary>
        /// Initializes a new NPU OpenVINO model.
        /// </summary>
        /// <param name="modelHandle">Native model handle.</param>
        internal NPUOpenVINOModel(IntPtr modelHandle)
        {
            _modelHandle = modelHandle;
        }

        /// <summary>
        /// Gets the native model handle.
        /// </summary>
        internal IntPtr ModelHandle => _modelHandle;

        /// <summary>
        /// Disposes the model.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_modelHandle != IntPtr.Zero)
                {
                    // Release native model handle
                    _modelHandle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// NPU execution configuration.
    /// </summary>
    public sealed class NPUExecutionConfig
    {
        /// <summary>
        /// Gets or sets the execution mode.
        /// </summary>
        public NPUExecutionMode Mode { get; set; } = NPUExecutionMode.Balanced;

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; } = 1;

        /// <summary>
        /// Gets or sets the number of threads.
        /// </summary>
        public int NumThreads { get; set; } = 1;

        /// <summary>
        /// Gets or sets whether to enable profiling.
        /// </summary>
        public bool EnableProfiling { get; set; } = false;
    }

    /// <summary>
    /// NPU optimization flags.
    /// </summary>
    [Flags]
    public enum NPUOptimizationFlags
    {
        /// <summary>
        /// No optimizations.
        /// </summary>
        None = 0,

        /// <summary>
        /// Enable kernel fusion.
        /// </summary>
        KernelFusion = 1,

        /// <summary>
        /// Enable memory optimization.
        /// </summary>
        MemoryOptimization = 2,

        /// <summary>
        /// Enable quantization.
        /// </summary>
        Quantization = 4,

        /// <summary>
        /// Enable all optimizations.
        /// </summary>
        All = KernelFusion | MemoryOptimization | Quantization
    }

    /// <summary>
    /// NPU power modes.
    /// </summary>
    public enum NPUPowerMode
    {
        /// <summary>
        /// High performance mode.
        /// </summary>
        HighPerformance = 0,

        /// <summary>
        /// Balanced mode.
        /// </summary>
        Balanced = 1,

        /// <summary>
        /// Low power mode.
        /// </summary>
        LowPower = 2,

        /// <summary>
        /// Ultra low power mode.
        /// </summary>
        UltraLowPower = 3
    }

    /// <summary>
    /// NPU tensor layout.
    /// </summary>
    public enum NPUTensorLayout
    {
        /// <summary>
        /// NCHW layout (batch, channels, height, width).
        /// </summary>
        NCHW = 0,

        /// <summary>
        /// NHWC layout (batch, height, width, channels).
        /// </summary>
        NHWC = 1,

        /// <summary>
        /// NCW layout (batch, channels, width).
        /// </summary>
        NCW = 2,

        /// <summary>
        /// NWC layout (batch, width, channels).
        /// </summary>
        NWC = 3
    }
}