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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Intel NPU device generations.
    /// </summary>
    public enum NPUGeneration
    {
        /// <summary>
        /// NPU is not supported on this device.
        /// </summary>
        NotSupported = 0,

        /// <summary>
        /// Unknown NPU generation.
        /// </summary>
        Unknown = 1,

        /// <summary>
        /// First generation NPU (Meteor Lake).
        /// </summary>
        NPU1 = 10,

        /// <summary>
        /// Second generation NPU (Arrow Lake).
        /// </summary>
        NPU2 = 20,

        /// <summary>
        /// Third generation NPU (Lunar Lake).
        /// </summary>
        NPU3 = 30,

        /// <summary>
        /// Fourth generation NPU (future).
        /// </summary>
        NPU4 = 40
    }

    /// <summary>
    /// Intel NPU activation functions.
    /// </summary>
    public enum NPUActivation
    {
        /// <summary>
        /// No activation function.
        /// </summary>
        None,

        /// <summary>
        /// Rectified Linear Unit (ReLU) activation.
        /// </summary>
        ReLU,

        /// <summary>
        /// Gaussian Error Linear Unit (GELU) activation.
        /// </summary>
        GELU,

        /// <summary>
        /// Swish activation function.
        /// </summary>
        Swish,

        /// <summary>
        /// Sigmoid activation function.
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Hyperbolic tangent (Tanh) activation.
        /// </summary>
        Tanh,

        /// <summary>
        /// Leaky ReLU activation.
        /// </summary>
        LeakyReLU,

        /// <summary>
        /// Parametric ReLU activation.
        /// </summary>
        PReLU
    }

    /// <summary>
    /// NPU precision modes.
    /// </summary>
    public enum NPUPrecision
    {
        /// <summary>
        /// Full precision (FP32).
        /// </summary>
        FP32,

        /// <summary>
        /// Half precision (FP16).
        /// </summary>
        FP16,

        /// <summary>
        /// 8-bit integer quantization.
        /// </summary>
        INT8,

        /// <summary>
        /// Mixed precision (INT8 input, FP16 weights).
        /// </summary>
        Mixed,

        /// <summary>
        /// Automatic precision selection.
        /// </summary>
        Auto
    }

    /// <summary>
    /// NPU tensor layout formats.
    /// </summary>
    public enum NPUTensorLayout
    {
        /// <summary>
        /// NCHW format (batch, channels, height, width).
        /// </summary>
        NCHW,

        /// <summary>
        /// NHWC format (batch, height, width, channels).
        /// </summary>
        NHWC,

        /// <summary>
        /// Blocked layout for cache optimization.
        /// </summary>
        Blocked,

        /// <summary>
        /// Planar layout for memory efficiency.
        /// </summary>
        Planar
    }

    /// <summary>
    /// NPU power management modes.
    /// </summary>
    public enum NPUPowerMode
    {
        /// <summary>
        /// Maximum performance mode.
        /// </summary>
        Performance,

        /// <summary>
        /// Balanced power and performance.
        /// </summary>
        Balanced,

        /// <summary>
        /// Power saving mode.
        /// </summary>
        PowerSaver,

        /// <summary>
        /// Ultra-low power mode.
        /// </summary>
        UltraLowPower
    }

    /// <summary>
    /// NPU cache modes for model compilation.
    /// </summary>
    public enum NPUCacheMode
    {
        /// <summary>
        /// No caching.
        /// </summary>
        None,

        /// <summary>
        /// Cache compiled models.
        /// </summary>
        Model,

        /// <summary>
        /// Cache kernels and operations.
        /// </summary>
        Kernel,

        /// <summary>
        /// Cache everything.
        /// </summary>
        All
    }

    /// <summary>
    /// NPU optimization flags.
    /// </summary>
    [Flags]
    public enum NPUOptimizationFlags : uint
    {
        /// <summary>
        /// No optimizations.
        /// </summary>
        None = 0,

        /// <summary>
        /// Enable fusion optimizations.
        /// </summary>
        Fusion = 1 << 0,

        /// <summary>
        /// Enable quantization optimizations.
        /// </summary>
        Quantization = 1 << 1,

        /// <summary>
        /// Enable memory layout optimizations.
        /// </summary>
        MemoryLayout = 1 << 2,

        /// <summary>
        /// Enable kernel scheduling optimizations.
        /// </summary>
        Scheduling = 1 << 3,

        /// <summary>
        /// Enable all optimizations.
        /// </summary>
        All = Fusion | Quantization | MemoryLayout | Scheduling
    }

    /// <summary>
    /// Represents tensor shape for Intel NPU operations.
    /// </summary>
    public readonly struct TensorShape(params int[] dimensions)
    {
        private readonly int[] _dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));

        public int this[int index] => _dimensions[index];
        public int Rank => _dimensions.Length;
        public IReadOnlyList<int> Dimensions => _dimensions;
    }

    /// <summary>
    /// Interface for tensors used in NPU operations.
    /// </summary>
    public interface ITensor<T> where T : unmanaged
    {
        TensorShape Shape { get; }
        unsafe void* GetDataPointer();
    }

    /// <summary>
    /// Compute location for tensor operations.
    /// </summary>
    public enum ComputeLocation
    {
        Cpu,
        Gpu,
        Npu
    }

    /// <summary>
    /// Factory for creating tensors.
    /// </summary>
    public static class TensorFactory
    {
        public static ITensor<T> Create<T>(TensorShape shape, ComputeLocation location) where T : unmanaged => new SimpleTensor<T>(shape);
    }

    /// <summary>
    /// Simple tensor implementation.
    /// </summary>
    internal class SimpleTensor<T> : ITensor<T> where T : unmanaged
    {
        private readonly T[] _data;

        public SimpleTensor(TensorShape shape)
        {
            Shape = shape;
            var totalElements = 1;
            for (int i = 0; i < shape.Rank; i++)
                totalElements *= shape[i];
            _data = new T[totalElements];
        }

        public TensorShape Shape { get; }

        public unsafe void* GetDataPointer()
        {
            fixed (T* ptr = _data)
                return ptr;
        }
    }

    /// <summary>
    /// Neural network definition.
    /// </summary>
    public class NeuralNetwork(string name, NeuralOperation[]? operations = null)
    {
        public string Name { get; } = name ?? throw new ArgumentNullException(nameof(name));
        public IReadOnlyList<NeuralOperation> Operations { get; } = operations ?? [];
    }

    /// <summary>
    /// Neural operation types.
    /// </summary>
    public enum NeuralOperationType
    {
        Convolution,
        MatMul,
        Attention
    }

    /// <summary>
    /// Base class for neural operations.
    /// </summary>
    public abstract class NeuralOperation
    {
        public abstract string Name { get; }
        public abstract NeuralOperationType Type { get; }
        public abstract TensorShape InputShape { get; }
        public abstract TensorShape CalculateOutputShape(TensorShape inputShape);
    }

    /// <summary>
    /// Convolution parameters.
    /// </summary>
    public class ConvolutionParameters
    {
        public (int Height, int Width) KernelSize { get; set; }
        public (int Height, int Width) Stride { get; set; } = (1, 1);
        public (int Height, int Width) Padding { get; set; } = (0, 0);
    }

    /// <summary>
    /// Attention parameters.
    /// </summary>
    public class AttentionParameters
    {
        public int NumHeads { get; set; } = 8;
        public int HeadDim { get; set; } = 64;
        public bool UseFlashAttention { get; set; } = true;
    }

    /// <summary>
    /// NPU optimization options.
    /// </summary>
    public class OptimizationOptions
    {
        public bool EnableQuantization { get; set; } = true;
        public bool OptimizeForLatency { get; set; } = true;
        public bool EnablePruning { get; set; }
    }


    /// <summary>
    /// Matrix multiplication configuration.
    /// </summary>
    public class MatMulConfiguration
    {
        public int M { get; set; }
        public int K { get; set; }
        public int N { get; set; }
        public bool UseBF16 { get; set; }
        public bool UseSparsity { get; set; }
    }

    /// <summary>
    /// Intel NPU capabilities.
    /// </summary>
    public sealed class NPUCapabilities
    {
        /// <summary>
        /// Gets whether the NPU is available.
        /// </summary>
        public bool IsAvailable { get; internal set; }

        /// <summary>
        /// Gets the NPU generation.
        /// </summary>
        public NPUGeneration Generation { get; internal set; }

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string DeviceName { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the maximum TOPS (Tera Operations Per Second) performance.
        /// </summary>
        public double MaxTOPS { get; internal set; }

        /// <summary>
        /// Gets the number of compute units in the NPU.
        /// </summary>
        public int NumComputeUnits { get; internal set; }

        /// <summary>
        /// Gets the maximum input tensor width.
        /// </summary>
        public int MaxInputWidth { get; internal set; }

        /// <summary>
        /// Gets the maximum input tensor height.
        /// </summary>
        public int MaxInputHeight { get; internal set; }

        /// <summary>
        /// Gets the optimal batch size for operations.
        /// </summary>
        public int OptimalBatchSize { get; internal set; }

        /// <summary>
        /// Gets the maximum concurrent inferences.
        /// </summary>
        public int MaxConcurrentInferences { get; internal set; }

        /// <summary>
        /// Gets the maximum shared memory per compute unit.
        /// </summary>
        public long MaxSharedMemoryPerUnit { get; internal set; }

        /// <summary>
        /// Gets the maximum constant memory.
        /// </summary>
        public long MaxConstantMemory { get; internal set; }

        /// <summary>
        /// Gets the memory bandwidth in GB/s.
        /// </summary>
        public double MemoryBandwidth { get; internal set; }

        /// <summary>
        /// Gets whether FP16 precision is supported.
        /// </summary>
        public bool SupportsFloat16 { get; internal set; }

        /// <summary>
        /// Gets whether INT8 quantization is supported.
        /// </summary>
        public bool SupportsInt8 { get; internal set; }

        /// <summary>
        /// Gets whether mixed precision is supported.
        /// </summary>
        public bool SupportsMixedPrecision { get; internal set; }

        /// <summary>
        /// Gets whether dynamic batching is supported.
        /// </summary>
        public bool SupportsDynamicBatching { get; internal set; }

        /// <summary>
        /// Gets whether OpenVINO integration is supported.
        /// </summary>
        public bool SupportsOpenVINO { get; internal set; }

        /// <summary>
        /// Gets the total memory size available to NPU.
        /// </summary>
        public ulong MemorySize { get; internal set; }

        /// <summary>
        /// Queries NPU capabilities from the system.
        /// </summary>
        /// <returns>NPU capabilities structure.</returns>
        public static NPUCapabilities Query()
        {
            try
            {
                var nativeCapabilities = Native.NPUNative.QueryCapabilities();
                
                return new NPUCapabilities
                {
                    IsAvailable = nativeCapabilities.IsAvailable != 0,
                    Generation = (NPUGeneration)nativeCapabilities.Generation,
                    DeviceName = nativeCapabilities.DeviceName,
                    MaxTOPS = nativeCapabilities.MaxTOPS,
                    NumComputeUnits = nativeCapabilities.NumComputeUnits,
                    MaxInputWidth = 4096,  // Typical NPU limits
                    MaxInputHeight = 4096,
                    OptimalBatchSize = nativeCapabilities.OptimalBatchSize,
                    MaxConcurrentInferences = 8,
                    MaxSharedMemoryPerUnit = 2 * 1024 * 1024, // 2MB per unit
                    MaxConstantMemory = nativeCapabilities.MaxConstantMemory,
                    MemoryBandwidth = 100.0, // GB/s for typical NPU
                    SupportsFloat16 = nativeCapabilities.SupportsFloat16 != 0,
                    SupportsInt8 = nativeCapabilities.SupportsInt8 != 0,
                    SupportsMixedPrecision = nativeCapabilities.SupportsMixedPrecision != 0,
                    SupportsDynamicBatching = nativeCapabilities.SupportsDynamicBatching != 0,
                    SupportsOpenVINO = nativeCapabilities.SupportsOpenVINO != 0,
                    MemorySize = nativeCapabilities.MemorySize
                };
            }
            catch
            {
                // Return default capabilities if query fails
                return new NPUCapabilities
                {
                    IsAvailable = false,
                    Generation = NPUGeneration.NotSupported
                };
            }
        }
    }

    /// <summary>
    /// Configuration for NPU convolution operations.
    /// </summary>
    public sealed class NPUConvolutionConfig
    {
        /// <summary>
        /// Gets or sets the input tensor shape (N, C, H, W).
        /// </summary>
        public (int N, int C, int H, int W) InputShape { get; set; }

        /// <summary>
        /// Gets or sets the output tensor shape (N, C, H, W).
        /// </summary>
        public (int N, int C, int H, int W) OutputShape { get; set; }

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
        /// Gets or sets the dilation (height, width).
        /// </summary>
        public (int Height, int Width) Dilation { get; set; } = (1, 1);

        /// <summary>
        /// Gets or sets the activation function to apply.
        /// </summary>
        public NPUActivation Activation { get; set; } = NPUActivation.None;

        /// <summary>
        /// Gets or sets whether to use bias.
        /// </summary>
        public bool UseBias { get; set; } = true;

        /// <summary>
        /// Gets or sets the precision mode.
        /// </summary>
        public NPUPrecision Precision { get; set; } = NPUPrecision.FP32;

        /// <summary>
        /// Creates a default convolution configuration.
        /// </summary>
        public static NPUConvolutionConfig Default => new();
    }

    /// <summary>
    /// Configuration for NPU quantization operations.
    /// </summary>
    public sealed class NPUQuantizationConfig
    {
        /// <summary>
        /// Gets or sets the input quantization scale.
        /// </summary>
        public float InputScale { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the weight quantization scale.
        /// </summary>
        public float WeightScale { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the output dequantization scale.
        /// </summary>
        public float OutputScale { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the input zero point.
        /// </summary>
        public sbyte InputZeroPoint { get; set; } = 0;

        /// <summary>
        /// Gets or sets the weight zero point.
        /// </summary>
        public sbyte WeightZeroPoint { get; set; } = 0;

        /// <summary>
        /// Gets or sets the number of input elements.
        /// </summary>
        public long InputElements { get; set; }

        /// <summary>
        /// Gets or sets the number of output elements.
        /// </summary>
        public long OutputElements { get; set; }

        /// <summary>
        /// Gets or sets whether to use symmetric quantization.
        /// </summary>
        public bool UseSymmetricQuantization { get; set; } = true;

        /// <summary>
        /// Creates a default quantization configuration.
        /// </summary>
        public static NPUQuantizationConfig Default => new();
    }

    /// <summary>
    /// Configuration for NPU model execution.
    /// </summary>
    public sealed class NPUExecutionConfig
    {
        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; } = 1;

        /// <summary>
        /// Gets or sets the precision mode.
        /// </summary>
        public NPUPrecision Precision { get; set; } = NPUPrecision.FP32;

        /// <summary>
        /// Gets or sets the cache mode.
        /// </summary>
        public NPUCacheMode CacheMode { get; set; } = NPUCacheMode.Model;

        /// <summary>
        /// Gets or sets the input size.
        /// </summary>
        public long InputSize { get; set; }

        /// <summary>
        /// Gets or sets the output size.
        /// </summary>
        public long OutputSize { get; set; }

        /// <summary>
        /// Gets or sets the maximum execution time in milliseconds.
        /// </summary>
        public int TimeoutMs { get; set; } = 5000;

        /// <summary>
        /// Gets or sets whether to enable profiling.
        /// </summary>
        public bool EnableProfiling { get; set; } = false;

        /// <summary>
        /// Creates a default execution configuration.
        /// </summary>
        public static NPUExecutionConfig Default => new();
    }

    /// <summary>
    /// OpenVINO model wrapper for NPU execution.
    /// </summary>
    public sealed class NPUOpenVINOModel : IDisposable
    {
        private IntPtr _modelHandle;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the NPUOpenVINOModel class.
        /// </summary>
        /// <param name="modelPath">Path to the OpenVINO model file (.xml).</param>
        public NPUOpenVINOModel(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

            // In a real implementation, this would load the OpenVINO model
            // For now, create a dummy handle for compilation
            _modelHandle = new IntPtr(0x87654321);
            ModelPath = modelPath;
        }

        /// <summary>
        /// Gets the path to the OpenVINO model file.
        /// </summary>
        public string ModelPath { get; }

        /// <summary>
        /// Gets the native model handle.
        /// </summary>
        internal IntPtr ModelHandle => _modelHandle;

        /// <summary>
        /// Gets whether the model is disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Compiles the model for NPU execution.
        /// </summary>
        /// <param name="config">Compilation configuration.</param>
        public void CompileForNPU(NPUExecutionConfig config)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(NPUOpenVINOModel));

            // In a real implementation, this would compile the model for NPU
            // using OpenVINO's compile_model API with NPU device
        }

        /// <summary>
        /// Disposes the OpenVINO model.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // In a real implementation, this would release the OpenVINO model
                _modelHandle = IntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

}