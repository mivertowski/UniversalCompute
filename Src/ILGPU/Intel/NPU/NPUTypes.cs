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
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Represents tensor shape for Intel NPU operations.
    /// </summary>
    public readonly struct TensorShape(params int[] dimensions)
    {
        private readonly int[] _dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));

        public int this[int index] => _dimensions[index];
        public int Rank => _dimensions.Length;
        public int[] Dimensions => (int[])_dimensions.Clone();
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
        public NeuralOperation[] Operations { get; } = operations ?? [];
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



}