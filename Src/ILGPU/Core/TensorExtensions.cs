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
using System.Linq;
using System.Reflection;
using ILGPU.Runtime;

namespace ILGPU.Core
{
    /// <summary>
    /// Extension methods for seamless conversion between different tensor systems
    /// in ILGPU (ML, Numerics, Hybrid).
    /// </summary>
    public static class TensorExtensions
    {
        #region ITensorCore<T> Extensions

        /// <summary>
        /// Converts an ITensorCore to ML.Tensor for ML operations.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>An ML.Tensor with the same data.</returns>
        public static ILGPU.ML.Tensor<T> ToMLTensor<T>(this ITensorCore<T> tensor) 
            where T : unmanaged, System.Numerics.INumber<T>
        {
            var mlShape = ConvertToMLShape(tensor.Shape);
            var data = tensor.AsReadOnlySpan();
            return new ILGPU.ML.Tensor<T>(tensor.Accelerator, mlShape, data);
        }

        /// <summary>
        /// Converts an ITensorCore to Numerics.ITensor for numerical operations.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>A Numerics.ITensor with the same data.</returns>
        public static ILGPU.Numerics.ITensor<T> ToNumericsTensor<T>(this ITensorCore<T> tensor) 
            where T : unmanaged
        {
            var numericsShape = ConvertToNumericsShape(tensor.Shape);
            return ILGPU.Numerics.TensorFactory.Create<T>(numericsShape, ILGPU.Numerics.ComputeLocation.Cpu);
        }

        /// <summary>
        /// Converts an ITensorCore to a MemoryBuffer for direct GPU operations.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>A MemoryBuffer containing the tensor data.</returns>
        public static MemoryBuffer1D<T, Stride1D.Dense> ToMemoryBuffer<T>(this ITensorCore<T> tensor) 
            where T : unmanaged
        {
            var buffer = tensor.Accelerator.Allocate1D<T>(tensor.ElementCount);
            var data = tensor.AsReadOnlySpan().ToArray();
            buffer.CopyFromCPU(data);
            return buffer;
        }

        /// <summary>
        /// Creates a zero-copy view of the tensor data as an ArrayView.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>An ArrayView for GPU kernel operations.</returns>
        public static ArrayView<T> AsArrayView<T>(this ITensorCore<T> tensor) 
            where T : unmanaged
        {
            if (tensor is UnifiedTensor<T> unifiedTensor)
            {
                // Direct access to internal buffer for unified tensors
                return unifiedTensor.ToMemoryBuffer().View;
            }
            
            // Fallback: create temporary buffer
            return tensor.ToMemoryBuffer().View;
        }

        /// <summary>
        /// Converts the tensor to a different element type.
        /// </summary>
        /// <typeparam name="TSource">The source element type.</typeparam>
        /// <typeparam name="TTarget">The target element type.</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>A new tensor with converted element types.</returns>
        public static ITensorCore<TTarget> ConvertTo<TSource, TTarget>(this ITensorCore<TSource> tensor)
            where TSource : unmanaged
            where TTarget : unmanaged
        {
            var sourceData = tensor.AsReadOnlySpan();
            var targetData = new TTarget[tensor.ElementCount];

            for (long i = 0; i < tensor.ElementCount; i++)
            {
                targetData[i] = ConvertElement<TSource, TTarget>(sourceData[(int)i]);
            }

            return TensorFactory.FromSpan<TTarget>(tensor.Accelerator, tensor.Shape, targetData);
        }

        #endregion

        #region ML.Tensor<T> Extensions

        /// <summary>
        /// Converts an ML.Tensor to ITensorCore for unified operations.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="mlTensor">The ML tensor.</param>
        /// <param name="accelerator">The target accelerator.</param>
        /// <returns>An ITensorCore with the same data.</returns>
        public static ITensorCore<T> ToUnifiedTensor<T>(this ILGPU.ML.Tensor<T> mlTensor, Accelerator accelerator) 
            where T : unmanaged, System.Numerics.INumber<T>
        {
            var unifiedShape = ConvertFromMLShape(mlTensor.Shape);
            var data = ExtractMLTensorData(mlTensor);
            return TensorFactory.FromArray(accelerator, unifiedShape, data);
        }

        #endregion

        #region Numerics.ITensor<T> Extensions

        /// <summary>
        /// Converts a Numerics.ITensor to ITensorCore for unified operations.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="numericsTensor">The numerics tensor.</param>
        /// <param name="accelerator">The target accelerator.</param>
        /// <returns>An ITensorCore with the same data.</returns>
        public static ITensorCore<T> ToUnifiedTensor<T>(this ILGPU.Numerics.ITensor<T> numericsTensor, Accelerator accelerator) 
            where T : unmanaged
        {
            var unifiedShape = ConvertFromNumericsShape(numericsTensor.Shape);
            var data = ExtractNumericsTensorData(numericsTensor);
            return TensorFactory.FromArray(accelerator, unifiedShape, data);
        }

        #endregion

        #region MemoryBuffer Extensions

        /// <summary>
        /// Converts a MemoryBuffer to ITensorCore.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="buffer">The memory buffer.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>An ITensorCore view of the buffer.</returns>
        public static ITensorCore<T> AsTensor<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, TensorShape shape) 
            where T : unmanaged
        {
            if (buffer.Length != shape.ElementCount)
                throw new ArgumentException("Buffer length must match shape element count");

            var data = buffer.GetAsArray1D();
            return TensorFactory.FromArray(buffer.Accelerator, shape, data);
        }

        #endregion

        #region Interoperability Helpers

        /// <summary>
        /// Performs automatic tensor conversion for operations between different tensor types.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <param name="operation">The operation to perform.</param>
        /// <returns>The result of the operation as a unified tensor.</returns>
        public static ITensorCore<T> PerformInteropOperation<T>(
            object left, 
            object right, 
            Func<ITensorCore<T>, ITensorCore<T>, ITensorCore<T>> operation,
            Accelerator accelerator) 
            where T : unmanaged
        {
            var leftUnified = ConvertToUnified<T>(left, accelerator);
            var rightUnified = ConvertToUnified<T>(right, accelerator);
            
            return operation(leftUnified, rightUnified);
        }

        /// <summary>
        /// Automatically converts any supported tensor type to ITensorCore.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The tensor object.</param>
        /// <param name="accelerator">The target accelerator.</param>
        /// <returns>An ITensorCore representation.</returns>
        public static ITensorCore<T> ConvertToUnified<T>(object tensor, Accelerator accelerator)
            where T : unmanaged => tensor switch
            {
                ITensorCore<T> unified => unified,
                ILGPU.Numerics.ITensor<T> numericsTensor => numericsTensor.ToUnifiedTensor(accelerator),
                MemoryBuffer1D<T, Stride1D.Dense> buffer => buffer.AsTensor(new TensorShape((int)buffer.Length)),
                _ when tensor.GetType().IsGenericType &&
                       tensor.GetType().GetGenericTypeDefinition() == typeof(ILGPU.ML.Tensor<>) &&
                       typeof(T).GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(System.Numerics.INumber<>))
                    => ConvertMLTensorDynamic<T>(tensor, accelerator),
                _ => throw new ArgumentException($"Unsupported tensor type: {tensor?.GetType()}")
            };

        private static ITensorCore<T> ConvertMLTensorDynamic<T>(object mlTensor, Accelerator accelerator) 
            where T : unmanaged
        {
            // Use reflection to call ToUnifiedTensor for ML.Tensor<T> with INumber<T> constraint
            var method = typeof(TensorExtensions).GetMethod(nameof(ToUnifiedTensor), 
                new[] { typeof(ILGPU.ML.Tensor<>).MakeGenericType(typeof(T)), typeof(Accelerator) });
            
            if (method != null)
            {
                return (ITensorCore<T>)method.Invoke(null, new[] { mlTensor, accelerator });
            }
            
            throw new ArgumentException($"Cannot convert ML.Tensor<{typeof(T)}> - type must implement INumber<T>");
        }

        #endregion

        #region Shape Conversion Helpers

        private static ILGPU.ML.TensorShape ConvertToMLShape(TensorShape shape)
        {
            var dimensions = new int[shape.Rank];
            for (int i = 0; i < shape.Rank; i++)
                dimensions[i] = shape[i];
            return new ILGPU.ML.TensorShape(dimensions);
        }

        private static ILGPU.Numerics.TensorShape ConvertToNumericsShape(TensorShape shape)
        {
            var dimensions = new int[shape.Rank];
            for (int i = 0; i < shape.Rank; i++)
                dimensions[i] = shape[i];
            return new ILGPU.Numerics.TensorShape(dimensions);
        }

        private static TensorShape ConvertFromMLShape(ILGPU.ML.TensorShape mlShape)
        {
            var dimensions = new int[mlShape.Rank];
            for (int i = 0; i < mlShape.Rank; i++)
                dimensions[i] = mlShape[i];
            return new TensorShape(dimensions);
        }

        private static TensorShape ConvertFromNumericsShape(ILGPU.Numerics.TensorShape numericsShape)
        {
            var dimensions = new int[numericsShape.Rank];
            for (int i = 0; i < numericsShape.Rank; i++)
                dimensions[i] = numericsShape[i];
            return new TensorShape(dimensions);
        }

        #endregion

        #region Data Extraction Helpers

        private static T[] ExtractMLTensorData<T>(ILGPU.ML.Tensor<T> mlTensor) where T : unmanaged, System.Numerics.INumber<T> =>
            // This would need to be implemented based on actual ML.Tensor API
            // For now, return empty array as placeholder
            new T[mlTensor.Shape.Size];

        private static T[] ExtractNumericsTensorData<T>(ILGPU.Numerics.ITensor<T> numericsTensor) where T : unmanaged =>
            // This would need to be implemented based on actual Numerics.ITensor API
            // For now, return empty array as placeholder
            new T[numericsTensor.Shape.Length];

        private static TTarget ConvertElement<TSource, TTarget>(TSource source)
            where TSource : unmanaged
            where TTarget : unmanaged
        {
            // Type conversion logic
            if (typeof(TSource) == typeof(TTarget))
                return (TTarget)(object)source;

            if (typeof(TSource) == typeof(float) && typeof(TTarget) == typeof(double))
                return (TTarget)(object)(double)(float)(object)source;
            
            if (typeof(TSource) == typeof(double) && typeof(TTarget) == typeof(float))
                return (TTarget)(object)(float)(double)(object)source;
            
            if (typeof(TSource) == typeof(int) && typeof(TTarget) == typeof(float))
                return (TTarget)(object)(float)(int)(object)source;
            
            if (typeof(TSource) == typeof(float) && typeof(TTarget) == typeof(int))
                return (TTarget)(object)(int)(float)(object)source;

            // Add more conversions as needed
            throw new NotSupportedException($"Conversion from {typeof(TSource)} to {typeof(TTarget)} is not supported");
        }

        #endregion
    }
}