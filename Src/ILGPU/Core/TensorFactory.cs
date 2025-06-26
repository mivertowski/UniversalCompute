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
using System.Numerics;
using ILGPU.Runtime;

namespace ILGPU.Core
{
    /// <summary>
    /// Factory class for creating unified tensors across all ILGPU tensor systems.
    /// This provides a single entry point for tensor creation that can be specialized
    /// for ML, Numerics, or Hybrid use cases.
    /// </summary>
    public static class TensorFactory
    {
        /// <summary>
        /// Creates a new tensor with the specified shape on the given accelerator.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>A new tensor.</returns>
        public static ITensorCore<T> Create<T>(Accelerator accelerator, TensorShape shape)
            where T : unmanaged => new UnifiedTensor<T>(accelerator, shape);

        /// <summary>
        /// Creates a new tensor with the specified shape on the CPU.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>A new CPU tensor.</returns>
        public static ITensorCore<T> CreateCPU<T>(Context context, TensorShape shape) 
            where T : unmanaged
        {
            // Find CPU device and create accelerator
            foreach (var device in context)
            {
                if (device is ILGPU.Runtime.CPU.CPUDevice cpuDevice)
                {
                    var cpuAccelerator = cpuDevice.CreateAccelerator(context);
                    return new UnifiedTensor<T>(cpuAccelerator, shape);
                }
            }
            
            throw new InvalidOperationException("No CPU device found in context");
        }

        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>A new tensor filled with zeros.</returns>
        public static ITensorCore<T> Zeros<T>(Accelerator accelerator, TensorShape shape) 
            where T : unmanaged
        {
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            tensor.Fill(default(T));
            return tensor;
        }

        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>A new tensor filled with ones.</returns>
        public static ITensorCore<T> Ones<T>(Accelerator accelerator, TensorShape shape) 
            where T : unmanaged, INumber<T>
        {
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            tensor.Fill(T.One);
            return tensor;
        }

        /// <summary>
        /// Creates a tensor from CPU data.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="data">The source data.</param>
        /// <returns>A new tensor with the specified data.</returns>
        public static ITensorCore<T> FromArray<T>(Accelerator accelerator, TensorShape shape, T[] data) 
            where T : unmanaged
        {
            if (data.Length != shape.ElementCount)
                throw new ArgumentException($"Data length {data.Length} doesn't match shape element count {shape.ElementCount}");

            var tensor = new UnifiedTensor<T>(accelerator, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates a tensor from a ReadOnlySpan.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="data">The source data span.</param>
        /// <returns>A new tensor with the specified data.</returns>
        public static ITensorCore<T> FromSpan<T>(Accelerator accelerator, TensorShape shape, ReadOnlySpan<T> data) 
            where T : unmanaged
        {
            if (data.Length != shape.ElementCount)
                throw new ArgumentException($"Data length {data.Length} doesn't match shape element count {shape.ElementCount}");

            var tensor = new UnifiedTensor<T>(accelerator, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates a random tensor with values between 0 and 1.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="seed">Optional random seed.</param>
        /// <returns>A new tensor with random values.</returns>
        public static ITensorCore<T> Random<T>(Accelerator accelerator, TensorShape shape, int? seed = null) 
            where T : unmanaged, INumber<T>
        {
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var data = new T[shape.ElementCount];
            
            for (long i = 0; i < shape.ElementCount; i++)
            {
                // Generate random value between 0 and 1
                var randomValue = random.NextDouble();
                data[i] = T.CreateChecked(randomValue);
            }
            
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates a tensor with random normal distribution (mean=0, std=1).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="seed">Optional random seed.</param>
        /// <returns>A new tensor with normally distributed random values.</returns>
        public static ITensorCore<T> RandomNormal<T>(Accelerator accelerator, TensorShape shape, int? seed = null) 
            where T : unmanaged, INumber<T>
        {
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var data = new T[shape.ElementCount];
            
            // Use Box-Muller transform for normal distribution
            for (long i = 0; i < shape.ElementCount; i += 2)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                
                var z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                var z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                
                data[i] = T.CreateChecked(z0);
                if (i + 1 < shape.ElementCount)
                    data[i + 1] = T.CreateChecked(z1);
            }
            
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates an identity matrix tensor.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="size">The size of the identity matrix (size x size).</param>
        /// <returns>A new identity matrix tensor.</returns>
        public static ITensorCore<T> Identity<T>(Accelerator accelerator, int size) 
            where T : unmanaged, INumber<T>
        {
            var shape = new TensorShape(size, size);
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            
            var data = new T[size * size];
            for (int i = 0; i < size; i++)
            {
                data[i * size + i] = T.One;  // Set diagonal elements to 1
            }
            
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates a tensor with values arranged in a sequence (0, 1, 2, ...).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <returns>A new tensor with sequential values.</returns>
        public static ITensorCore<T> Arange<T>(Accelerator accelerator, TensorShape shape) 
            where T : unmanaged, INumber<T>
        {
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            
            var data = new T[shape.ElementCount];
            for (long i = 0; i < shape.ElementCount; i++)
            {
                data[i] = T.CreateChecked(i);
            }
            
            tensor.CopyFrom(data);
            return tensor;
        }

        /// <summary>
        /// Creates a tensor with values arranged in a sequence with custom start, stop, and step.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="start">The start value.</param>
        /// <param name="stop">The stop value (exclusive).</param>
        /// <param name="step">The step size.</param>
        /// <returns>A new tensor with sequential values.</returns>
        public static ITensorCore<T> Arange<T>(Accelerator accelerator, T start, T stop, T step) 
            where T : unmanaged, INumber<T>
        {
            // Calculate the number of elements
            var range = stop - start;
            var elementCount = (long)Math.Ceiling(double.CreateChecked(range) / double.CreateChecked(step));
            
            var shape = new TensorShape((int)elementCount);
            var tensor = new UnifiedTensor<T>(accelerator, shape);
            
            var data = new T[elementCount];
            var current = start;
            for (long i = 0; i < elementCount; i++)
            {
                data[i] = current;
                current += step;
            }
            
            tensor.CopyFrom(data);
            return tensor;
        }
    }
}