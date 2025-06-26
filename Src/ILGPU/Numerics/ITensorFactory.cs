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

using System.Numerics;
using System.Threading.Tasks;

namespace ILGPU.Numerics
{
    /// <summary>
    /// Factory interface for creating tensors from various data sources.
    /// </summary>
    public interface ITensorFactory
    {
        /// <summary>
        /// Creates a tensor from input data.
        /// </summary>
        /// <typeparam name="T">The input data type.</typeparam>
        /// <param name="input">The input data.</param>
        /// <returns>A task that represents the asynchronous creation operation.</returns>
        Task<ITensor<float>> CreateFromInputAsync<T>(T input) where T : class;

        /// <summary>
        /// Creates output data from a tensor.
        /// </summary>
        /// <typeparam name="T">The output data type.</typeparam>
        /// <param name="tensor">The tensor containing the data.</param>
        /// <returns>A task that represents the asynchronous conversion operation.</returns>
        Task<T> CreateOutputFromTensorAsync<T>(ITensor<float> tensor) where T : class;

        /// <summary>
        /// Creates a tensor with the specified shape and data type.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <returns>A new tensor instance.</returns>
        ITensor<T> Create<T>(TensorShape shape, ComputeLocation location) where T : unmanaged, INumber<T>;

        /// <summary>
        /// Creates a tensor with the specified shape, data type, and accelerator.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <param name="accelerator">The accelerator for GPU/unified tensors.</param>
        /// <returns>A new tensor instance.</returns>
        ITensor<T> Create<T>(TensorShape shape, ComputeLocation location, Runtime.Accelerator accelerator) 
            where T : unmanaged, INumber<T>;

        /// <summary>
        /// Creates a tensor from array data.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <returns>A new tensor instance.</returns>
        ITensor<T> CreateFromArray<T>(T[] data, TensorShape shape, ComputeLocation location) 
            where T : unmanaged, INumber<T>;

        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <returns>A new tensor filled with zeros.</returns>
        ITensor<T> Zeros<T>(TensorShape shape, ComputeLocation location) where T : unmanaged, INumber<T>;

        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <returns>A new tensor filled with ones.</returns>
        ITensor<T> Ones<T>(TensorShape shape, ComputeLocation location) where T : unmanaged, INumber<T>;
    }
}