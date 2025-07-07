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

namespace ILGPU.Runtime
{
    /// <summary>
    /// Represents kernel launch parameters.
    /// </summary>
    public sealed class KernelParameters
    {
        /// <summary>
        /// Gets or sets the parameter values.
        /// </summary>
#pragma warning disable CA1819 // Properties should not return arrays
        public object[] Values { get; set; } = [];
#pragma warning restore CA1819 // Properties should not return arrays

        /// <summary>
        /// Gets the number of parameters.
        /// </summary>
        public int Count => Values.Length;

        /// <summary>
        /// Gets the parameter at the specified index.
        /// </summary>
        /// <param name="index">Parameter index.</param>
        /// <returns>Parameter value.</returns>
        public object this[int index]
        {
            get => Values[index];
            set => Values[index] = value;
        }

        /// <summary>
        /// Creates empty kernel parameters.
        /// </summary>
        public static KernelParameters Empty => new();

        /// <summary>
        /// Creates kernel parameters from an array of values.
        /// </summary>
        /// <param name="values">Parameter values.</param>
        /// <returns>Kernel parameters instance.</returns>
        public static KernelParameters Create(params object[] values) => new() { Values = values };
    }
}