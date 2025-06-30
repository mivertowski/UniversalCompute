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

namespace ILGPU.Numerics.AI
{
    /// <summary>
    /// Represents a neural network that can be executed on various accelerators.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the NeuralNetwork class.
    /// </remarks>
    /// <param name="modelPath">The path to the model file.</param>
    public sealed class NeuralNetwork(string modelPath, IReadOnlyList<NeuralOperation> operations) : IDisposable
    {
        private bool _disposed;

        /// <summary>
        /// Gets the path to the model file.
        /// </summary>
        public string ModelPath { get; } = modelPath ?? throw new ArgumentNullException(nameof(modelPath));

        /// <summary>
        /// Gets the operations in the neural network.
        /// </summary>
        public IReadOnlyList<NeuralOperation> Operations { get; } = operations ?? Array.Empty<NeuralOperation>();

        /// <summary>
        /// Gets the input shape expected by the network.
        /// </summary>
        public TensorShape InputShape { get; internal set; } = new TensorShape(1, 3, 224, 224);

        /// <summary>
        /// Gets the output shape produced by the network.
        /// </summary>
        public TensorShape OutputShape { get; internal set; } = new TensorShape(1, 1000);

        /// <summary>
        /// Gets whether the network has been loaded and is ready for inference.
        /// </summary>
        public bool IsLoaded { get; internal set; }

        /// <summary>
        /// Gets the output shape for a given input shape.
        /// </summary>
        /// <param name="inputShape">The input tensor shape.</param>
        /// <returns>The expected output shape.</returns>
        public TensorShape GetOutputShape(TensorShape inputShape) =>
            // For now, return a default output shape
            // In a real implementation, this would compute the actual output shape
            OutputShape;

        /// <summary>
        /// Loads the neural network model.
        /// </summary>
        public void Load() =>
            // Stub implementation - would load the actual model
            IsLoaded = true;

        /// <summary>
        /// Disposes the neural network.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // Clean up model resources
                IsLoaded = false;
                _disposed = true;
            }
        }
    }
}