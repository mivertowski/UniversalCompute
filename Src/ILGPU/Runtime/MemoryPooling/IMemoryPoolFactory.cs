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
// Change License: Apache License, Version 2.0using System;

namespace ILGPU.Runtime.MemoryPooling
{
    /// <summary>
    /// Factory interface for creating memory pools.
    /// </summary>
    public interface IMemoryPoolFactory
    {
        /// <summary>
        /// Creates a memory pool for the specified accelerator and element type.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to create the pool for.</param>
        /// <param name="configuration">The pool configuration.</param>
        /// <returns>A new memory pool instance.</returns>
        IMemoryPool<T> CreatePool<T>(Accelerator accelerator, MemoryPoolConfiguration? configuration = null) 
            where T : unmanaged;
    }

    /// <summary>
    /// Default implementation of the memory pool factory.
    /// </summary>
    public sealed class DefaultMemoryPoolFactory : IMemoryPoolFactory
    {
        /// <inheritdoc/>
        public IMemoryPool<T> CreatePool<T>(Accelerator accelerator, MemoryPoolConfiguration? configuration = null) 
            where T : unmanaged
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            return new AdaptiveMemoryPool<T>(accelerator, configuration);
        }
    }

    /// <summary>
    /// Predefined memory pool configuration presets.
    /// </summary>
    public enum MemoryPoolPreset
    {
        /// <summary>
        /// Default configuration suitable for most applications.
        /// </summary>
        Default,

        /// <summary>
        /// High-performance configuration with larger pools and more retention.
        /// </summary>
        HighPerformance,

        /// <summary>
        /// Memory-efficient configuration with smaller pools and aggressive trimming.
        /// </summary>
        MemoryEfficient,

        /// <summary>
        /// Development configuration with detailed statistics and frequent trimming.
        /// </summary>
        Development
    }
}
