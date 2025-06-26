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

using ILGPU;
using System;

namespace ILGPU.Benchmarks
{
    /// <summary>
    /// Provides a shared ILGPU context for benchmarks to avoid resolver conflicts.
    /// </summary>
    public static class SharedBenchmarkContext
    {
        private static Context? _sharedContext;
        private static readonly object _lock = new object();

        /// <summary>
        /// Gets or creates a shared ILGPU context for benchmarks.
        /// </summary>
        public static Context GetOrCreateContext()
        {
            if (_sharedContext != null && !_sharedContext.IsDisposed)
                return _sharedContext;

            lock (_lock)
            {
                if (_sharedContext != null && !_sharedContext.IsDisposed)
                    return _sharedContext;

                try
                {
                    _sharedContext = Context.CreateDefault();
                    return _sharedContext;
                }
                catch (InvalidOperationException)
                {
                    // Resolver already set - try to get the existing context
                    // This is a fallback for when multiple benchmarks run
                    throw new InvalidOperationException("ILGPU context resolver conflict. Run benchmarks individually or restart process.");
                }
            }
        }

        /// <summary>
        /// Disposes the shared context. Should only be called at the end of all benchmarks.
        /// </summary>
        public static void DisposeSharedContext()
        {
            lock (_lock)
            {
                _sharedContext?.Dispose();
                _sharedContext = null;
            }
        }
    }
}