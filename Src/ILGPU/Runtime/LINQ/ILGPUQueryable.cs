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
using System.Linq;
using System.Linq.Expressions;

namespace ILGPU.Runtime.LINQ
{
    /// <summary>
    /// Provides LINQ-style operations for GPU arrays with lazy evaluation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public interface IGPUQueryable<T> : IQueryable<T>, IDisposable
        where T : unmanaged
    {
        /// <summary>
        /// Gets the accelerator associated with this queryable.
        /// </summary>
        Accelerator Accelerator { get; }

        /// <summary>
        /// Gets the underlying memory buffer.
        /// </summary>
        MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }

        /// <summary>
        /// Executes the query and returns the results as an enumerable.
        /// </summary>
        /// <returns>The query results.</returns>
        IEnumerable<T> Execute();

        /// <summary>
        /// Executes the query and returns the results as an array.
        /// </summary>
        /// <returns>The query results as an array.</returns>
        T[] ToArray();

        /// <summary>
        /// Executes the query and stores the results in the specified buffer.
        /// </summary>
        /// <param name="outputBuffer">The output buffer.</param>
        void ExecuteTo(MemoryBuffer1D<T, Stride1D.Dense> outputBuffer);
    }

    /// <summary>
    /// Represents a GPU queryable with lazy evaluation of operations.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class GPUQueryable<T> : IGPUQueryable<T>
        where T : unmanaged
    {
        #region Instance

        private bool disposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPUQueryable{T}"/> class.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="buffer">The memory buffer.</param>
        internal GPUQueryable(
            Accelerator accelerator,
            MemoryBuffer1D<T, Stride1D.Dense> buffer)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
            Provider = new GPUQueryProvider(accelerator);
            Expression = Expression.Constant(this);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GPUQueryable{T}"/> class.
        /// </summary>
        /// <param name="provider">The query provider.</param>
        /// <param name="expression">The expression tree.</param>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="buffer">The memory buffer.</param>
        internal GPUQueryable(
            IQueryProvider provider,
            Expression expression,
            Accelerator accelerator,
            MemoryBuffer1D<T, Stride1D.Dense> buffer)
        {
            Provider = provider ?? throw new ArgumentNullException(nameof(provider));
            Expression = expression ?? throw new ArgumentNullException(nameof(expression));
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the accelerator associated with this queryable.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Gets the underlying memory buffer.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }

        /// <summary>
        /// Gets the element type.
        /// </summary>
        public Type ElementType => typeof(T);

        /// <summary>
        /// Gets the expression tree.
        /// </summary>
        public Expression Expression { get; }

        /// <summary>
        /// Gets the query provider.
        /// </summary>
        public IQueryProvider Provider { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Returns an enumerator that iterates through the query results.
        /// </summary>
        /// <returns>An enumerator for the query results.</returns>
        public IEnumerator<T> GetEnumerator() => Execute().GetEnumerator();

        /// <summary>
        /// Returns an enumerator that iterates through the query results.
        /// </summary>
        /// <returns>An enumerator for the query results.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Executes the query and returns the results as an enumerable.
        /// </summary>
        /// <returns>The query results.</returns>
        public IEnumerable<T> Execute()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(GPUQueryable<T>));

            // Execute the expression tree and return results
            var executor = new GPUQueryExecutor(Accelerator);
            return executor.Execute<T>(Expression);
        }

        /// <summary>
        /// Executes the query and returns the results as an array.
        /// </summary>
        /// <returns>The query results as an array.</returns>
        public T[] ToArray() => [.. Execute()];

        /// <summary>
        /// Executes the query and stores the results in the specified buffer.
        /// </summary>
        /// <param name="outputBuffer">The output buffer.</param>
        public void ExecuteTo(MemoryBuffer1D<T, Stride1D.Dense> outputBuffer)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(GPUQueryable<T>));
            if (outputBuffer == null)
                throw new ArgumentNullException(nameof(outputBuffer));

            var executor = new GPUQueryExecutor(Accelerator);
            executor.ExecuteTo(Expression, outputBuffer);
        }

        /// <summary>
        /// Releases all resources used by the <see cref="GPUQueryable{T}"/>.
        /// </summary>
        public void Dispose()
        {
            if (!disposed)
            {
                Buffer?.Dispose();
                disposed = true;
            }
        }

        #endregion
    }
}
