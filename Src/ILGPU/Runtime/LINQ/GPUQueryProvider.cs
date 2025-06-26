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
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace ILGPU.Runtime.LINQ
{
    /// <summary>
    /// Provides query execution services for GPU-based LINQ operations.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the <see cref="GPUQueryProvider"/> class.
    /// </remarks>
    /// <param name="accelerator">The accelerator.</param>
    internal sealed class GPUQueryProvider(Accelerator accelerator) : IQueryProvider
    {
        #region Instance

        private readonly Accelerator accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

        #endregion

        #region IQueryProvider

        /// <summary>
        /// Creates a new queryable from the given expression.
        /// </summary>
        /// <param name="expression">The expression tree.</param>
        /// <returns>A queryable representing the expression.</returns>
        public IQueryable CreateQuery(Expression expression)
        {
            if (expression == null)
                throw new ArgumentNullException(nameof(expression));

            var elementType = GetElementType(expression.Type);
            var queryableType = typeof(GPUQueryable<>).MakeGenericType(elementType);
            
            return (IQueryable)Activator.CreateInstance(
                queryableType,
                this,
                expression,
                accelerator,
                null)!;
        }

        /// <summary>
        /// Creates a new queryable from the given expression.
        /// </summary>
        /// <typeparam name="TElement">The element type.</typeparam>
        /// <param name="expression">The expression tree.</param>
        /// <returns>A queryable representing the expression.</returns>
        public IQueryable<TElement> CreateQuery<TElement>(Expression expression)
        {
            if (expression == null)
                throw new ArgumentNullException(nameof(expression));

            // For now, return a simple implementation - this is a limitation of the constraint system
            return (IQueryable<TElement>)CreateQuery(expression);
        }

        /// <summary>
        /// Executes the query represented by the given expression.
        /// </summary>
        /// <param name="expression">The expression tree.</param>
        /// <returns>The query result.</returns>
        public object Execute(Expression expression)
        {
            if (expression == null)
                throw new ArgumentNullException(nameof(expression));

            var executor = new GPUQueryExecutor(accelerator);
            return executor.Execute(expression);
        }

        /// <summary>
        /// Executes the query represented by the given expression.
        /// </summary>
        /// <typeparam name="TResult">The result type.</typeparam>
        /// <param name="expression">The expression tree.</param>
        /// <returns>The query result.</returns>
        public TResult Execute<TResult>(Expression expression)
        {
            if (expression == null)
                throw new ArgumentNullException(nameof(expression));

            var executor = new GPUQueryExecutor(accelerator);
            var result = executor.Execute(expression);
            return (TResult)result;
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Gets the element type from a queryable type.
        /// </summary>
        /// <param name="seqType">The sequence type.</param>
        /// <returns>The element type.</returns>
        private static Type GetElementType(Type seqType)
        {
            var ienum = FindIEnumerable(seqType);
            return ienum?.GetGenericArguments()[0] ?? seqType;
        }

        /// <summary>
        /// Finds the IEnumerable interface in the type hierarchy.
        /// </summary>
        /// <param name="seqType">The sequence type.</param>
        /// <returns>The IEnumerable interface type.</returns>
        private static Type? FindIEnumerable(Type seqType)
        {
            if (seqType == null || seqType == typeof(string))
                return null;

            if (seqType.IsArray)
                return typeof(IEnumerable<>).MakeGenericType(seqType.GetElementType()!);

            if (seqType.IsGenericType)
            {
                foreach (var arg in seqType.GetGenericArguments())
                {
                    var ienum = typeof(IEnumerable<>).MakeGenericType(arg);
                    if (ienum.IsAssignableFrom(seqType))
                        return ienum;
                }
            }

            var ifaces = seqType.GetInterfaces();
            if (ifaces.Length > 0)
            {
                foreach (var iface in ifaces)
                {
                    var ienum = FindIEnumerable(iface);
                    if (ienum != null)
                        return ienum;
                }
            }

            if (seqType.BaseType != null && seqType.BaseType != typeof(object))
                return FindIEnumerable(seqType.BaseType);

            return null;
        }

        #endregion
    }
}
