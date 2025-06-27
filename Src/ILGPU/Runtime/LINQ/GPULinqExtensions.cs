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
using System.Linq;

namespace ILGPU.Runtime.LINQ
{
    /// <summary>
    /// Extension methods for LINQ-style GPU operations.
    /// </summary>
    public static class GPULinqExtensions
    {
        /// <summary>
        /// Creates a GPU queryable from a memory buffer.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="buffer">The memory buffer.</param>
        /// <returns>A GPU queryable.</returns>
        public static IGPUQueryable<T> AsGPUQueryable<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer)
            where T : unmanaged
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            return new GPUQueryable<T>(buffer.Accelerator, buffer);
        }

        /// <summary>
        /// Creates a GPU queryable from an array by copying to GPU memory.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="data">The source data.</param>
        /// <returns>A GPU queryable.</returns>
        public static IGPUQueryable<T> AsGPUQueryable<T>(this Accelerator accelerator, T[] data)
            where T : unmanaged
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var buffer = accelerator.Allocate1D(data);
            return new GPUQueryable<T>(accelerator, buffer);
        }

        /// <summary>
        /// Creates a GPU queryable from a span by copying to GPU memory.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="data">The source data.</param>
        /// <returns>A GPU queryable.</returns>
        public static IGPUQueryable<T> AsGPUQueryable<T>(this Accelerator accelerator, ReadOnlySpan<T> data)
            where T : unmanaged
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            var buffer = accelerator.Allocate1D<T>(data.Length);
            buffer.View.BaseView.CopyFromCPU(data);
            return new GPUQueryable<T>(accelerator, buffer);
        }

        /// <summary>
        /// Projects each element of a GPU sequence into a new form using a GPU kernel.
        /// </summary>
        /// <typeparam name="TSource">The source element type.</typeparam>
        /// <typeparam name="TResult">The result element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="selector">The transformation function.</param>
        /// <returns>A GPU queryable with transformed elements.</returns>
        public static IGPUQueryable<TResult> Select<TSource, TResult>(
            this IGPUQueryable<TSource> source,
            Func<TSource, TResult> selector)
            where TSource : unmanaged
            where TResult : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (selector == null)
                throw new ArgumentNullException(nameof(selector));

            var parameter = System.Linq.Expressions.Expression.Parameter(typeof(TSource), "x");
            return source.Provider.CreateQuery<TResult>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Select),
                    [typeof(TSource), typeof(TResult)],
                    source.Expression,
                    System.Linq.Expressions.Expression.Quote(
                        System.Linq.Expressions.Expression.Lambda(
                            System.Linq.Expressions.Expression.Invoke(
                                System.Linq.Expressions.Expression.Constant(selector),
                                parameter),
                            parameter)))) as IGPUQueryable<TResult> ?? throw new InvalidOperationException("Failed to create GPU queryable");
        }

        /// <summary>
        /// Filters a GPU sequence based on a predicate using a GPU kernel.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="predicate">The filter predicate.</param>
        /// <returns>A GPU queryable with filtered elements.</returns>
        public static IGPUQueryable<T> Where<T>(
            this IGPUQueryable<T> source,
            Func<T, bool> predicate)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (predicate == null)
                throw new ArgumentNullException(nameof(predicate));

            var parameter = System.Linq.Expressions.Expression.Parameter(typeof(T), "x");
            return source.Provider.CreateQuery<T>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Where),
                    [typeof(T)],
                    source.Expression,
                    System.Linq.Expressions.Expression.Quote(
                        System.Linq.Expressions.Expression.Lambda(
                            System.Linq.Expressions.Expression.Invoke(
                                System.Linq.Expressions.Expression.Constant(predicate),
                                parameter),
                            parameter)))) as IGPUQueryable<T> ?? throw new InvalidOperationException("Failed to create GPU queryable");
        }

        /// <summary>
        /// Computes the sum of a GPU sequence using a GPU reduction kernel.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <returns>The sum of all elements.</returns>
        public static T Sum<T>(this IGPUQueryable<T> source)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            return source.Provider.Execute<T>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Sum),
                    [typeof(T)],
                    source.Expression));
        }

        /// <summary>
        /// Computes the average of a GPU sequence using a GPU reduction kernel.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <returns>The average of all elements.</returns>
        public static double Average<T>(this IGPUQueryable<T> source)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            return source.Provider.Execute<double>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Average),
                    [typeof(T)],
                    source.Expression));
        }

        /// <summary>
        /// Finds the minimum value in a GPU sequence using a GPU reduction kernel.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <returns>The minimum value.</returns>
        public static T Min<T>(this IGPUQueryable<T> source)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            return source.Provider.Execute<T>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Min),
                    [typeof(T)],
                    source.Expression));
        }

        /// <summary>
        /// Finds the maximum value in a GPU sequence using a GPU reduction kernel.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <returns>The maximum value.</returns>
        public static T Max<T>(this IGPUQueryable<T> source)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            return source.Provider.Execute<T>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Max),
                    [typeof(T)],
                    source.Expression));
        }

        /// <summary>
        /// Counts the number of elements in a GPU sequence.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <returns>The number of elements.</returns>
        public static int Count<T>(this IGPUQueryable<T> source)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            return source.Provider.Execute<int>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Count),
                    [typeof(T)],
                    source.Expression));
        }

        /// <summary>
        /// Determines whether any element in a GPU sequence satisfies a condition.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="predicate">The condition predicate.</param>
        /// <returns>True if any element satisfies the condition.</returns>
        public static bool Any<T>(this IGPUQueryable<T> source, Func<T, bool> predicate)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (predicate == null)
                throw new ArgumentNullException(nameof(predicate));

            return source.Provider.Execute<bool>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.Any),
                    [typeof(T)],
                    source.Expression,
                    System.Linq.Expressions.Expression.Quote(
                        System.Linq.Expressions.Expression.Lambda(
                            System.Linq.Expressions.Expression.Invoke(
                                System.Linq.Expressions.Expression.Constant(predicate),
                                System.Linq.Expressions.Expression.Parameter(typeof(T), "x")),
                            System.Linq.Expressions.Expression.Parameter(typeof(T), "x")))));
        }

        /// <summary>
        /// Determines whether all elements in a GPU sequence satisfy a condition.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="predicate">The condition predicate.</param>
        /// <returns>True if all elements satisfy the condition.</returns>
        public static bool All<T>(this IGPUQueryable<T> source, Func<T, bool> predicate)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (predicate == null)
                throw new ArgumentNullException(nameof(predicate));

            return source.Provider.Execute<bool>(
                System.Linq.Expressions.Expression.Call(
                    typeof(Queryable),
                    nameof(Queryable.All),
                    [typeof(T)],
                    source.Expression,
                    System.Linq.Expressions.Expression.Quote(
                        System.Linq.Expressions.Expression.Lambda(
                            System.Linq.Expressions.Expression.Invoke(
                                System.Linq.Expressions.Expression.Constant(predicate),
                                System.Linq.Expressions.Expression.Parameter(typeof(T), "x")),
                            System.Linq.Expressions.Expression.Parameter(typeof(T), "x")))));
        }

        /// <summary>
        /// Performs a parallel transformation on each element of a GPU sequence.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="action">The transformation action.</param>
        /// <returns>A GPU queryable with transformed elements.</returns>
        public static IGPUQueryable<T> ForEach<T>(this IGPUQueryable<T> source, Action<T> action)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            // Execute the action on each element using a GPU kernel
            var kernel = source.Accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<T>>(
                (index, view) => action(view[index]));

            using var stream = source.Accelerator.CreateStream();
            kernel(stream, (Index1D)source.Buffer.Length, source.Buffer.View);
            stream.Synchronize();

            return source;
        }

        /// <summary>
        /// Applies an accumulator function over a GPU sequence.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source queryable.</param>
        /// <param name="func">The accumulator function.</param>
        /// <returns>The final accumulator value.</returns>
        public static T Aggregate<T>(this IGPUQueryable<T> source, Func<T, T, T> func)
            where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (func == null)
                throw new ArgumentNullException(nameof(func));

            // Use GPU reduction for aggregation
            var data = source.ToArray();
            if (data.Length == 0)
                throw new InvalidOperationException("Sequence contains no elements");

            var result = data[0];
            for (int i = 1; i < data.Length; i++)
            {
                result = func(result, data[i]);
            }

            return result;
        }
    }
}
