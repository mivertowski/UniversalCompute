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

using ILGPU.Runtime;
using System;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators
{
    /// <summary>
    /// Base class for all universal accelerator tests.
    /// </summary>
    public abstract class TestBase : IDisposable
    {
        #region Instance

        private bool _disposed;

        /// <summary>
        /// Initializes a new test instance.
        /// </summary>
        protected TestBase()
        {
            Context = Context.CreateDefault();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the ILGPU context for testing.
        /// </summary>
        protected Context Context { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Creates an accelerator for testing if available.
        /// </summary>
        /// <typeparam name="T">The accelerator type.</typeparam>
        /// <returns>The accelerator instance or null if not available.</returns>
        protected T? CreateAcceleratorIfAvailable<T>() where T : Accelerator
        {
            try
            {
                return Context.CreateAccelerator<T>();
            }
            catch (NotSupportedException)
            {
                return null;
            }
            catch (PlatformNotSupportedException)
            {
                return null;
            }
        }

        /// <summary>
        /// Skips a test if the accelerator is not available.
        /// </summary>
        /// <param name="accelerator">The accelerator to check.</param>
        protected static void SkipIfNotAvailable(Accelerator? accelerator)
        {
            if (accelerator == null)
                throw new SkipException("Accelerator not available on this platform");
        }

        /// <summary>
        /// Verifies that two arrays are approximately equal.
        /// </summary>
        /// <param name="expected">Expected values.</param>
        /// <param name="actual">Actual values.</param>
        /// <param name="tolerance">Tolerance for floating-point comparison.</param>
        protected static void AssertEqual(float[] expected, float[] actual, float tolerance = 1e-5f)
        {
            Assert.Equal(expected.Length, actual.Length);
            
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(
                    Math.Abs(expected[i] - actual[i]) <= tolerance,
                    $"Arrays differ at index {i}: expected {expected[i]}, actual {actual[i]}");
            }
        }

        /// <summary>
        /// Verifies that two integer arrays are equal.
        /// </summary>
        /// <param name="expected">Expected values.</param>
        /// <param name="actual">Actual values.</param>
        protected static void AssertEqual(int[] expected, int[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }

        /// <summary>
        /// Creates test data for basic operations.
        /// </summary>
        /// <param name="size">Size of the test data.</param>
        /// <returns>Test data array.</returns>
        protected static float[] CreateTestData(int size)
        {
            var data = new float[size];
            var random = new Random(42); // Fixed seed for reproducible tests
            
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble() * 100.0f;
            }
            
            return data;
        }

        /// <summary>
        /// Creates sequential test data.
        /// </summary>
        /// <param name="size">Size of the test data.</param>
        /// <returns>Sequential test data array.</returns>
        protected static float[] CreateSequentialData(int size)
        {
            var data = new float[size];
            for (int i = 0; i < size; i++)
                data[i] = i;
            return data;
        }

        /// <summary>
        /// Measures execution time of an action.
        /// </summary>
        /// <param name="action">Action to measure.</param>
        /// <returns>Execution time in milliseconds.</returns>
        protected static double MeasureTime(Action action)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            action();
            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Disposes the test instance.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                Context?.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        #endregion
    }

    /// <summary>
    /// Exception thrown to skip tests when hardware is not available.
    /// </summary>
    public class SkipException : Exception
    {
        /// <summary>
        /// Initializes a new skip exception.
        /// </summary>
        /// <param name="reason">Reason for skipping the test.</param>
        public SkipException(string reason) : base(reason) { }
    }
}