﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2022-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: StaticAbstractInterfaceMembers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Linq;
using System.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class StaticAbstractInterfaceMembers(
#pragma warning restore CA1515 // Consider making public types internal
        ITestOutputHelper output,
        TestContext testContext) : TestBase(output, testContext)
    {

#if NET7_0_OR_GREATER

        private const int Length = 1024;

        public static T GeZeroIfBigger<T>(T value, T max) where T : INumber<T>
        {
            return value > max ? T.Zero : value;
        }

        #region Generic math

        internal static void GenericMathKernel<T>(
            Index1D index,
            ArrayView1D<T, Stride1D.Dense> input,
            ArrayView1D<T, Stride1D.Dense> output,
            T maxValue)
            where T : unmanaged, INumber<T>
        {
            output[index] = GeZeroIfBigger(input[index], maxValue);
        }

        private void TestGenericMathKernel<T>(T[] inputValues, T[] expected, T maxValue)
            where T : unmanaged, INumber<T>
        {
            using var input = Accelerator.Allocate1D<T>(inputValues);
            using var output = Accelerator.Allocate1D<T>(Length);

            using var start = Accelerator.DefaultStream.AddProfilingMarker();
            Accelerator.LaunchAutoGrouped<
                Index1D,
                ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>,
                T>(
                GenericMathKernel,
                Accelerator.DefaultStream,
                (int)input.Length,
                input.View,
                output.View,
                maxValue);

            Verify(output.View, expected);
        }

        [Fact]
        public void GenericMathIntTest()
        {
            const int MaxValue = 50;
            var input = Enumerable.Range(0, Length).ToArray();

            var expected = input
                .Select(x => GeZeroIfBigger(x, MaxValue))
                .ToArray();

            TestGenericMathKernel(input, expected, MaxValue);
        }

        [Fact]
        public void GenericMathDoubleTest()
        {
            const double MaxValue = 75.0;
            var input = Enumerable.Range(0, Length)
                .Select(x => (double)x)
                .ToArray();

            var expected = input
                .Select(x => GeZeroIfBigger(x, MaxValue))
                .ToArray();

            TestGenericMathKernel(input, expected, MaxValue);
        }

        #endregion

        internal static void IncrementingKernel<T>(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> input,
            ArrayView1D<int, Stride1D.Dense> output)
            where T : IStaticAbstract
        {
            output[index] = T.Inc(input[index]);
        }

        private void TestIncrementingKernel<T>(int[] inputValues, int[] expected)
            where T : IStaticAbstract
        {
            using var input = Accelerator.Allocate1D<int>(inputValues);
            using var output = Accelerator.Allocate1D<int>(Length);

            using var start = Accelerator.DefaultStream.AddProfilingMarker();
            Accelerator.LaunchAutoGrouped<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(
                IncrementingKernel<T>,
                Accelerator.DefaultStream,
                (int)input.Length,
                input.View,
                output.View);

            Verify(output.View, expected);
        }

        internal interface IStaticAbstract
        {
            static abstract int Inc(int x);
        }

        internal class Incrementer : IStaticAbstract
        {
            public static int Inc(int x)
            {
                return x + 1;
            }

        }

        [Fact]
        public void StaticInterfaceTest()
        {
            int[] input = [.. Enumerable.Range(0, Length)];

            int[] expected = [.. input.Select(Incrementer.Inc)];

            TestIncrementingKernel<Incrementer>(input, expected);
        }
#endif
    }
}
