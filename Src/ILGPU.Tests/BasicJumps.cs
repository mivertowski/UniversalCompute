﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: BasicJumps.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

#pragma warning disable CS0162
#pragma warning disable CS0164

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class BasicJumps(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        [SuppressMessage(
            "Style",
            "IDE0059:Unnecessary assignment of a value",
            Justification = "Testing unconditional jump")]
        internal static void BasicJumpKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> source)
        {
            var value = source[index];
            goto exit;

            data[index] = value;
            return;

        exit:
            data[index] = 23;
        }

        [Theory]
        [InlineData(32)]
        [InlineData(1024)]
        [KernelMethod(nameof(BasicJumpKernel))]
        public void BasicJump(int length)
        {
            using var buffer = Accelerator.Allocate1D<int>(length);
            using var source = Accelerator.Allocate1D<int>(length);
            var sourceData = Enumerable.Repeat(42, (int)buffer.Length).ToArray();
            source.CopyFromCPU(Accelerator.DefaultStream, sourceData);

            Execute(buffer.Length, buffer.View, source.View);

            var expected = Enumerable.Repeat(23, (int)buffer.Length).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void BasicIfJumpKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> source)
        {
            var value = source[index];
            if (value < 23)
            {
                goto exit;
            }

            data[index] = value;
            return;

        exit:
            data[index] = 23;
        }

        [Theory]
        [InlineData(32)]
        [InlineData(1024)]
        [KernelMethod(nameof(BasicIfJumpKernel))]
        public void BasicIfJump(int length)
        {
            using var buffer = Accelerator.Allocate1D<int>(length);
            using var source = Accelerator.Allocate1D<int>(length);
            var partLength = (int)source.Length / 3;
            var sourceData = Enumerable.Repeat(13, partLength).Concat(
                Enumerable.Repeat(42, (int)source.Length - partLength)).ToArray();
            source.CopyFromCPU(Accelerator.DefaultStream, sourceData);

            Execute(buffer.Length, buffer.View, source.View);

            var expected = Enumerable.Repeat(23, partLength).Concat(
                Enumerable.Repeat(42, length - partLength)).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void BasicLoopJumpKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> source)
        {
            for (int i = 0; i < source.Length; ++i)
            {
                if (source[i] == 23)
                {
                    goto exit;
                }
            }

            data[index] = 42;
            return;

        exit:
            data[index] = 23;
        }

        [Theory]
        [InlineData(32)]
        [InlineData(1024)]
        [KernelMethod(nameof(BasicLoopJumpKernel))]
        public void BasicLoopJump(int length)
        {
            using var buffer = Accelerator.Allocate1D<int>(length);
            using var source = Accelerator.Allocate1D<int>(64);
            var sourceData = Enumerable.Range(0, (int)source.Length).ToArray();
            sourceData[57] = 23;
            source.CopyFromCPU(Accelerator.DefaultStream, sourceData);

            Execute(buffer.Length, buffer.View, source.View);

            var expected = Enumerable.Repeat(23, length).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void BasicNestedLoopJumpKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> source,
            int c)
        {
            int k = 0;
        entry:
            for (int i = 0; i < source.Length; ++i)
            {
                if (source[i] == 23)
                {
                    if (k < c)
                    {
                        goto exit;
                    }

                    goto nested;
                }
            }

            data[index] = 42;
            return;

        nested:
            k = 43;

        exit:
            if (k++ < 1)
            {
                goto entry;
            }

            data[index] = 23 + k;
        }

        [Theory]
        [InlineData(0, 67)]
        [InlineData(2, 25)]
        [KernelMethod(nameof(BasicNestedLoopJumpKernel))]
        public void BasicNestedLoopJump(int c, int res)
        {
            const int Length = 64;
            using var buffer = Accelerator.Allocate1D<int>(Length);
            using var source = Accelerator.Allocate1D<int>(Length);
            var sourceData = Enumerable.Range(0, (int)source.Length).ToArray();
            sourceData[57] = 23;
            source.CopyFromCPU(Accelerator.DefaultStream, sourceData);

            Execute(buffer.Length, buffer.View, source.View, c);

            var expected = Enumerable.Repeat(res, Length).ToArray();
            Verify(buffer.View, expected);
        }

        private static void BasicNestedLoopJumpKernel2(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> target,
            ArrayView1D<int, Stride1D.Dense> source)
        {
            int k = 0;
        entry:
            for (int i = 0; i < source.Length; ++i)
            {
                goto exit;
            }

            target[index] = 42;
            return;

        nested:
            k = 43;

        exit:
            if (k++ < 1)
            {
                goto entry;
            }

            target[index] = 23 + k;
        }

        [Fact]
        [KernelMethod(nameof(BasicNestedLoopJumpKernel2))]
        public void BasicNestedLoopJump2()
        {
            const int Length = 32;
            using var buffer = Accelerator.Allocate1D<int>(Length);
            using var source = Accelerator.Allocate1D<int>(Length);

            Execute(buffer.Length, buffer.View, source.View);

            var expected = Enumerable.Repeat(25, Length).ToArray();
            Verify(buffer.View, expected);
        }
    }
}

#pragma warning restore CS0164
#pragma warning restore CS0162
