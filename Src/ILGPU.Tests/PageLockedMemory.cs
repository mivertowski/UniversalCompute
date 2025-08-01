﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: PageLockedMemory.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class PageLockedMemory(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        private const int Length = 1024;

        public static TheoryData<long> Numbers => new()
        {
            { 10 },
            { -10 },
            { int.MaxValue },
            { int.MinValue },
        };

        internal static void PinnedMemoryKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[index] = data[index] == 0 ? 42 : 24;
        }

        [Fact]
        [KernelMethod(nameof(PinnedMemoryKernel))]
        public unsafe void PinnedUsingGCHandle()
        {
            var expected = Enumerable.Repeat(42, Length).ToArray();
            var array = new int[Length];
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            try
            {
                using var buffer = Accelerator.Allocate1D<int>(array.Length);
                using var scope = Accelerator.CreatePageLockFromPinned(array);

                buffer.View.CopyFrom(scope.ArrayView);
                Execute(buffer.Length, buffer.View);

                buffer.View.CopyTo(scope.ArrayView);
                Accelerator.Synchronize();
                Verify1D(array, expected);
            }
            finally
            {
                handle.Free();
            }
        }

        [Fact]
        [KernelMethod(nameof(PinnedMemoryKernel))]
        public void PinnedUsingGCAllocateArray()
        {
            var expected = Enumerable.Repeat(42, Length).ToArray();
            var array = System.GC.AllocateArray<int>(Length, pinned: true);
            using var buffer = Accelerator.Allocate1D<int>(array.Length);
            using var scope = Accelerator.CreatePageLockFromPinned(array);

            buffer.View.CopyFrom(scope.ArrayView);
            Execute(buffer.Length, buffer.View);

            buffer.View.CopyTo(scope.ArrayView);
            Accelerator.Synchronize();
            Verify1D(array, expected);
        }

        internal static void CopyKernel(
            Index1D index,
            ArrayView1D<long, Stride1D.Dense> data)
        {
            data[index] -= 5;
        }

        [Theory]
        [MemberData(nameof(Numbers))]
        [KernelMethod(nameof(CopyKernel))]
        public void Copy(long constant)
        {
            using var array = Accelerator.AllocatePageLocked1D<long>(Length);
            for (int i = 0; i < Length; i++)
            {
                array[i] = constant;
            }

            using var buff = Accelerator.Allocate1D<long>(Length);

            // Start copying, create the expected array in the meantime
            buff.View.CopyFrom(array.ArrayView);
            var expected = Enumerable.Repeat(constant - 5, Length).ToArray();
            Accelerator.Synchronize();

            Execute(array.Extent.ToIntIndex(), buff.View);
            Accelerator.Synchronize();

            buff.View.CopyTo(array.ArrayView);
            Accelerator.Synchronize();

            Assert.Equal(expected.Length, array.Length);
            for (int i = 0; i < Length; i++)
            {
                Assert.Equal(expected[i], array[i]);
            }
        }

        // No need for kernel, assuming copy tests pass.
        // Just going to confirm integrity in this test.
        [Fact]
        public void GetAsArrayPageLocked()
        {
            using var array = Accelerator.AllocatePageLocked1D<long>(Length);
            for (int i = 0; i < Length; i++)
            {
                array[i] = 10;
            }

            using var buff = Accelerator.Allocate1D<long>(Length);
            buff.View.CopyFrom(array.ArrayView);

            var expected = new int[Length];
            for (int i = 0; i < Length; i++)
            {
                expected[i] = 10;
            }

            Accelerator.Synchronize();

            var data = buff.View.GetAsPageLocked1D();
            Accelerator.Synchronize();

            Assert.Equal(expected.Length, data.Length);

            for (int i = 0; i < Length; i++)
            {
                Assert.Equal(expected[i], data[i]);
            }
        }
    }
}
