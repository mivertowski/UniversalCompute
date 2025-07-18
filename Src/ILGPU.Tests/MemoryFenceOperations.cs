﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: MemoryFenceOperations.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class MemoryFenceOperations(
#pragma warning restore CA1515 // Consider making public types internal
        ITestOutputHelper output,
        TestContext testContext) : TestBase(output, testContext)
    {
        private const int Length = 1024;

        internal static void MemoryFenceGroupLevelKernel(
            ArrayView1D<int, Stride1D.Dense> data)
        {
            var idx = Grid.GlobalIndex.X;
            data[idx] = idx;

            MemoryFence.GroupLevel();
        }

        [Fact]
        [KernelMethod(nameof(MemoryFenceGroupLevelKernel))]
        public void MemoryFenceGroupLevel()
        {
            for (int i = 1; i < Accelerator.MaxNumThreadsPerGroup; i <<= 1)
            {
                var extent = new KernelConfig(Length, i);
                using var buffer = Accelerator.Allocate1D<int>(extent.Size);
                Execute(extent, buffer.View);

                var expected = Enumerable.Range(0, (int)extent.Size).ToArray();
                Verify(buffer.View, expected);
            }
        }

        internal static void MemoryFenceDeviceLevelKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[index] = index;

            MemoryFence.DeviceLevel();
        }

        [Fact]
        [KernelMethod(nameof(MemoryFenceDeviceLevelKernel))]
        public void MemoryFenceDeviceLevel()
        {
            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);

            var expected = Enumerable.Range(0, Length).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void MemoryFenceSystemLevelKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[index] = index;

            MemoryFence.SystemLevel();
        }

        [Fact]
        [KernelMethod(nameof(MemoryFenceSystemLevelKernel))]
        public void MemoryFenceSystemLevel()
        {
            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);

            var expected = Enumerable.Range(0, Length).ToArray();
            Verify(buffer.View, expected);
        }
    }
}
