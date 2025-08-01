﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: WarpOperations.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class WarpOperations(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        internal static void WarpDimensionKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> length,
            ArrayView1D<int, Stride1D.Dense> idx)
        {
            length[index] = Warp.WarpSize;
            idx[index] = Warp.LaneIdx;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [KernelMethod(nameof(WarpDimensionKernel))]
        public void WarpDimension(int warpMultiplier)
        {
            var length = Accelerator.WarpSize * warpMultiplier;
            using var lengthBuffer = Accelerator.Allocate1D<int>(length);
            using var idxBuffer = Accelerator.Allocate1D<int>(length);
            Execute(length, lengthBuffer.View, idxBuffer.View);

            var expectedLength = Enumerable.Repeat(
                Accelerator.WarpSize, length).ToArray();
            Verify(lengthBuffer.View, expectedLength);

            var expectedIndices = new int[length];
            for (int i = 0; i < length; ++i)
            {
                expectedIndices[i] = i % Accelerator.WarpSize;
            }

            Verify(idxBuffer.View, expectedIndices);
        }

        internal static void WarpBarrierKernel(ArrayView1D<int, Stride1D.Dense> data)
        {
            var idx = Grid.GlobalIndex.X;
            Warp.Barrier();
            data[idx] = idx;
        }

        [Theory]
        [InlineData(32)]
        [InlineData(256)]
        [InlineData(1024)]
        [KernelMethod(nameof(WarpBarrierKernel))]
        public void WarpBarrier(int length)
        {
            var warpSize = Accelerator.WarpSize;
            using var buffer = Accelerator.Allocate1D<int>(length * warpSize);
            var extent = new KernelConfig(
                length,
                warpSize);
            Execute(extent, buffer.View);

            var expected = Enumerable.Range(0, length * warpSize).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void WarpBroadcastKernel(
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> data2,
            int c)
        {
            var idx = Grid.GlobalIndex.X;
            data[idx] = Warp.Broadcast(Group.IdxX, Warp.WarpSize - 1);
            data2[idx] = Warp.Broadcast(c, Warp.WarpSize - 2);
        }

        [Theory]
        [InlineData(32)]
        [InlineData(256)]
        [InlineData(1024)]
        [KernelMethod(nameof(WarpBroadcastKernel))]
        public void WarpBroadcast(int length)
        {
            var warpSize = Accelerator.WarpSize;
            using var buffer = Accelerator.Allocate1D<int>(length * warpSize);
            using var buffer2 = Accelerator.Allocate1D<int>(length * warpSize);
            var extent = new KernelConfig(
                length,
                warpSize);
            Execute(extent, buffer.View, buffer2.View, length);

            var expected = Enumerable.Repeat(warpSize - 1, length * warpSize).ToArray();
            Verify(buffer.View, expected);

            var expected2 = Enumerable.Repeat(length, length * warpSize).ToArray();
            Verify(buffer2.View, expected2);
        }

        internal static void WarpShuffleKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            ArrayView1D<int, Stride1D.Dense> data2,
            int c)
        {
            var targetIdx = Warp.WarpSize - 1;
            data[index] = Warp.Shuffle(Warp.LaneIdx, targetIdx);
            data2[index] = Warp.Shuffle(c, targetIdx);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [KernelMethod(nameof(WarpShuffleKernel))]
        public void WarpShuffle(int warpMultiplier)
        {
            var length = Accelerator.WarpSize * warpMultiplier;
            using var dataBuffer = Accelerator.Allocate1D<int>(length);
            using var dataBuffer2 = Accelerator.Allocate1D<int>(length);
            Execute(length, dataBuffer.View, dataBuffer2.View, length);

            var expected = Enumerable.Repeat(
                Accelerator.WarpSize - 1, length).ToArray();
            Verify(dataBuffer.View, expected);

            var expected2 = Enumerable.Repeat(length, length).ToArray();
            Verify(dataBuffer2.View, expected2);
        }

        internal static void WarpShuffleDownKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            int shiftAmount)
        {
            data[index] = Warp.ShuffleDown(Warp.LaneIdx, shiftAmount);
        }

        [Fact]
        [KernelMethod(nameof(WarpShuffleDownKernel))]
        public void WarpShuffleDown()
        {
            int warpMultiplier = 1;
            for (
                int shiftAmount = 0;
                shiftAmount < Math.Min(4, Accelerator.WarpSize);
                ++shiftAmount)
            {
                var length = Accelerator.WarpSize * warpMultiplier;
                using var dataBuffer = Accelerator.Allocate1D<int>(length);
                Execute(length, dataBuffer.View, shiftAmount);

                var expected = new int[Accelerator.WarpSize - shiftAmount];
                for (int i = 0; i < warpMultiplier; ++i)
                {
                    var baseIdx = i * Accelerator.WarpSize;
                    for (int j = 0; j < Accelerator.WarpSize - shiftAmount; ++j)
                    {
                        expected[baseIdx + j] = j + shiftAmount;
                    }

                    // Do no test the remaining values as they are undefined
                    // for (
                    //     int j = Accelerator.WarpSize - shiftAmount;
                    //     j < Accelerator.WarpSize;
                    //     ++j)
                    //     expected[baseIdx + j] = j;
                }

                Verify(dataBuffer.View, expected, 0, Accelerator.WarpSize - shiftAmount);
            }
        }

        internal static void WarpShuffleUpKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data,
            int shiftAmount)
        {
            data[index] = Warp.ShuffleUp(Warp.LaneIdx, shiftAmount);
        }

        [Theory]
        [InlineData(1)]
        [KernelMethod(nameof(WarpShuffleUpKernel))]
        public void WarpShuffleUp(int warpMultiplier)
        {
            for (
                int shiftAmount = 0;
                shiftAmount < Math.Min(4, Accelerator.WarpSize);
                ++shiftAmount)
            {
                var length = Accelerator.WarpSize * warpMultiplier;
                using var dataBuffer = Accelerator.Allocate1D<int>(length);
                Execute(length, dataBuffer.View, shiftAmount);

                var expected = new int[length];
                for (int i = 0; i < warpMultiplier; ++i)
                {
                    var baseIdx = i * Accelerator.WarpSize;
                    for (int j = shiftAmount; j < Accelerator.WarpSize; ++j)
                    {
                        expected[baseIdx + j] = j - shiftAmount;
                    }

                    // Do no test the remaining values as they are undefined
                    // for (int j = 0; j < shiftAmount; ++j)
                    //     expected[baseIdx + j] = j;
                }

                Verify(dataBuffer.View, expected, shiftAmount);
            }
        }

        internal static void WarpShuffleXorKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
        {
            var value = Warp.LaneIdx;
            for (int laneMask = Warp.WarpSize / 2; laneMask > 0; laneMask >>= 1)
            {
                var shuffled = Warp.ShuffleXor(value, laneMask);
                value += shuffled;
            }
            data[index] = value;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [KernelMethod(nameof(WarpShuffleXorKernel))]
        public void WarpShuffleXor(int warpMultiplier)
        {
            var length = Accelerator.WarpSize * warpMultiplier;
            using var dataBuffer = Accelerator.Allocate1D<int>(length);
            Execute(length, dataBuffer.View);

            var expected = Enumerable.Repeat(
                Accelerator.WarpSize * (Accelerator.WarpSize - 1) / 2,
                length).ToArray();

            Verify(dataBuffer.View, expected);
        }

        internal static void DivergentWarpBarrierKernel(
            ArrayView1D<int, Stride1D.Dense> data)
        {
            // Divergent warp execution involving barriers
            var sharedMemory = ILGPU.SharedMemory.Allocate<int>(4);
            switch (Warp.WarpIdx)
            {
                case 0:
                    sharedMemory[0] = 0;
                    Warp.Barrier();
                    break;
                case 1:
                    sharedMemory[1] = 1;
                    Warp.Barrier();
                    break;
                case 2:
                    sharedMemory[2] = 2;
                    Warp.Barrier();
                    break;
                case 3:
                    sharedMemory[3] = 3;
                    Warp.Barrier();
                    break;
            }

            // Warp wide barrier
            int value = Warp.LaneIdx;
            Warp.Barrier();

            // Read shared memory
            int index = Grid.GlobalIndex.X;
            data[index] = sharedMemory[Warp.WarpIdx] * Warp.WarpSize + value;
        }

        [SkippableTheory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [KernelMethod(nameof(DivergentWarpBarrierKernel))]
        public void DivergentWarpBarrier(int warpMultiplier)
        {
            int warpSize = Accelerator.WarpSize;
            var length = warpSize * warpMultiplier;
            Skip.If(length > Accelerator.MaxNumThreadsPerGroup);

            using var dataBuffer = Accelerator.Allocate1D<int>(length);
            Execute(new KernelConfig(1, length), dataBuffer.View);

            var expected = Enumerable.Range(0, length).ToArray();
            Verify(dataBuffer.View, expected);
        }
    }
}
