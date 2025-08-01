﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ProfilingMarkers.cs
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
    public abstract class ProfilingMarkers(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        private const int Length = 1024;

        internal static void ProfilingMarkerKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[index] = index;
        }

        [Fact]
        [KernelMethod(nameof(ProfilingMarkerKernel))]
        public void MeasureProfilingMarker()
        {
            var expected = Enumerable.Range(0, Length).ToArray();
            using var buffer = Accelerator.Allocate1D<int>(Length);

            using var start = Accelerator.DefaultStream.AddProfilingMarker();
            Accelerator.LaunchAutoGrouped<Index1D, ArrayView1D<int, Stride1D.Dense>>(
                ProfilingMarkerKernel,
                Accelerator.DefaultStream,
                (int)buffer.Length,
                buffer.View);
            using var end = Accelerator.DefaultStream.AddProfilingMarker();

            Verify(buffer.View, expected);

            var elapsedTime = end - start;
            Assert.True(elapsedTime.Ticks >= 0);
        }

        [Fact]
        [KernelMethod(nameof(ProfilingMarkerKernel))]
        public void MeasureReverseProfilingMarker()
        {
            var expected = Enumerable.Range(0, Length).ToArray();
            using var buffer = Accelerator.Allocate1D<int>(Length);

            using var start = Accelerator.DefaultStream.AddProfilingMarker();
            Accelerator.LaunchAutoGrouped<Index1D, ArrayView1D<int, Stride1D.Dense>>(
                ProfilingMarkerKernel,
                Accelerator.DefaultStream,
               (int)buffer.Length,
                buffer.View);
            using var end = Accelerator.DefaultStream.AddProfilingMarker();

            Verify(buffer.View, expected);

            var elapsedTime = start - end;
            Assert.True(elapsedTime.Ticks <= 0);
        }
    }
}
