﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2022-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: GetKernelTests.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class GetKernelTests(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        [Fact]
        public void LoadAutoGroupedKernel()
        {
            var launcher =
                Accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<int>>(
                (index, view) =>
                {
                    view[index] = index;
                });

            var kernel = launcher.GetKernel();
            Assert.NotNull(kernel);
        }

        [Fact]
        public void LoadAutoGroupedStreamKernel()
        {
            var launcher =
                Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                (index, view) =>
                {
                    view[index] = index;
                });

            var kernel = launcher.GetKernel();
            Assert.NotNull(kernel);
        }

        [Fact]
        public void LoadKernel()
        {
            var launcher =
                Accelerator.LoadKernel<Index1D, ArrayView<int>>(
                (index, view) =>
                {
                    view[index] = index;
                });

            var kernel = launcher.GetKernel();
            Assert.NotNull(kernel);
        }

        [Fact]
        public void LoadStreamKernel()
        {
            var launcher =
                Accelerator.LoadStreamKernel<Index1D, ArrayView<int>>(
                (index, view) =>
                {
                    view[index] = index;
                });

            var kernel = launcher.GetKernel();
            Assert.NotNull(kernel);
        }
    }
}
