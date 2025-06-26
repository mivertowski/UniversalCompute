// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2020-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: XMathTests.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Tests;
using Xunit.Abstractions;

namespace ILGPU.Algorithms.Tests
{
    public abstract partial class XMathTests(ITestOutputHelper output, TestContext testContext) : AlgorithmsTestBase(output, testContext)
    {
        internal readonly struct XMathTuple<T>(T x, T y) where T : struct
        {
            public T X { get; } = x;
            public T Y { get; } = y;
        }
    }
}
