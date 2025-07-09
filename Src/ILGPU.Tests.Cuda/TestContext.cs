// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: TestContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.Tests.Cuda
{
    /// <summary>
    /// An abstract test context for Cuda accelerators.
    /// </summary>
    /// <remarks>
    /// Creates a new test context instance.
    /// </remarks>
    /// <param name="optimizationLevel">The optimization level to use.</param>
    /// <param name="enableAssertions">
    /// Enables use of assertions.
    /// </param>
    /// <param name="forceDebugConfig">
    /// Forces use of debug configuration in O1 and O2 builds.
    /// </param>
    /// <param name="prepareContext">The context preparation handler.</param>
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class CudaTestContext(
#pragma warning restore CA1515 // Consider making public types internal
        OptimizationLevel optimizationLevel,
        bool enableAssertions,
        bool forceDebugConfig,
        Action<Context.Builder> prepareContext) : TestContext(
              optimizationLevel,
              enableAssertions,
              forceDebugConfig,
              builder => prepareContext(builder.Cuda()),
              context => context.CreateCudaAccelerator(0))
    {
    }
}
