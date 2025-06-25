// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2024 ILGPU Project
//                                    www.ilgpu.net
//
// File: TestContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Velocity;
using System;

#if NET7_0_OR_GREATER

namespace ILGPU.Tests.Velocity128
{
    /// <summary>
    /// An abstract test context for Velocity accelerators.
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
    public abstract class Velocity128TestContext(
        OptimizationLevel optimizationLevel,
        bool enableAssertions,
        bool forceDebugConfig,
        Action<Context.Builder> prepareContext) : TestContext(
              optimizationLevel,
              enableAssertions,
              forceDebugConfig,
              builder => prepareContext(
                      builder.Velocity(VelocityDeviceType.Vector128)),
              context => context.CreateVelocityAccelerator())
    {
    }
}

#endif
