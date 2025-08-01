﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: TestContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.CPU;
using System;

namespace ILGPU.Tests.CPU
{
    /// <summary>
    /// An abstract test context for CPU accelerators.
    /// </summary>
    /// <remarks>
    /// The CPU simulation mode can be change via the environment variable
    /// <see cref="CPUKindEnvVariable"/>. The content should be the raw
    /// string representation of one of the enumeration members of
    /// <see cref="CPUDeviceKind"/>
    /// </remarks>
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
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1515:Consider making types internal", Justification = "Base class must be accessible to generated configurations")]
    public abstract class CPUTestContext(
        OptimizationLevel optimizationLevel,
        bool enableAssertions,
        bool forceDebugConfig,
        Action<Context.Builder> prepareContext) : TestContext(
              optimizationLevel,
              enableAssertions,
              forceDebugConfig,
              builder => prepareContext(builder.CPU(GetCPUDeviceKind())),
              context => context.CreateCPUAccelerator(0))
    {
        /// <summary>
        /// The name of the environment variable to control the kind of all CPU tests.
        /// </summary>
        public const string CPUKindEnvVariable = "ILGPU_CPU_TEST_KIND";

        /// <summary>
        /// Gets the <see cref="CPUDeviceKind"/> based on the environment variable
        /// <see cref="CPUKindEnvVariable"/>.
        /// </summary>
        /// <remarks>
        /// If the environment variables does not exists or does not contain a valid kind
        /// (specified by the <see cref="CPUDeviceKind"/> enumeration), this
        /// function creates a simulator compatible with the kind
        /// <see cref="CPUDeviceKind.Default"/>.
        /// </remarks>
        private static CPUDeviceKind GetCPUDeviceKind()
        {
            var cpuConfig = Environment.GetEnvironmentVariable(CPUKindEnvVariable);
            return Enum.TryParse(cpuConfig, out CPUDeviceKind kind)
                ? kind
                : CPUDeviceKind.Default;
        }
    }
}
