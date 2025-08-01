﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: KernelMethodAttribute.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Diagnostics;
using System.Reflection;

namespace ILGPU.Tests
{
    /// <summary>
    /// Links test methods to kernels.
    /// </summary>
    /// <remarks>
    /// Constructs a new kernel attribute.
    /// </remarks>
    /// <param name="methodName">The associated method name.</param>
    [AttributeUsage(AttributeTargets.Method)]
#pragma warning disable CA1515 // Consider making public types internal
    public sealed class KernelMethodAttribute(string methodName) : Attribute
#pragma warning restore CA1515 // Consider making public types internal
    {

        /// <summary>
        /// Constructs a new kernel attribute.
        /// </summary>
        /// <param name="methodName">The associated method name.</param>
        /// <param name="type">The source type.</param>
        public KernelMethodAttribute(string methodName, Type type)
            : this(methodName)
        {
            Type = type ?? throw new ArgumentNullException(nameof(type));
        }

        /// <summary>
        /// Returns the kernel name.
        /// </summary>
        public string MethodName { get; } = methodName
                ?? throw new ArgumentNullException(nameof(methodName));

        /// <summary>
        /// Returns the type in which the kernel method could be found (if any).
        /// </summary>
        public Type? Type { get; }

        /// <summary>
        /// Resolves the kernel method using the current configuration.
        /// </summary>
        /// <param name="typeArguments">The kernel type arguments.</param>
        /// <returns>The resolved kernel method.</returns>
        public static MethodInfo GetKernelMethod(
            Type[]? typeArguments = null,
            int offset = 1)
        {
            // TODO: create a nicer way ;)
            var stackTrace = new StackTrace();
            for (int i = offset; i < stackTrace.FrameCount; ++i)
            {
                var frame = stackTrace.GetFrame(i).ThrowIfNull();
                var callingMethod = frame.GetMethod().ThrowIfNull();
                var attribute = callingMethod.GetCustomAttribute<
                    KernelMethodAttribute>();
                if (attribute == null)
                {
                    continue;
                }

                var type = attribute.Type ?? callingMethod.DeclaringType.ThrowIfNull();
                return TestBase.GetKernelMethod(
                    type,
                    attribute.MethodName,
                    typeArguments);
            }
            throw new NotSupportedException(
                "Not supported kernel attribute. Missing attribute?");
        }
    }
}
