﻿// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Backends;
using ILGPU.Backends.OpenCL;
using ILGPU.IR;
using ILGPU.IR.Intrinsics;
using ILGPU.IR.Values;
using ILGPU.Util;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.Algorithms.CL
{
    /// <summary>
    /// Manages custom CL-specific intrinsics.
    /// </summary>
    static partial class CLContext
    {
        /// <summary>
        /// The <see cref="CLGroupExtensions"/> type.
        /// </summary>
        internal static readonly Type CLGroupExtensionsType = typeof(CLGroupExtensions);

        /// <summary>
        /// The <see cref="CLWarpExtensions"/> type.
        /// </summary>
        internal static readonly Type CLWarpExtensionsType = typeof(CLWarpExtensions);

        /// <summary>
        /// Registers an intrinsic mapping.
        /// </summary>
        /// <param name="manager">The current manager.</param>
        /// <param name="sourceType">The source type.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="name">The method name to register.</param>
        private static void RegisterIntrinsicMapping(
            IntrinsicImplementationManager manager,
            Type sourceType,
            Type targetType,
            string name)
        {
            var sourceMethod = sourceType.GetMethod(
                name,
                AlgorithmContext.IntrinsicBindingFlags)
                .ThrowIfNull();
            manager.RegisterMethod(
                sourceMethod,
                new CLIntrinsic(targetType, name, IntrinsicImplementationMode.Redirect));
        }

        /// <summary>
        /// Registers an intrinsic mapping using a code generator.
        /// </summary>
        /// <param name="manager">The current manager.</param>
        /// <param name="sourceType">The source type.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="name">The method name to register.</param>
        /// <param name="codeGeneratorName">
        /// The name of the code generator to register.
        /// </param>
        private static void RegisterIntrinsicCodeGenerator(
            IntrinsicImplementationManager manager,
            Type sourceType,
            Type targetType,
            string name,
            string codeGeneratorName)
        {
            var sourceMethod = sourceType.GetMethod(
                name,
                AlgorithmContext.IntrinsicBindingFlags)
                .ThrowIfNull();
            manager.RegisterMethod(
                sourceMethod,
                new CLIntrinsic(
                    targetType,
                    codeGeneratorName,
                    IntrinsicImplementationMode.GenerateCode));
        }

        /// <summary>
        /// Registers an XMath replacement mapping using a code generator.
        /// </summary>
        /// <param name="manager">The current manager.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="functionName">The method name to register.</param>
        /// <param name="codeGeneratorName">
        /// The name of the code generator to register.
        /// </param>
        /// <param name="types">The argument types for the target method.</param>
        private static void RegisterXMathCodeGenerator(
            IntrinsicImplementationManager manager,
            Type targetType,
            string functionName,
            string codeGeneratorName,
            params Type[] types)
        {
            manager.RegisterMethod(
                AlgorithmContext.XMathType.GetMethod(
                    functionName,
                    AlgorithmContext.IntrinsicBindingFlags,
                    null,
                    types,
                    null)
                    .ThrowIfNull(),
                new CLIntrinsic(
                    targetType,
                    codeGeneratorName,
                    IntrinsicImplementationMode.GenerateCode));
        }

        /// <summary>
        /// Generates an intrinsic reduce.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TScanReduce">The reduction logic.</typeparam>
        /// <param name="backend">The current backend.</param>
        /// <param name="codeGenerator">The code generator.</param>
        /// <param name="value">The value to generate code for.</param>
        /// <param name="scanReduceOperation">The basic reduction operation.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void GenerateScanReduce<T, TScanReduce>(
            CLBackend backend,
            CLCodeGenerator codeGenerator,
            Value value,
            string scanReduceOperation)
            where T : struct
            where TScanReduce : struct, IScanReduceOperation<T>
        {
            // Allocate target and load source argument
            var reduce = value.AsNotNullCast<MethodCall>();
            var sourceValue = codeGenerator.Load(reduce[0]);
            var target = codeGenerator.Allocate(value);

            // Resolve OpenCL command
            TScanReduce scanReduce = default;
            var clCommand = scanReduce.CLCommand;
            if (string.IsNullOrWhiteSpace(clCommand))
                throw new InvalidCodeGenerationException();

            using var statement = codeGenerator.BeginStatement(target);
            statement.AppendCommand(scanReduceOperation + clCommand);
            statement.BeginArguments();
            statement.AppendCast(typeof(T).GetArithmeticBasicValueType());
            statement.Append(sourceValue);
            statement.EndArguments();
        }
    }
}
