﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: PTXIntrinsics.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.AtomicOperations;
using ILGPU.IR.Intrinsics;
using ILGPU.IR.Values;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ILGPU.Backends.PTX
{
    /// <summary>
    /// Implements and initializes PTX intrinsics.
    /// </summary>
    static partial class PTXIntrinsics
    {
        #region Specializers

        /// <summary>
        /// The PTXIntrinsics type.
        /// </summary>
        private static readonly Type PTXIntrinsicsType = typeof(PTXIntrinsics);

        /// <summary>
        /// The Half implementation type.
        /// </summary>
        private static readonly Type HalfType = typeof(HalfExtensions);

        /// <summary>
        /// Creates a new PTX intrinsic.
        /// </summary>
        /// <param name="name">The name of the intrinsic.</param>
        /// <param name="mode">The implementation mode.</param>
        /// <param name="minArchitecture">The minimum architecture.</param>
        /// <param name="maxArchitecture">The maximum architecture.</param>
        /// <returns>The created intrinsic.</returns>
        private static PTXIntrinsic CreateIntrinsic(
            string name,
            IntrinsicImplementationMode mode,
            CudaArchitecture? minArchitecture,
            CudaArchitecture maxArchitecture) =>
            new(
                PTXIntrinsicsType,
                name,
                mode,
                minArchitecture,
                maxArchitecture);

        /// <summary>
        /// Creates a new PTX intrinsic.
        /// </summary>
        /// <param name="name">The name of the intrinsic.</param>
        /// <param name="mode">The implementation mode.</param>
        /// <returns>The created intrinsic.</returns>
        private static PTXIntrinsic CreateIntrinsic(
            string name,
            IntrinsicImplementationMode mode) =>
            new(PTXIntrinsicsType, name, mode);

        /// <summary>
        /// Creates a new FP16 intrinsic.
        /// </summary>
        /// <param name="name">The name of the intrinsic.</param>
        /// <param name="maxArchitecture">The maximum PTX architecture.</param>
        /// <returns>The created intrinsic.</returns>
        private static PTXIntrinsic CreateFP16Intrinsic(
            string name,
            CudaArchitecture? maxArchitecture) =>
            maxArchitecture.HasValue
            ? new PTXIntrinsic(
                HalfType,
                name,
                IntrinsicImplementationMode.Redirect,
                null,
                maxArchitecture.Value)
            : new PTXIntrinsic(HalfType, name, IntrinsicImplementationMode.Redirect);

        /// <summary>
        /// Creates a PTX intrinsic for the given math function.
        /// </summary>
        /// <param name="name">The intrinsic name.</param>
        /// <param name="types">The parameter types.</param>
        /// <returns>The resolved intrinsic representation.</returns>
        private static PTXIntrinsic CreateLibDeviceMathIntrinsic(
            string name,
            params Type[] types)
        {
            var targetMethod = typeof(LibDevice).GetMethod(
                name,
                BindingFlags.Public | BindingFlags.Static,
                null,
                types,
                null)
                .ThrowIfNull();
            return new PTXIntrinsic(
                targetMethod,
                IntrinsicImplementationMode.Redirect,
                libDeviceRequired: true);
        }

        /// <summary>
        /// Creates a PTX intrinsic for the given math function.
        /// </summary>
        /// <param name="baseType">The source type containing the intrinsic.</param>
        /// <param name="name">The intrinsic name.</param>
        /// <param name="types">The parameter types.</param>
        /// <returns>The resolved intrinsic representation.</returns>
        private static PTXIntrinsic CreateMathIntrinsic(
            [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)] Type baseType,
            string name,
            params Type[] types)
        {
            var targetMethod = baseType.GetMethod(
                name,
                BindingFlags.Public | BindingFlags.Static,
                null,
                types,
                null)
                .ThrowIfNull();
            return new PTXIntrinsic(targetMethod, IntrinsicImplementationMode.Redirect);
        }

        /// <summary>
        /// Registers all PTX intrinsics with the given manager.
        /// </summary>
        /// <param name="manager">The target implementation manager.</param>
        public static void Register(IntrinsicImplementationManager manager)
        {
            RegisterAtomics(manager);
            RegisterBroadcasts(manager);
            RegisterWarpShuffles(manager);
            RegisterFP16(manager);
            RegisterBitFunctions(manager);
            RegisterMathFunctions(manager);
        }

        #endregion

        #region Atomics

        /// <summary>
        /// Registers all atomic intrinsics with the given manager.
        /// </summary>
        /// <param name="manager">The target implementation manager.</param>
        private static void RegisterAtomics(IntrinsicImplementationManager manager)
        {
            manager.RegisterGenericAtomic(
                AtomicKind.Min,
                BasicValueType.Float32,
                CreateIntrinsic(
                    nameof(AtomicMinF32),
                    IntrinsicImplementationMode.Redirect));
            manager.RegisterGenericAtomic(
                AtomicKind.Max,
                BasicValueType.Float32,
                CreateIntrinsic(
                    nameof(AtomicMaxF32),
                    IntrinsicImplementationMode.Redirect));

            manager.RegisterGenericAtomic(
                AtomicKind.Min,
                BasicValueType.Float64,
                CreateIntrinsic(
                    nameof(AtomicMinF64),
                    IntrinsicImplementationMode.Redirect));
            manager.RegisterGenericAtomic(
                AtomicKind.Max,
                BasicValueType.Float64,
                CreateIntrinsic(
                    nameof(AtomicMaxF64),
                    IntrinsicImplementationMode.Redirect));

            manager.RegisterGenericAtomic(
                AtomicKind.Add,
                BasicValueType.Float64,
                CreateIntrinsic(
                    nameof(AtomicAddF64),
                    IntrinsicImplementationMode.Redirect,
                    null,
                    CudaArchitecture.SM_60));
        }

        /// <summary>
        /// Represents an atomic min operation in software.
        /// </summary>
        private readonly struct MinFloat : IAtomicOperation<float>
        {
            public float Operation(float current, float value) =>
                IntrinsicMath.Min(current, value);
        }

        /// <summary>
        /// A software implementation for atomic max on 32-bit floats.
        /// </summary>
        /// <param name="target">The target address.</param>
        /// <param name="value">The value to add.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float AtomicMinF32(ref float target, float value) =>
            Atomic.MakeAtomic(
                ref target,
                value,
                new MinFloat(),
                new CompareExchangeFloat());

        /// <summary>
        /// Represents an atomic max operation in software.
        /// </summary>
        private readonly struct MaxFloat : IAtomicOperation<float>
        {
            public float Operation(float current, float value) =>
                IntrinsicMath.Max(current, value);
        }

        /// <summary>
        /// A software implementation for atomic max on 32-bit floats.
        /// </summary>
        /// <param name="target">The target address.</param>
        /// <param name="value">The value to add.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float AtomicMaxF32(ref float target, float value) =>
            Atomic.MakeAtomic(
                ref target,
                value,
                new MaxFloat(),
                new CompareExchangeFloat());

        /// <summary>
        /// Represents an atomic min operation in software.
        /// </summary>
        private readonly struct MinDouble : IAtomicOperation<double>
        {
            public double Operation(double current, double value) =>
                IntrinsicMath.Min(current, value);
        }

        /// <summary>
        /// A software implementation for atomic max on 64-bit floats.
        /// </summary>
        /// <param name="target">The target address.</param>
        /// <param name="value">The value to add.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double AtomicMinF64(ref double target, double value) =>
            Atomic.MakeAtomic(
                ref target,
                value,
                new MinDouble(),
                new CompareExchangeDouble());

        /// <summary>
        /// Represents an atomic max operation in software.
        /// </summary>
        private readonly struct MaxDouble : IAtomicOperation<double>
        {
            public double Operation(double current, double value) =>
                IntrinsicMath.Max(current, value);
        }

        /// <summary>
        /// A software implementation for atomic max on 64-bit floats.
        /// </summary>
        /// <param name="target">The target address.</param>
        /// <param name="value">The value to add.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double AtomicMaxF64(ref double target, double value) =>
            Atomic.MakeAtomic(
                ref target,
                value,
                new MaxDouble(),
                new CompareExchangeDouble());

        /// <summary>
        /// Represents an atomic compare-exchange operation of type double.
        /// </summary>
        private readonly struct AddDouble : IAtomicOperation<double>
        {
            public double Operation(double current, double value) => current + value;
        }

        /// <summary>
        /// A software implementation for atomic adds on 64-bit floats.
        /// </summary>
        /// <param name="target">The target address.</param>
        /// <param name="value">The value to add.</param>
        private static double AtomicAddF64(ref double target, double value) =>
            Atomic.MakeAtomic(
                ref target,
                value,
                new AddDouble(),
                new CompareExchangeDouble());

        #endregion

        #region Broadcasts

        /// <summary>
        /// Registers all broadcast intrinsics with the given manager.
        /// </summary>
        /// <param name="manager">The target implementation manager.</param>
        private static void RegisterBroadcasts(
            IntrinsicImplementationManager manager)
        {
            manager.RegisterBroadcast(
                BroadcastKind.GroupLevel,
                CreateIntrinsic(
                    nameof(GroupBroadcast),
                    IntrinsicImplementationMode.Redirect));
            manager.RegisterBroadcast(
                BroadcastKind.WarpLevel,
                CreateIntrinsic(
                    nameof(WarpBroadcast),
                    IntrinsicImplementationMode.Redirect));
        }

        /// <summary>
        /// Implements a single group-broadcast operation.
        /// </summary>
        /// <typeparam name="T">The type to broadcast.</typeparam>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T GroupBroadcast<T>(T value, int groupIndex)
            where T : unmanaged
        {
            ref var sharedMemory = ref SharedMemory.Allocate<T>();
            if (Group.LinearIndex == groupIndex)
                sharedMemory = value;
            Group.Barrier();

            return sharedMemory;
        }

        /// <summary>
        /// Wraps a single warp-broadcast operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T WarpBroadcast<T>(T value, int laneIndex)
            where T : unmanaged =>
            Warp.Shuffle(value, laneIndex);

        #endregion

        #region Bit Functions

        /// <summary>
        /// Registers all unary bit intrinsics with the given manager.
        /// </summary>
        /// <param name="manager">The target implementation manager.</param>
        private static void RegisterBitFunctions(
            IntrinsicImplementationManager manager)
        {
            manager.RegisterUnaryArithmetic(
                UnaryArithmeticKind.CTZ,
                BasicValueType.Int32,
                CreateIntrinsic(
                    nameof(TrailingZeroCountI32),
                    IntrinsicImplementationMode.Redirect));
            manager.RegisterUnaryArithmetic(
                UnaryArithmeticKind.CTZ,
                BasicValueType.Int64,
                CreateIntrinsic(
                    nameof(TrailingZeroCountI64),
                    IntrinsicImplementationMode.Redirect));
        }

        /// <summary>
        /// Wraps a CTZ operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TrailingZeroCountI32(int value) =>
            IntrinsicMath.BitOperations.TrailingZeroCount(value);

        /// <summary>
        /// Wraps a CTZ operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TrailingZeroCountI64(long value) =>
            IntrinsicMath.BitOperations.TrailingZeroCount(value);

        #endregion
    }
}
