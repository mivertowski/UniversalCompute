// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2024-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: Vec128TypeGenerator.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.IL;
using ILGPU.Runtime.Velocity;
using System;
using System.Numerics;

#if NET7_0_OR_GREATER

namespace ILGPU.Backends.Velocity.Vec128
{
    /// <summary>
    /// A vector type generator of 128bit vectors to be used with the Velocity backend.
    /// </summary>
    /// <remarks>
    /// Constructs a new IL vector type generator.
    /// </remarks>
    /// <param name="capabilityContext">The parent capability context.</param>
    /// <param name="runtimeSystem">The parent runtime system.</param>
    sealed class Vec128TypeGenerator(
        VelocityCapabilityContext capabilityContext,
        RuntimeSystem runtimeSystem) : VelocityTypeGenerator(capabilityContext, runtimeSystem, Vector<int>.Count)
    {
        #region Static

        /// <summary>
        /// Maps basic types to vectorized basic types.
        /// </summary>
        private static readonly Type[] VectorizedBasicTypeMapping =
        [
            Vec128Operations.WarpType32, // None/Unknown

            Vec128Operations.WarpType32, // Int1
            Vec128Operations.WarpType32, // Int8
            Vec128Operations.WarpType32, // Int16
            Vec128Operations.WarpType32, // Int32
            Vec128Operations.WarpType64, // Int64

            Vec128Operations.WarpType32, // Float16
            Vec128Operations.WarpType32, // Float32
            Vec128Operations.WarpType64, // Float64
        ];

        #endregion
        #region Instance

        #endregion

        #region Type System

        public override Type GetVectorizedBasicType(BasicValueType basicValueType) => basicValueType == BasicValueType.Float16 && !CapabilityContext.Float16
                ? throw VelocityCapabilityContext.GetNotSupportedFloat16Exception()
                : VectorizedBasicTypeMapping[(int)basicValueType];

        #endregion
    }
}

#endif
