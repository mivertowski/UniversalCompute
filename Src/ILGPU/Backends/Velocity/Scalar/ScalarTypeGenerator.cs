// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2023-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ScalarTypeGenerator.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.IL;
using ILGPU.Runtime.Velocity;
using System;

namespace ILGPU.Backends.Velocity.Scalar
{
    /// <summary>
    /// A scalar type generator to be used with the Velocity backend.
    /// </summary>
    /// <remarks>
    /// Constructs a new IL scalar code generator.
    /// </remarks>
    /// <param name="capabilityContext">The parent capability context.</param>
    /// <param name="runtimeSystem">The parent runtime system.</param>
    sealed class ScalarTypeGenerator(
        VelocityCapabilityContext capabilityContext,
        RuntimeSystem runtimeSystem) : VelocityTypeGenerator(capabilityContext, runtimeSystem, ScalarOperations2.WarpSize)
    {
        #region Static

        /// <summary>
        /// Maps basic types to vectorized basic types.
        /// </summary>
        private static readonly Type[] VectorizedBasicTypeMapping =
        [
            ScalarOperations2.WarpType32, // None/Unknown

            ScalarOperations2.WarpType32, // Int1
            ScalarOperations2.WarpType32, // Int8
            ScalarOperations2.WarpType32, // Int16
            ScalarOperations2.WarpType32, // Int32
            ScalarOperations2.WarpType64, // Int64

            ScalarOperations2.WarpType32, // Float16
            ScalarOperations2.WarpType32, // Float32
            ScalarOperations2.WarpType64, // Float64
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
