﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2018-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: DefaultILBackend.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime.CPU;
using ILGPU.Util;
using System.Collections.Immutable;
#if !NATIVE_AOT && !AOT_COMPATIBLE
using System.Reflection.Emit;
#endif

namespace ILGPU.Backends.IL
{
#if !NATIVE_AOT && !AOT_COMPATIBLE
    /// <summary>
    /// The default IL backend that uses the original kernel method.
    /// </summary>
    /// <remarks>
    /// This backend is not available in AOT compilation modes as it requires
    /// dynamic IL generation through System.Reflection.Emit.
    /// </remarks>
    public class DefaultILBackend : ILBackend
    {
        #region Instance

        /// <summary>
        /// Constructs a new IL backend.
        /// </summary>
        /// <param name="context">The context to use.</param>
        protected internal DefaultILBackend(Context context)
            : base(context, new CPUCapabilityContext(), 1, new ILArgumentMapper(context))
        { }

        #endregion

        #region Methods

        /// <summary>
        /// Generates the actual kernel invocation call.
        /// </summary>
        protected override void GenerateCode<TEmitter>(
            EntryPoint entryPoint,
            in BackendContext backendContext,
            TEmitter emitter,
            in ILLocal task,
            in ILLocal index,
            ImmutableArray<ILLocal> locals)
        {
            // Load placeholder 'this' argument to satisfy IL evaluation stack
            if (entryPoint.MethodInfo.IsNotCapturingLambda())
                emitter.Emit(OpCodes.Ldnull);

            if (entryPoint.IsImplicitlyGrouped)
            {
                // Load index
                emitter.Emit(LocalOperation.Load, index);
            }

            // Load kernel arguments
            foreach (var local in locals)
                emitter.Emit(LocalOperation.Load, local);

            // Invoke kernel
            emitter.EmitCall(entryPoint.MethodInfo);
        }

        #endregion
    }
#endif
}
