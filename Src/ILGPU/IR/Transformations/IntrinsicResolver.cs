// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: IntrinsicResolver.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Intrinsics;
using ILGPU.IR.Values;
using System;

namespace ILGPU.IR.Transformations
{
    /// <summary>
    /// Resolved required intrinsic IR implementations.
    /// </summary>
    /// <typeparam name="TDelegate">The backend-specific delegate type.</typeparam>
    /// <remarks>
    /// Constructs a new intrinsic resolver.
    /// </remarks>
    /// <param name="implementationProvider">
    /// The implementation provider to use.
    /// </param>
    public sealed class IntrinsicResolver<TDelegate>(
        IntrinsicImplementationProvider<TDelegate> implementationProvider) :
        UnorderedTransformation<
            IntrinsicImplementationProvider<TDelegate>.IRSpecializationPhase>
        where TDelegate : Delegate
    {
        #region Instance

        private readonly IntrinsicImplementationProvider<TDelegate> provider = implementationProvider
                ?? throw new ArgumentNullException(nameof(implementationProvider));

        #endregion

        #region Methods

        /// <summary>
        /// Begins a new intrinsic specialization phase.
        /// </summary>
        protected override
            IntrinsicImplementationProvider<TDelegate>.IRSpecializationPhase
            CreateIntermediate(in MethodCollection methods) =>
            provider.BeginIRSpecialization();

        /// <summary>
        /// Finishes an intrinsic specialization phase.
        /// </summary>
        /// <param name="intermediate"></param>
        protected override void FinishProcessing(
            IntrinsicImplementationProvider<TDelegate>.
                IRSpecializationPhase intermediate) =>
            intermediate.Dispose();

        /// <summary>
        /// Applies an intrinsic implementation transformation.
        /// </summary>
        protected override bool PerformTransformation(
            Method.Builder builder,
            IntrinsicImplementationProvider<TDelegate>.
                IRSpecializationPhase intermediate)
        {
            // Check whether we are currently processing an intrinsic method
            var blocks = builder.SourceBlocks;

            bool applied = false;
            // Analyze intrinsic nodes
            foreach (Value value in blocks.Values)
            {
                if (value is MethodCall methodCall)
                    applied |= intermediate.RegisterIntrinsic(methodCall.Target);
                else
                    applied |= intermediate.RegisterIntrinsic(value);
            }

            return applied;
        }

        #endregion
    }
}
