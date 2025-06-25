// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2018-2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: ViewArgumentMapper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.IL;
using System;

namespace ILGPU.Backends.PointerViews
{
    /// <summary>
    /// Maps array views to pointer implementations.
    /// </summary>
    /// <remarks>Members of this class are not thread safe.</remarks>
    /// <remarks>
    /// Constructs a new view argument mapper.
    /// </remarks>
    /// <param name="context">The current context.</param>
    public abstract class ViewArgumentMapper(Context context) : ArgumentMapper(context)
    {
        #region Nested Types

        /// <summary>
        /// Wraps a value source and created a new view instance from value references.
        /// </summary>
        /// <typeparam name="TSource">The source type.</typeparam>
        private readonly struct ViewImplementationSource<TSource>(TSource source, Type targetType) : IRawValueSource
            where TSource : struct, ISource
        {

            /// <summary>
            /// Returns the parent source.
            /// </summary>
            public TSource Source { get; } = source;

            public Type TargetType { get; } = targetType;

            /// <summary>
            /// Emits a new view-value construction.
            /// </summary>
            public readonly void EmitLoadSource<TILEmitter>(
                in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                // Load source and create custom view type
                Source.EmitLoadSource(emitter);
                emitter.EmitNewObject(
                    ViewImplementation.GetViewConstructor(TargetType));
            }
        }

        #endregion
        #region Instance

        #endregion

        #region Methods

        /// <summary>
        /// Maps an internal view type to a pointer implementation type.
        /// </summary>
        protected sealed override Type MapViewType(Type viewType, Type elementType) =>
            ViewImplementation.GetImplementationType(elementType);

        /// <summary>
        /// Maps an internal view instance to a pointer instance.
        /// </summary>
        protected sealed override void MapViewInstance<TILEmitter, TSource, TTarget>(
            in TILEmitter emitter,
            Type elementType,
            in TSource source,
            in TTarget target)
        {
            var viewSource = new ViewImplementationSource<TSource>(
                source,
                target.TargetType);
            target.EmitStoreTarget(emitter, viewSource);
        }

        #endregion
    }
}
