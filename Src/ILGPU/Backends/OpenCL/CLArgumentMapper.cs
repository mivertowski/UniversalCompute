﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLArgumentMapper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.IL;
using ILGPU.Backends.SeparateViews;
using ILGPU.IR.Types;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using ILGPU.Util;
using System;
using System.Reflection;
using System.Reflection.Emit;

namespace ILGPU.Backends.OpenCL
{
    /// <summary>
    /// Constructs mappings for CL kernels.
    /// </summary>
    /// <remarks>Members of this class are not thread safe.</remarks>
    /// <remarks>
    /// Constructs a new OpenCL argument mapper.
    /// </remarks>
    /// <param name="context">The current context.</param>
    public sealed class CLArgumentMapper(Context context) : ViewArgumentMapper(context)
    {
        #region Static

        /// <summary>
        /// The method to set OpenCL kernel arguments.
        /// </summary>
        private static readonly MethodInfo SetKernelArgumentMethod =
            typeof(CLAPI).GetMethod(
                nameof(CLAPI.SetKernelArgumentUnsafeWithKernel),
                BindingFlags.Public | BindingFlags.Instance)
            .ThrowIfNull();

        #endregion

        #region Nested Types

        /// <summary>
        /// Implements the actual argument mapping.
        /// </summary>
        /// <remarks>
        /// Constructs a new mapping handler.
        /// </remarks>
        /// <param name="parent">The parent mapper.</param>
        /// <param name="kernelLocal">
        /// The local variable holding the associated kernel reference.
        /// </param>
        /// <param name="resultLocal">
        /// The local variable holding the result API status.
        /// </param>
        /// <param name="startIndex">The start argument index.</param>
        private readonly struct MappingHandler(
            CLArgumentMapper parent,
            ILLocal kernelLocal,
            ILLocal resultLocal,
            int startIndex) : IMappingHandler
        {
            /// <summary>
            /// A source mapper.
            /// </summary>
            /// <typeparam name="TSource">The internal source type.</typeparam>
            /// <remarks>
            /// Constructs a new source mapper.
            /// </remarks>
            /// <param name="source">The underlying source.</param>
            private readonly struct MapperSource<TSource>(TSource source) : ISource
                where TSource : struct, ISource
            {

                /// <summary>
                /// Returns the associated source.
                /// </summary>
                public TSource Source { get; } = source;

                /// <summary cref="ArgumentMapper.ISource.SourceType"/>
                public Type SourceType => Source.SourceType;

                /// <summary>
                /// Emits a nested source address.
                /// </summary>
                public readonly void EmitLoadSourceAddress<TILEmitter>(
                    in TILEmitter emitter)
                    where TILEmitter : struct, IILEmitter =>
                    Source.EmitLoadSourceAddress(emitter);

                /// <summary>
                /// Emits a nested source value.
                /// </summary>
                public readonly void EmitLoadSource<TILEmitter>(
                    in TILEmitter emitter)
                    where TILEmitter : struct, IILEmitter =>
                    Source.EmitLoadSource(emitter);
            }

            /// <summary>
            /// Returns the underlying ABI.
            /// </summary>
            public CLArgumentMapper Parent { get; } = parent;

            /// <summary>
            /// Returns the associated kernel local.
            /// </summary>
            public ILLocal KernelLocal { get; } = kernelLocal;

            /// <summary>
            /// Returns the associated result variable which is
            /// used to accumulate all intermediate method return values.
            /// </summary>
            public ILLocal ResultLocal { get; } = resultLocal;

            /// <summary>
            /// Returns the start argument index.
            /// </summary>
            public int StartIndex { get; } = startIndex;

            /// <summary>
            /// Emits code to set an individual argument.
            /// </summary>
            public readonly void MapArgument<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source,
                int argumentIndex)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, ISource =>
                Parent.SetKernelArgument(
                    emitter,
                    KernelLocal,
                    ResultLocal,
                    StartIndex + argumentIndex,
                    new MapperSource<TSource>(source));
        }

        /// <summary>
        /// Implements the actual argument mapping.
        /// </summary>
        /// <remarks>
        /// Constructs a new mapping handler.
        /// </remarks>
        /// <param name="parent">The parent mapper.</param>
        /// <param name="kernelLocal">
        /// The local variable holding the associated kernel reference.
        /// </param>
        /// <param name="resultLocal">
        /// The local variable holding the result API status.
        /// </param>
        /// <param name="startIndex">The start argument index.</param>
        private readonly struct ViewMappingHandler(
            CLArgumentMapper parent,
            ILLocal kernelLocal,
            ILLocal resultLocal,
            int startIndex) : ISeparateViewMappingHandler
        {
            /// <summary>
            /// A source mapper.
            /// </summary>
            /// <typeparam name="TSource">The internal source type.</typeparam>
            /// <remarks>
            /// Constructs a new source mapper.
            /// </remarks>
            /// <param name="source">The underlying source.</param>
            /// <param name="viewParameter">The view parameter.</param>
            private readonly struct MapperSource<TSource>(
                TSource source,
                in SeparateViewEntryPoint.ViewParameter viewParameter) : ISource
                where TSource : struct, ISource
            {

                /// <summary>
                /// Returns the associated source.
                /// </summary>
                public TSource Source { get; } = source;

                /// <summary cref="ArgumentMapper.ISource.SourceType"/>
                public Type SourceType => typeof(IntPtr);

                /// <summary>
                /// The associated parameter.
                /// </summary>
                public SeparateViewEntryPoint.ViewParameter Parameter { get; } = viewParameter;

                /// <summary>
                /// Emits a source local that contains the native view pointer.
                /// </summary>
                private ILLocal EmitSourceLocal<TILEmitter>(in TILEmitter emitter)
                    where TILEmitter : struct, IILEmitter
                {
                    // Load source
                    Source.EmitLoadSourceAddress(emitter);

                    // Extract native pointer
                    emitter.EmitCall(
                        ViewImplementation.GetNativePtrMethod(Parameter.ElementType));

                    // Store the resolved pointer in a local variable in order to pass
                    // the reference to the local to the actual set-argument method.
                    var tempLocal = emitter.DeclareLocal(typeof(IntPtr));
                    emitter.Emit(LocalOperation.Store, tempLocal);

                    return tempLocal;
                }

                /// <summary>
                /// Converts a view into its native implementation form and maps it to
                /// an argument.
                /// </summary>
                public void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                    where TILEmitter : struct, IILEmitter
                {
                    var tempLocal = EmitSourceLocal(emitter);
                    emitter.Emit(LocalOperation.LoadAddress, tempLocal);
                }

                /// <summary>
                /// Converts a view into its native implementation form and maps it to
                /// an argument.
                /// </summary>
                public void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                    where TILEmitter : struct, IILEmitter
                {
                    var tempLocal = EmitSourceLocal(emitter);
                    emitter.Emit(LocalOperation.Load, tempLocal);
                }
            }

            /// <summary>
            /// Returns the underlying ABI.
            /// </summary>
            public CLArgumentMapper Parent { get; } = parent;

            /// <summary>
            /// Returns the associated kernel local.
            /// </summary>
            public ILLocal KernelLocal { get; } = kernelLocal;

            /// <summary>
            /// Returns the associated result variable which is
            /// used to accumulate all intermediate method return values.
            /// </summary>
            public ILLocal ResultLocal { get; } = resultLocal;

            /// <summary>
            /// Returns the start argument index.
            /// </summary>
            public int StartIndex { get; } = startIndex;

            /// <summary>
            /// Maps a view input argument.
            /// </summary>
            public void MapViewArgument<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source,
                in SeparateViewEntryPoint.ViewParameter viewParameter,
                int viewArgumentIndex)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, ISource =>
                Parent.SetKernelArgument(
                    emitter,
                    KernelLocal,
                    ResultLocal,
                    StartIndex + viewArgumentIndex,
                    new MapperSource<TSource>(source, viewParameter));
        }

        #endregion
        #region Instance

        #endregion

        #region Methods

        /// <summary>
        /// Returns the ABI size of the given managed type.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>The interop size in bytes.</returns>
        private int GetSizeOf(Type type) => TypeContext.CreateType(type).Size;

        /// <summary>
        /// Emits code that sets an OpenCL kernel argument.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TSource">The value source type.</typeparam>
        /// <param name="emitter">The current emitter.</param>
        /// <param name="kernelLocal">
        /// The local variable holding the associated kernel reference.
        /// </param>
        /// <param name="resultLocal">
        /// The local variable holding the result API status.
        /// </param>
        /// <param name="argumentIndex">The argument index.</param>
        /// <param name="source">The value source.</param>
        private void SetKernelArgument<TILEmitter, TSource>(
            in TILEmitter emitter,
            ILLocal kernelLocal,
            ILLocal resultLocal,
            int argumentIndex,
            in TSource source)
            where TILEmitter : struct, IILEmitter
            where TSource : struct, ISource
        {
            // Load current driver API
            emitter.EmitCall(CLAccelerator.GetCLAPIMethod);

            // Load kernel reference
            emitter.Emit(LocalOperation.Load, kernelLocal);

            // Load target argument index
            emitter.EmitConstant(argumentIndex);

            // Load size of the argument value
            var size = GetSizeOf(source.SourceType);
            emitter.EmitConstant(size);

            // Load source address
            source.EmitLoadSourceAddress(emitter);

            // Set argument
            emitter.EmitCall(SetKernelArgumentMethod);

            // Merge API results
            emitter.Emit(LocalOperation.Load, resultLocal);
            emitter.Emit(OpCodes.Or);
            emitter.Emit(LocalOperation.Store, resultLocal);
        }

        /// <summary>
        /// Creates code that maps all parameters of the given entry point using
        /// OpenCL API calls.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <param name="emitter">The target emitter to write to.</param>
        /// <param name="kernel">A local that holds the kernel driver reference.</param>
        /// <param name="typeInformationManager">
        /// The parent type information manager.
        /// </param>
        /// <param name="entryPoint">The entry point.</param>
        public void Map<TILEmitter>(
            in TILEmitter emitter,
            ILLocal kernel,
            TypeInformationManager typeInformationManager,
            SeparateViewEntryPoint entryPoint)
            where TILEmitter : struct, IILEmitter
        {
            if (entryPoint == null)
                throw new ArgumentNullException(nameof(entryPoint));

            // Declare local
            var resultLocal = emitter.DeclareLocal(typeof(int));
            emitter.Emit(OpCodes.Ldc_I4_0);
            emitter.Emit(LocalOperation.Store, resultLocal);

            // Compute the base offset that reserves additional parameters for dynamic
            // shared memory allocations - buffer and buffer size.
            int baseOffset = entryPoint.SharedMemory.HasDynamicMemory
                ? 2
                : 0;

            // Map all views
            var viewMappingHandler = new ViewMappingHandler(
                this,
                kernel,
                resultLocal,
                baseOffset);
            MapViews(
                emitter,
                viewMappingHandler,
                typeInformationManager,
                entryPoint);

            // Map implicit kernel length (if required)
            int parameterOffset = entryPoint.NumViewParameters + baseOffset;
            if (!entryPoint.IsExplicitlyGrouped)
            {
                var lengthSource = new ArgumentSource(
                    entryPoint.KernelIndexType,
                    Kernel.KernelParamDimensionIdx);
                SetKernelArgument(
                    emitter,
                    kernel,
                    resultLocal,
                    parameterOffset,
                    lengthSource);
                ++parameterOffset;
            }

            // Map all remaining arguments
            var mappingHandler = new MappingHandler(
                this,
                kernel,
                resultLocal,
                parameterOffset);
            MapArguments(emitter, mappingHandler, entryPoint.Parameters);

            // Check mapping result
            emitter.Emit(LocalOperation.Load, resultLocal);
            emitter.EmitCall(CLAccelerator.ThrowIfFailedMethod);
        }

        #endregion
    }
}
