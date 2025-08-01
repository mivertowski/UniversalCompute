﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ArgumentMapper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.IL;
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using ILGPU.Resources;
using ILGPU.Runtime;
using ILGPU.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ILGPU.Backends.EntryPoints
{
    /// <summary>
    /// Maps kernel arguments to a compatible representation that
    /// can be accessed by the native kernel.
    /// </summary>
    /// <remarks>Members of this class are not thread safe.</remarks>
    public abstract class ArgumentMapper : ICache
    {
        #region Constants

        /// <summary>
        /// The intrinsic kernel length parameter field name.
        /// </summary>
        private const string KernelLengthField = "KernelLength";

        #endregion

        #region Nested Types

        /// <summary>
        /// An emission source.
        /// </summary>
        protected interface IRawValueSource
        {
            /// <summary>
            /// Emits a load command.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <param name="emitter">The current emitter.</param>
            void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter;
        }

        /// <summary>
        /// An emission source.
        /// </summary>
        protected interface ISource : IRawValueSource
        {
            /// <summary>
            /// Returns the source type.
            /// </summary>
            Type SourceType { get; }

            /// <summary>
            /// Emits a load command that loads a reference to the underlying data.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <param name="emitter">The current emitter.</param>
            void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter;
        }

        /// <summary>
        /// An emission target.
        /// </summary>
        protected interface ITarget
        {
            /// <summary>
            /// Returns the target type.
            /// </summary>
            Type TargetType { get; }

            /// <summary>
            /// Emits a target command.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <param name="emitter">The current emitter.</param>
            void EmitLoadTargetAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter;

            /// <summary>
            /// Emits a target command.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <typeparam name="TSource">The source value.</typeparam>
            /// <param name="emitter">The current emitter.</param>
            /// <param name="source">The source value.</param>
            void EmitStoreTarget<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, IRawValueSource;
        }

        /// <summary>
        /// A structure source.
        /// </summary>
        /// <typeparam name="TParentTarget">The parent source type.</typeparam>
        /// <remarks>
        /// Constructs a new structure target.
        /// </remarks>
        /// <param name="parentTarget">The parent target.</param>
        /// <param name="targetField">The target field.</param>
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected readonly struct StructureTarget<TParentTarget>(in TParentTarget parentTarget, FieldInfo targetField) : ITarget
            where TParentTarget : struct, ITarget
        {

            /// <summary cref="ITarget.TargetType"/>
            public Type TargetType => TargetField.FieldType;

            /// <summary>
            /// Returns the parent target.
            /// </summary>
            public TParentTarget ParentTarget { get; } = parentTarget;

            /// <summary>
            /// Returns the target field.
            /// </summary>
            public FieldInfo TargetField { get; } = targetField;

            /// <summary>
            /// Emits a target field address.
            /// </summary>
            public readonly void EmitLoadTargetAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                ParentTarget.EmitLoadTargetAddress(emitter);
                emitter.Emit(OpCodes.Ldflda, TargetField);
            }

            /// <summary>
            /// Emits a store field address.
            /// </summary>
            public readonly void EmitStoreTarget<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, IRawValueSource
            {
                ParentTarget.EmitLoadTargetAddress(emitter);
                source.EmitLoadSource(emitter);
                emitter.Emit(OpCodes.Stfld, TargetField);
            }
        }

        /// <summary>
        /// A <see cref="ILLocal"/> target.
        /// </summary>
        /// <remarks>
        /// Constructs a new local target.
        /// </remarks>
        /// <param name="local">The current local.</param>
        protected readonly struct LocalTarget(ILLocal local) : ITarget
        {

            /// <summary cref="ITarget.TargetType"/>
            public Type TargetType => Local.VariableType;

            /// <summary>
            /// Returns the associated local variable.
            /// </summary>
            public ILLocal Local { get; } = local;

            /// <summary>
            /// Emits a target field address.
            /// </summary>
            public readonly void EmitLoadTargetAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter =>
                emitter.Emit(LocalOperation.LoadAddress, Local);

            /// <summary>
            /// Emits a store local.
            /// </summary>
            public readonly void EmitStoreTarget<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, IRawValueSource
            {
                source.EmitLoadSource(emitter);
                emitter.Emit(LocalOperation.Store, Local);
            }
        }

        /// <summary>
        /// An argument source.
        /// </summary>
        /// <remarks>
        /// Constructs a new argument source.
        /// </remarks>
        /// <param name="type">The argument type.</param>
        /// <param name="argumentIndex">The argument index.</param>
        protected readonly struct ArgumentSource(Type type, int argumentIndex) : ISource
        {

            /// <summary cref="ISource.SourceType"/>
            public Type SourceType { get; } = type;

            /// <summary>
            /// Returns the argument index.
            /// </summary>
            public int ArgumentIndex { get; } = argumentIndex;

            /// <summary>
            /// Emits the address of an argument.
            /// </summary>
            public readonly void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter =>
                emitter.Emit(ArgumentOperation.LoadAddress, ArgumentIndex);

            /// <summary>
            /// Emits the value of an argument.
            /// </summary>
            public readonly void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter =>
                emitter.Emit(ArgumentOperation.Load, ArgumentIndex);
        }

        /// <summary>
        /// A <see cref="ILLocal"/> source.
        /// </summary>
        /// <remarks>
        /// Constructs a new local source.
        /// </remarks>
        /// <param name="local">The current local.</param>
        protected readonly struct LocalSource(ILLocal local) : ISource
        {

            /// <summary cref="ISource.SourceType"/>
            public Type SourceType => Local.VariableType;

            /// <summary>
            /// Returns the associated local variable.
            /// </summary>
            public ILLocal Local { get; } = local;

            /// <summary>
            /// Emits the address of a local variable.
            /// </summary>
            public readonly void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter =>
                emitter.Emit(LocalOperation.LoadAddress, Local);

            /// <summary>
            /// Emits the value of a local variable.
            /// </summary>
            public readonly void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter =>
                emitter.Emit(LocalOperation.Load, Local);
        }

        /// <summary>
        /// A structure source.
        /// </summary>
        /// <typeparam name="TParentSource">The parent source type.</typeparam>
        /// <remarks>
        /// Construct a new structure source.
        /// </remarks>
        /// <param name="parentSource">The parent source.</param>
        /// <param name="sourceField">The source field.</param>
        protected readonly struct StructureSource<TParentSource>(in TParentSource parentSource, FieldInfo sourceField) : ISource
            where TParentSource : struct, ISource
        {

            /// <summary cref="ISource.SourceType"/>
            public Type SourceType => SourceField.FieldType;

            /// <summary>
            /// Returns the parent source.
            /// </summary>
            public TParentSource ParentSource { get; } = parentSource;

            /// <summary>
            /// Returns the source field.
            /// </summary>
            public FieldInfo SourceField { get; } = sourceField;

            /// <summary>
            /// Emits the address of a structure field.
            /// </summary>
            public readonly void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                ParentSource.EmitLoadSourceAddress(emitter);
                emitter.Emit(OpCodes.Ldflda, SourceField);
            }

            /// <summary>
            /// Emits the value of a structure field.
            /// </summary>
            public readonly void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                ParentSource.EmitLoadSource(emitter);
                emitter.Emit(OpCodes.Ldfld, SourceField);
            }
        }

        /// <summary>
        /// A view-parameter source.
        /// </summary>
        /// <remarks>
        /// Constructs a new view source.
        /// </remarks>
        /// <param name="typeInformationManager">
        /// The parent type information manager.
        /// </param>
        /// <param name="source">The underlying source.</param>
        /// <param name="viewParameter">The view parameter to map.</param>
        protected readonly struct ViewSource<TSource>(
            TypeInformationManager typeInformationManager,
            in TSource source,
            in SeparateViewEntryPoint.ViewParameter viewParameter) : ISource
            where TSource : struct, ISource
        {

            /// <summary>
            /// Returns the underlying source.
            /// </summary>
            public TSource Source { get; } = source;

            /// <summary cref="ISource.SourceType"/>
            public Type SourceType { get; } = viewParameter.ViewType;

            /// <summary>
            /// Returns the parameter type.
            /// </summary>
            public TypeInformationManager.TypeInformation ParameterType { get; } = typeInformationManager.GetTypeInfo(
                    viewParameter.ParameterType);

            /// <summary>
            /// Returns the access chain to resolve the actual view instance.
            /// </summary>
            public FieldAccessChain AccessChain { get; } = viewParameter.SourceChain;

            /// <summary>
            /// Loads a nested access chain address.
            /// </summary>
            public readonly void EmitLoadSourceAddress<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                Source.EmitLoadSourceAddress(emitter);
                var type = ParameterType;
                foreach (var fieldIndex in AccessChain)
                {
                    emitter.Emit(OpCodes.Ldflda, type.Fields[(int)fieldIndex]);
                    type = type.GetFieldTypeInfo((int)fieldIndex);
                }
            }

            /// <summary>
            /// Loads a nested access chain.
            /// </summary>
            public readonly void EmitLoadSource<TILEmitter>(in TILEmitter emitter)
                where TILEmitter : struct, IILEmitter
            {
                Source.EmitLoadSource(emitter);
                var type = ParameterType;
                foreach (var fieldIndex in AccessChain)
                {
                    emitter.Emit(OpCodes.Ldfld, type.Fields[(int)fieldIndex]);
                    type = type.GetFieldTypeInfo((int)fieldIndex);
                }
            }
        }

        /// <summary>
        /// An abstract argument mapping handler.
        /// </summary>
        protected interface IMappingHandler
        {
            /// <summary>
            /// Emits a mapping command that maps a kernel argument.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <typeparam name="TSource">The value source type.</typeparam>
            /// <param name="emitter">The target emitter.</param>
            /// <param name="source">The value source.</param>
            /// <param name="argumentIndex">The index of the kernel argument.</param>
            void MapArgument<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source,
                int argumentIndex)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, ISource;
        }

        /// <summary>
        /// An abstract argument mapping handler.
        /// </summary>
        /// <typeparam name="T">The custom return type of this mapper.</typeparam>
        protected interface IStructMappingHandler<T>
        {
            /// <summary>
            /// Returns true if the current kernel supports an implicit kernel length.
            /// </summary>
            /// <param name="indexType">The current index type (if any).</param>
            /// <returns>
            /// True, if the current kernel supports an implicit kernel length.
            /// </returns>
            bool CanMapKernelLength(out Type indexType);

            /// <summary>
            /// Maps a kernel length parameter.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <typeparam name="TTarget">The value target type.</typeparam>
            /// <param name="emitter">The target emitter.</param>
            /// <param name="kernelLengthTarget">The length target.</param>
            void MapKernelLength<TILEmitter, TTarget>(
                in TILEmitter emitter,
                in StructureTarget<TTarget> kernelLengthTarget)
                where TILEmitter : struct, IILEmitter
                where TTarget : struct, ITarget;

            /// <summary>
            /// Emits a mapping command that maps all kernel arguments via a struct.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <param name="emitter">The target emitter.</param>
            /// <param name="local">The local variable reference.</param>
            /// <param name="rawSizeInBytesWithoutPadding">
            /// The raw size in bytes of the argument structure without taking the
            /// structure-conversion induced alignment padding into account.
            /// </param>
            T MapArgumentStruct<TILEmitter>(
                in TILEmitter emitter,
                ILLocal local,
                int rawSizeInBytesWithoutPadding)
                where TILEmitter : struct, IILEmitter;
        }

        /// <summary>
        /// An abstract argument mapping handler.
        /// </summary>
        protected interface ISeparateViewMappingHandler
        {
            /// <summary>
            /// Emits a set of commands that map an implementation view instance
            /// and stores the converted instance into the given target.
            /// </summary>
            /// <typeparam name="TILEmitter">The emitter type.</typeparam>
            /// <typeparam name="TSource">The value source type.</typeparam>
            /// <param name="emitter">The current emitter.</param>
            /// <param name="source">The value source.</param>
            /// <param name="viewParameter">The source view parameter.</param>
            /// <param name="viewArgumentIndex">The argument index.</param>
            void MapViewArgument<TILEmitter, TSource>(
                in TILEmitter emitter,
                in TSource source,
                in SeparateViewEntryPoint.ViewParameter viewParameter,
                int viewArgumentIndex)
                where TILEmitter : struct, IILEmitter
                where TSource : struct, ISource;
        }

        #endregion

        #region Static

        /// <summary>
        /// Constructs a new field name based on the relative field index within a
        /// dynamically generated structure.
        /// </summary>
        /// <param name="index">The relative field index.</param>
        /// <returns>The field name.</returns>
        private static string GetFieldName(int index)
        {
            Debug.Assert(index >= 0, "Invalid field index");
            return StructureType.GetFieldName(index);
        }

        #endregion

        #region Instance

        /// <summary>
        /// The internal type mapping (from old to new types).
        /// </summary>
        private readonly Dictionary<Type, Type> typeMapping =
            [];

        /// <summary>
        /// Constructs a new argument mapper.
        /// </summary>
        /// <param name="context">The current context.</param>
        protected ArgumentMapper(Context context)
        {
            if (context is null)
                throw new ArgumentNullException(nameof(context));

            RuntimeSystem = context.RuntimeSystem;
            TypeContext = context.TypeContext;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the current runtime system.
        /// </summary>
        public RuntimeSystem RuntimeSystem { get; }

        /// <summary>
        /// Returns the current type context.
        /// </summary>
        public IRTypeContext TypeContext { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Maps a view type to its implementation specific type.
        /// </summary>
        /// <param name="viewType">The view type.</param>
        /// <param name="elementType">The element type.</param>
        /// <returns>The resulting implementation type.</returns>
        protected abstract Type MapViewType(Type viewType, Type elementType);

        /// <summary>
        /// Maps the given structure type to a compatible structure type.
        /// </summary>
        /// <param name="structType">The structure type to map.</param>
        /// <returns>The mapped structure type.</returns>
        protected Type MapStructType(Type structType)
        {
            // Check all element types
            var typeInfo = TypeContext.GetTypeInfo(structType);
            var sourceFields = typeInfo.Fields;
            if (sourceFields.Length < 1)
                return structType;

            var nestedTypes = InlineList<Type>.Create(sourceFields.Length);
            bool requireCustomType = false;
            for (int i = 0, e = sourceFields.Length; i < e; ++i)
            {
                var sourceFieldType = sourceFields[i].FieldType;
                var fieldType = MapType(sourceFieldType);
                requireCustomType |= fieldType != sourceFieldType;
                nestedTypes.Add(fieldType);
            }
            if (!requireCustomType)
                return structType;

            // We need a custom structure type and map all fields
            using var scopedLock = RuntimeSystem.DefineRuntimeStruct(
                out var typeBuilder);
            for (int i = 0, e = sourceFields.Length; i < e; ++i)
            {
                typeBuilder.DefineField(
                    GetFieldName(i),
                    nestedTypes[i],
                    FieldAttributes.Public);
            }
            // Build wrapper type and return it
            return typeBuilder.CreateType();
        }

        /// <summary>
        /// Registers a type mapping entry and returns the mapped type.
        /// </summary>
        /// <param name="type">The source type.</param>
        /// <param name="mappedType">The target type.</param>
        /// <returns>The mapped type.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected Type RegisterTypeMapping(Type type, Type mappedType)
        {
            typeMapping.Add(type, mappedType);
            return mappedType;
        }

        /// <summary>
        /// Maps the given source type to a compatible target type.
        /// </summary>
        /// <param name="type">The source type.</param>
        /// <returns>The compatible target type.</returns>
        [SuppressMessage(
            "Style",
            "IDE0046:Convert to conditional expression",
            Justification = "Difficult to understand in this case")]
        protected Type MapType(Type type)
        {
            Debug.Assert(type != null, "Invalid source type");
            if (typeMapping.TryGetValue(type, out Type? mappedType))
                return mappedType;

            if (type.IsByRef)
            {
                throw new NotSupportedException(
                    ErrorMessages.NotSupportedByRefKernelParameters);
            }
            else if (type.IsArray || type.IsClass)
            {
                throw new NotSupportedException(string.Format(
                    ErrorMessages.NotSupportedClassType,
                    type));
            }
            else if (type == typeof(void))
            {
                throw new NotSupportedException(ErrorMessages.NotSupportedVoidType);
            }
            else if (type.IsPointer || type.IsVoidPtr())
            {
                throw new NotSupportedException(string.Format(
                    ErrorMessages.NotSupportedPointerType,
                    type));
            }
            else if (type.IsDelegate())
            {
                throw new NotSupportedException(string.Format(
                    ErrorMessages.NotSupportedDelegateType,
                    type));
            }

            if (type.IsILGPUPrimitiveType())
                return RegisterTypeMapping(type, type);
            else if (type.IsPointer)
                return RegisterTypeMapping(type, typeof(void*));
            else if (type.IsEnum)
                return RegisterTypeMapping(type, type.GetEnumUnderlyingType());
            else if (type.IsArrayViewType(out Type? elementType))
                return RegisterTypeMapping(type, MapViewType(type, elementType));
            else
                return RegisterTypeMapping(type, MapStructType(type));
        }

        /// <summary>
        /// Emits a set of commands that map an implementation view instance
        /// and stores the converted instance into the given target.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TSource">The value source type.</typeparam>
        /// <typeparam name="TTarget">The value target type.</typeparam>
        /// <param name="emitter">The current emitter.</param>
        /// <param name="elementType">The element type.</param>
        /// <param name="source">The value source.</param>
        /// <param name="target">The value target.</param>
        protected abstract void MapViewInstance<TILEmitter, TSource, TTarget>(
            in TILEmitter emitter,
            Type elementType,
            in TSource source,
            in TTarget target)
            where TILEmitter : struct, IILEmitter
            where TSource : struct, ISource
            where TTarget : struct, ITarget;

        /// <summary>
        /// Maps a specific structure instance.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TSource">The value source type.</typeparam>
        /// <typeparam name="TTarget">The value target type.</typeparam>
        /// <param name="emitter">The current emitter.</param>
        /// <param name="source">The value source.</param>
        /// <param name="target">The value target.</param>
        protected void MapStructInstance<TILEmitter, TSource, TTarget>(
            in TILEmitter emitter,
            in TSource source,
            in TTarget target)
            where TILEmitter : struct, IILEmitter
            where TSource : struct, ISource
            where TTarget : struct, ITarget
        {
            // Resolve type info of source and target types
            var sourceInfo = TypeContext.GetTypeInfo(source.SourceType);
            var targetInfo = TypeContext.GetTypeInfo(target.TargetType);
            Debug.Assert(
                sourceInfo.NumFields == targetInfo.NumFields,
                "Incompatible types");

            // Map all field entries
            for (int i = 0, e = sourceInfo.NumFields; i < e; ++i)
            {
                var fieldSource = new StructureSource<TSource>(
                    source,
                    sourceInfo.Fields[i]);
                var fieldTarget = new StructureTarget<TTarget>(
                    target,
                    targetInfo.Fields[i]);
                MapInstance(emitter, fieldSource, fieldTarget);
            }
        }

        /// <summary>
        /// Maps a value instance.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TSource">The value source type.</typeparam>
        /// <typeparam name="TTarget">The value target type.</typeparam>
        /// <param name="emitter">The current emitter.</param>
        /// <param name="source">The value source.</param>
        /// <param name="target">The value target.</param>
        protected void MapInstance<TILEmitter, TSource, TTarget>(
            in TILEmitter emitter,
            in TSource source,
            in TTarget target)
            where TILEmitter : struct, IILEmitter
            where TSource : struct, ISource
            where TTarget : struct, ITarget
        {
            var sourceType = source.SourceType;
            if (sourceType == target.TargetType ||
                sourceType.IsEnum)
            {
                // Copy object from source to target
                target.EmitStoreTarget(emitter, source);
            }
            else if (sourceType.IsArrayViewType(out Type? elementType))
            {
                MapViewInstance(emitter, elementType, source, target);
            }
            else
            {
                MapStructInstance(emitter, source, target);
            }
        }

        /// <summary>
        /// Maps a single parameter value.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TTarget">The value target type.</typeparam>
        /// <param name="emitter">The current emitter.</param>
        /// <param name="parameters">The parameter collection to map.</param>
        /// <param name="index">The source parameter index.</param>
        /// <param name="parameterTarget">The parameter local target.</param>
        private void MapParameter<TILEmitter, TTarget>(
            in TILEmitter emitter,
            in ParameterCollection parameters,
            int index,
            TTarget parameterTarget)
            where TILEmitter : struct, IILEmitter
            where TTarget : struct, ITarget
        {
            if (parameters.IsByRef(index))
            {
                throw new NotSupportedException(
                    ErrorMessages.InvalidEntryPointParameter);
            }

            // Load parameter argument and map instance
            var parameterType = parameters.ParameterTypes[index];
            var parameterIndex = index + Kernel.KernelParameterOffset;
            var argumentSource = new ArgumentSource(parameterType, parameterIndex);

            // Perform actual instance mapping on local
            MapInstance(emitter, argumentSource, parameterTarget);
        }

        /// <summary>
        /// Creates code that maps the given parameter specification to
        /// a compatible representation.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TMappingHandler">The handler type.</typeparam>
        /// <param name="emitter">The target emitter to write to.</param>
        /// <param name="mappingHandler">The target mapping handler to use.</param>
        /// <param name="parameters">The parameter collection to map.</param>
        protected void MapArguments<TILEmitter, TMappingHandler>(
            in TILEmitter emitter,
            in TMappingHandler mappingHandler,
            in ParameterCollection parameters)
            where TILEmitter : struct, IILEmitter
            where TMappingHandler : struct, IMappingHandler
        {
            // Map all parameters
            for (int i = 0, e = parameters.Count; i < e; ++i)
            {
                var paramType = parameters.ParameterTypes[i];
                try
                {
                    // Ensure kernel parameters are blittable.
                    if (!TypeContext.GetTypeInfo(paramType).IsValidKernelParameter)
                    {
                        throw new NotSupportedException(
                            string.Format(
                                RuntimeErrorMessages.NotSupportedNonBlittableType,
                                paramType.FullName));
                    }

                    // Map type and store the mapped instance in a pinned local
                    var mappedType = MapType(paramType);
                    var mappingLocal = emitter.DeclarePinnedLocal(mappedType);
                    var localTarget = new LocalTarget(mappingLocal);

                    // Perform actual instance mapping on local
                    MapParameter(emitter, parameters, i, localTarget);

                    // Map the argument from the pinned local
                    var localSource = new LocalSource(mappingLocal);
                    mappingHandler.MapArgument(emitter, localSource, i);
                }
                catch (NotSupportedException nse)
                {
                    throw new ArgumentException(
                        string.Format(
                            ErrorMessages.NotSupportedKernelParameterType,
                            paramType),
                        nse);
                }
            }
        }

        /// <summary>
        /// Creates code that maps the given parameter specification to
        /// a compatible representation.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TMappingHandler">The handler type.</typeparam>
        /// <typeparam name="T">The custom handler type of this mapper.</typeparam>
        /// <param name="emitter">The target emitter to write to.</param>
        /// <param name="mappingHandler">The target mapping handler to use.</param>
        /// <param name="parameters">The parameter collection to map.</param>
        protected T MapArgumentsStruct<TILEmitter, TMappingHandler, T>(
            in TILEmitter emitter,
            in TMappingHandler mappingHandler,
            in ParameterCollection parameters)
            where TILEmitter : struct, IILEmitter
            where TMappingHandler : struct, IStructMappingHandler<T>
        {
            // Map type and store the mapped instance in a pinned local
            var argumentType = CreateArgumentStructType<TMappingHandler, T>(
                mappingHandler,
                parameters);

            var mappingLocal = emitter.DeclarePinnedLocal(argumentType);
            var localTarget = new LocalTarget(mappingLocal);

            // Map the kernel length separately (if any)
            var kernelLength = argumentType.GetField(KernelLengthField);
            if (kernelLength != null)
            {
                var kernelLengthLocal = new StructureTarget<LocalTarget>(
                    localTarget,
                    kernelLength);
                mappingHandler.MapKernelLength(emitter, kernelLengthLocal);
            }

            // Map all parameter instances
            for (int i = 0, e = parameters.Count; i < e; ++i)
            {
                var fieldTarget = new StructureTarget<LocalTarget>(
                    localTarget,
                    argumentType.GetField(GetFieldName(i)).AsNotNull());

                // Perform actual instance mapping on local
                MapParameter(emitter, parameters, i, fieldTarget);
            }

            // Compute the actual raw size of the argument structure without taking
            // the actual alignment-induced padding into account
            var lastFieldName = GetFieldName(parameters.Count - 1);
            int lastOffset = Marshal.OffsetOf(argumentType, lastFieldName).ToInt32();
            int lastFieldSize = Interop.SizeOf(
                argumentType.GetField(lastFieldName).AsNotNull().FieldType);

            // Map the whole argument structure
            return mappingHandler.MapArgumentStruct(
                emitter,
                mappingLocal,
                lastOffset + lastFieldSize);
        }

        /// <summary>
        /// Creates a mapping argument structure type.
        /// </summary>
        /// <typeparam name="TMappingHandler">The handler type.</typeparam>
        /// <typeparam name="T">The custom handler type of this mapper.</typeparam>
        /// <param name="mappingHandler">The target mapping handler to use.</param>
        /// <param name="parameters">The parameter collection to map.</param>
        /// <returns>The argument mapping structure type.</returns>
        [return: DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)]
        private Type CreateArgumentStructType<TMappingHandler, T>(
            in TMappingHandler mappingHandler,
            in ParameterCollection parameters)
            where TMappingHandler : struct, IStructMappingHandler<T>
        {
            using var scopedLock = RuntimeSystem.DefineRuntimeStruct(
                out var typeBuilder);

            // Define the main kernel length
            if (mappingHandler.CanMapKernelLength(out var indexType))
            {
                typeBuilder.DefineField(
                    KernelLengthField,
                    indexType,
                    FieldAttributes.Public);
            }

            // Define all parameter types
            for (int i = 0, e = parameters.Count; i < e; ++i)
            {
                var paramType = parameters.ParameterTypes[i];
                try
                {
                    // Ensure kernel parameters are blittable.
                    if (!TypeContext.GetTypeInfo(paramType).IsValidKernelParameter)
                    {
                        throw new NotSupportedException(
                            string.Format(
                                RuntimeErrorMessages.NotSupportedNonBlittableType,
                                paramType.FullName));
                    }

                    // Map parameter to argument struct.
                    var mappedType = MapType(paramType);
                    typeBuilder.DefineField(
                        GetFieldName(i),
                        mappedType,
                        FieldAttributes.Public);
                }
                catch (NotSupportedException nse)
                {
                    throw new ArgumentException(
                        string.Format(
                            ErrorMessages.NotSupportedKernelParameterType,
                            paramType),
                        nse);
                }
            }

            return typeBuilder.CreateType();
        }

        /// <summary>
        /// Creates code that maps (potentially nested) views of kernel arguments
        /// separately.
        /// </summary>
        /// <typeparam name="TILEmitter">The emitter type.</typeparam>
        /// <typeparam name="TMappingHandler">The handler type.</typeparam>
        /// <param name="emitter">The target emitter to write to.</param>
        /// <param name="mappingHandler">The target mapping handler to use.</param>
        /// <param name="typeInformationManager">
        /// The parent type information manager.
        /// </param>
        /// <param name="entryPoint">The entry point to use.</param>
        protected static void MapViews<TILEmitter, TMappingHandler>(
            in TILEmitter emitter,
            in TMappingHandler mappingHandler,
            TypeInformationManager typeInformationManager,
            SeparateViewEntryPoint entryPoint)
            where TILEmitter : struct, IILEmitter
            where TMappingHandler : struct, ISeparateViewMappingHandler
        {
            Debug.Assert(entryPoint != null, "Invalid entry point");

            // Resolve all information from all kernel arguments
            int viewArgumentIndex = 0;
            var specification = entryPoint.Parameters;
            for (int i = 0, e = specification.Count; i < e; ++i)
            {
                if (specification.IsByRef(i))
                {
                    throw new NotSupportedException(
                        ErrorMessages.InvalidEntryPointParameter);
                }

                // Check for matching view specifications
                if (!entryPoint.TryGetViewParameters(i, out var views))
                    continue;

                // Load parameter argument source and resolve the access chain
                var parameterType = specification.ParameterTypes[i];
                var parameterIndex = i + Kernel.KernelParameterOffset;
                var argumentSource = new ArgumentSource(parameterType, parameterIndex);

                // Map all view parameters
                foreach (var view in views)
                {
                    var viewSource = new ViewSource<ArgumentSource>(
                        typeInformationManager,
                        argumentSource,
                        view);
                    mappingHandler.MapViewArgument(
                        emitter,
                        viewSource,
                        view,
                        viewArgumentIndex++);
                }
            }
        }

        /// <summary>
        /// Clears internal caches.
        /// </summary>
        /// <param name="mode">The clear mode.</param>
        public void ClearCache(ClearCacheMode mode) =>
            typeMapping.Clear();

        #endregion
    }
}
