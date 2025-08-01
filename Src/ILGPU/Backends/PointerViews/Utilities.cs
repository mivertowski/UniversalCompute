﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2018-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: Utilities.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;

namespace ILGPU.Backends.PointerViews
{
    /// <summary>
    /// General extensions for pointer-based array view implementations.
    /// </summary>
    public static class ViewImplementation
    {
        /// <summary>
        /// The generic implementation type.
        /// </summary>
        public static readonly Type ImplementationType = typeof(ViewImplementation<>);

        /// <summary>
        /// Returns a specialized implementation type.
        /// </summary>
        /// <param name="elementType">The view element type.</param>
        /// <returns>The implement type.</returns>
        [RequiresDynamicCode("Creates generic implementation types at runtime")]
        public static Type GetImplementationType(Type elementType) =>
            ImplementationType.MakeGenericType(elementType);

        /// <summary>
        /// Append all implementation-specific element types.
        /// </summary>
        /// <typeparam name="TCollection">The target collection type.</typeparam>
        /// <param name="collection">The target element collection.</param>
        public static void AppendImplementationTypes<TCollection>(TCollection collection)
            where TCollection : ICollection<Type>
        {
            collection.Add(typeof(void*));
            collection.Add(typeof(long));
        }

        /// <summary>
        /// Returns a specialized view constructor.
        /// </summary>
        /// <param name="implType">The view implementation type.</param>
        /// <returns>The resolved view constructor.</returns>
        [RequiresDynamicCode("Creates generic ArrayView types at runtime")]
        [RequiresUnreferencedCode("Uses reflection to access constructor that may be trimmed")]
        public static ConstructorInfo GetViewConstructor(Type implType) =>
            implType.GetConstructor(
            [
                typeof(ArrayView<>).MakeGenericType(
                    implType.GetGenericArguments()[0])
            ])
            .ThrowIfNull();

        /// <summary>
        /// Returns the pointer field of a view implementation.
        /// </summary>
        /// <param name="implType">The view implementation type.</param>
        /// <returns>The resolved field.</returns>
        public static FieldInfo GetPtrField([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type implType) =>
            implType.GetField(nameof(ViewImplementation<int>.Ptr)).ThrowIfNull();

        /// <summary>
        /// Returns the length field of a view implementation.
        /// </summary>
        /// <param name="implType">The view implementation type.</param>
        /// <returns>The resolved field.</returns>
        public static FieldInfo GetLengthField([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields)] Type implType) =>
            implType.GetField(nameof(ViewImplementation<int>.Length)).ThrowIfNull();

        /// <summary>
        /// The method handle of the <see cref="GetNativePtrMethod(Type)"/> method.
        /// </summary>
        private static readonly MethodInfo GetNativePtrMethodInfo =
            typeof(ViewImplementation).GetMethod(
                nameof(GetNativePtr),
                BindingFlags.NonPublic | BindingFlags.Static)
            .ThrowIfNull();

        /// <summary>
        /// Gets the associated native pointer that is stored inside the given view.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="view">The view type.</param>
        /// <returns>The underlying native pointer.</returns>
        private static IntPtr GetNativePtr<T>(in ArrayView<T> view)
            where T : unmanaged =>
            view.Buffer?.NativePtr ?? IntPtr.Zero;

        /// <summary>
        /// Gets the native-pointer method for the given element type.
        /// </summary>
        /// <param name="elementType">The element type.</param>
        /// <returns>The instantiated native method.</returns>
        public static MethodInfo GetNativePtrMethod(Type elementType) =>
            GetNativePtrMethodInfo.MakeGenericMethod(elementType);
    }
}
