// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLVariableAllocator.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Analyses;
using ILGPU.IR.Values;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.Backends.OpenCL
{
    /// <summary>
    /// Represents a specialized OpenCL variable allocator.
    /// </summary>
    /// <remarks>
    /// Constructs a new register allocator.
    /// </remarks>
    /// <param name="typeGenerator">The associated type generator.</param>
    public class CLVariableAllocator(CLTypeGenerator typeGenerator) : VariableAllocator
    {
        #region Nested Types

        /// <summary>
        /// A virtual globally accessible shared memory variable.
        /// </summary>
        /// <remarks>
        /// Instances of this class will not return valid variable ids.
        /// </remarks>
        /// <remarks>
        /// Constructs a new variable instance.
        /// </remarks>
        /// <param name="allocaInfo">The source allocation info.</param>
        private sealed class GloballySharedMemoryVariable(in AllocaInformation allocaInfo) : TypedVariable(-1, allocaInfo.Alloca.Type)
        {

            /// <summary>
            /// Returns the allocation name.
            /// </summary>
            public string Name { get; } = GetSharedMemoryAllocationName(allocaInfo);

            /// <summary>
            /// Returns the allocation name.
            /// </summary>
            /// <returns>The allocation name.</returns>
            public override string ToString() => Name;
        }

        /// <summary>
        /// A virtual globally accessible shared memory length variable.
        /// </summary>
        /// <remarks>
        /// Instances of this class will not return valid variable ids.
        /// </remarks>
        /// <remarks>
        /// Constructs a new variable instance.
        /// </remarks>
        /// <param name="allocaInfo">The source allocation info.</param>
        private sealed class GloballySharedMemoryLengthVariable(in AllocaInformation allocaInfo) : TypedVariable(-1, allocaInfo.Alloca.Type)
        {

            /// <summary>
            /// Returns the allocation name.
            /// </summary>
            public string Name { get; } = GetSharedMemoryAllocationLengthName(allocaInfo);

            /// <summary>
            /// Returns the allocation name.
            /// </summary>
            /// <returns>The allocation name.</returns>
            public override string ToString() => Name;
        }

        #endregion

        #region Static

        /// <summary>
        /// Returns a shared memory allocation variable reference.
        /// </summary>
        /// <param name="allocaInfo">The source allocation info.</param>
        /// <returns>
        /// The allocation variable reference pointing to the allocation object.
        /// </returns>
        public static Variable GetSharedMemoryAllocationVariable(
            in AllocaInformation allocaInfo) =>
            new GloballySharedMemoryVariable(allocaInfo);

        /// <summary>
        /// Returns a unique shared memory allocation name.
        /// </summary>
        /// <param name="allocaInfo">The source allocation info.</param>
        /// <returns>The allocation name.</returns>
        public static string GetSharedMemoryAllocationName(
            in AllocaInformation allocaInfo) =>
            "shared_var_" + allocaInfo.Alloca.Id;

        /// <summary>
        /// Returns a shared memory allocation length variable reference.
        /// </summary>
        /// <param name="allocaInfo">The source allocation info.</param>
        /// <returns>
        /// The allocation variable reference pointing to the allocation object.
        /// </returns>
        public static Variable GetSharedMemoryAllocationLengthVariable(
            in AllocaInformation allocaInfo) =>
            new GloballySharedMemoryLengthVariable(allocaInfo);

        /// <summary>
        /// Returns a unique shared memory allocation length name.
        /// </summary>
        /// <param name="allocaInfo">The source allocation info.</param>
        /// <returns>The allocation name.</returns>
        public static string GetSharedMemoryAllocationLengthName(
            in AllocaInformation allocaInfo) =>
            GetSharedMemoryAllocationLengthName(allocaInfo.Alloca);

        /// <summary>
        /// Returns a unique shared memory allocation length name.
        /// </summary>
        /// <param name="alloca">The source allocation operation.</param>
        /// <returns>The allocation name.</returns>
        internal static string GetSharedMemoryAllocationLengthName(
            Alloca alloca) =>
            "shared_var_len_" + alloca.Id;

        #endregion
        #region Instance

        #endregion

        #region Properties

        /// <summary>
        /// Returns the associated type generator.
        /// </summary>
        public CLTypeGenerator TypeGenerator { get; } = typeGenerator;

        #endregion

        #region Methods

        /// <summary>
        /// Resolves the type name of the given variable.
        /// </summary>
        /// <param name="variable">The variable.</param>
        /// <returns>The resolved variable type name.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string GetVariableType(Variable variable) =>
            variable switch
            {
                PrimitiveVariable primitiveVariable =>
                    TypeGenerator.GetBasicValueType(
                        primitiveVariable.BasicValueType),
                TypedVariable typedVariable => TypeGenerator[typedVariable.Type],
                _ => throw new NotSupportedException(),
            };

        #endregion
    }
}
