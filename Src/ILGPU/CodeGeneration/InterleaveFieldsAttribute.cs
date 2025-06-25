// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2023-2024 ILGPU Project
//                                    www.ilgpu.net
//
// File: InterleaveFieldsAttribute.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

namespace ILGPU.CodeGeneration
{
    /// <summary>
    /// Generates a structure-of-arrays from a definition struct, for a given length.
    /// </summary>
    /// <remarks>
    /// Constructs
    /// </remarks>
    /// <param name="structureType">The definition struct.</param>
    /// <param name="length">The number of elements.</param>
    [AttributeUsage(AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
    public sealed class InterleaveFieldsAttribute(Type structureType, int length) : Attribute
    {
        /// <summary>
        /// The structure type to use as a definition.
        /// </summary>
        public Type StructureType { get; } = structureType;

        /// <summary>
        /// The number of elements in the array.
        /// </summary>
        public int Length { get; } = length;
    }
}
