// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ExternalAttribute.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Reflection;

namespace ILGPU.Frontend
{
    /// <summary>
    /// Marks external methods that are opaque in the scope of the ILGPU IR.
    /// </summary>
    /// <remarks>
    /// Constructs a new external attribute.
    /// </remarks>
    /// <param name="name">The external name.</param>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public sealed class ExternalAttribute(string name) : Attribute
    {

        /// <summary>
        /// Returns the associated internal function name.
        /// </summary>
        public string Name { get; } = name;

        /// <summary>
        /// Resolves the actual IR name.
        /// </summary>
        /// <param name="method">The source method.</param>
        /// <returns>The IR name.</returns>
        public string GetName(MethodInfo method) =>
            string.IsNullOrEmpty(Name) ? method.Name : Name;
    }
}
