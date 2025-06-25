// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLCompiledKernel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.Util;

namespace ILGPU.Backends.OpenCL
{
    /// <summary>
    /// Represents a compiled kernel in OpenCL source form.
    /// </summary>
    /// <remarks>
    /// Constructs a new compiled kernel in OpenCL source form.
    /// </remarks>
    /// <param name="context">The associated context.</param>
    /// <param name="entryPoint">The entry point.</param>
    /// <param name="info">Detailed kernel information.</param>
    /// <param name="source">The source code.</param>
    /// <param name="version">The OpenCL C version.</param>
    public sealed class CLCompiledKernel(
        Context context,
        SeparateViewEntryPoint entryPoint,
CompiledKernel.KernelInfo? info,
        string source,
        CLCVersion version) : CompiledKernel(context, entryPoint, info)
    {

        #region Instance

        #endregion

        #region Properties

        /// <summary>
        /// Returns the OpenCL source code.
        /// </summary>
        public string Source { get; } = source;

        /// <summary>
        /// Returns the used OpenCL C version.
        /// </summary>
        public CLCVersion CVersion { get; } = version;

        /// <summary>
        /// Returns the internally used entry point.
        /// </summary>
        internal new SeparateViewEntryPoint EntryPoint =>
            base.EntryPoint.AsNotNullCast<SeparateViewEntryPoint>();

        #endregion
    }
}
