// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2023-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ROCmArgumentMapper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.PointerViews;

namespace ILGPU.Backends.ROCm
{
    /// <summary>
    /// Represents a ROCm argument mapper for kernel arguments.
    /// </summary>
    /// <remarks>
    /// ROCm uses a similar argument passing mechanism to OpenCL,
    /// where arguments are passed as individual kernel parameters.
    /// This mapper handles the conversion from ILGPU views to 
    /// ROCm-compatible argument structures.
    /// </remarks>
    /// <param name="context">The current context.</param>
    public sealed class ROCmArgumentMapper(Context context) : ViewArgumentMapper(context)
    {
        // ROCm argument mapping follows similar patterns to OpenCL
        // but may have ROCm-specific optimizations in the future
    }
}