// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2022-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: VelocityArgumentMapper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.PTX;

namespace ILGPU.Backends.Velocity
{
    /// <summary>
    /// Constructs mappings Velocity kernels.
    /// </summary>
    /// <remarks>The current velocity backend uses the PTX argument mapper.</remarks>
    /// <remarks>
    /// Constructs a new IL argument mapper.
    /// </remarks>
    /// <param name="context">The current context.</param>
    sealed class VelocityArgumentMapper(Context context) : PTXArgumentMapper(context)
    {

        #region Instance

        #endregion
    }
}
