// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: ObjectType.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.IR.Types
{
    /// <summary>
    /// Represents an abstract object value.
    /// </summary>
    /// <remarks>
    /// Constructs a new object type.
    /// </remarks>
    /// <param name="typeContext">The parent type context.</param>
    public abstract class ObjectType(IRTypeContext typeContext) : TypeNode(typeContext)
    {

        #region Instance

        #endregion

        #region Properties

        /// <inheritdoc/>
        public override bool IsObjectType => true;

        #endregion
    }
}
