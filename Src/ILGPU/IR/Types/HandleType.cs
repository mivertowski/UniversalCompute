﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: HandleType.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

namespace ILGPU.IR.Types
{
    /// <summary>
    /// Represents a .Net runtime-specific handle type.
    /// </summary>
    public sealed class HandleType : TypeNode
    {
        #region Instance

        /// <summary>
        /// Constructs a new .Net runtime-specific handle type.
        /// </summary>
        /// <param name="typeContext">The parent type context.</param>
        internal HandleType(IRTypeContext typeContext)
            : base(typeContext)
        { }

        #endregion

        #region Properties

        /// <inheritdoc/>
        public override TypeKind TypeKind => TypeKind.Handle;

        #endregion

        #region Methods

        /// <summary>
        /// Creates an object type.
        /// </summary>
        protected override Type GetManagedType<TTypeProvider>(
            TTypeProvider typeProvider) =>
            typeof(object);

        /// <summary cref="TypeNode.Write{T}(T)"/>
        protected internal override void Write<T>(T writer) { }

        #endregion

        #region Object

        /// <summary cref="Node.ToPrefixString"/>
        protected override string ToPrefixString() => "handle";

        /// <summary cref="TypeNode.GetHashCode"/>
        public override int GetHashCode() =>
            base.GetHashCode() ^ 0xAA713C3;

        /// <summary cref="TypeNode.Equals(object?)"/>
        public override bool Equals(object? obj) =>
            obj is HandleType && base.Equals(obj);

        #endregion
    }
}
