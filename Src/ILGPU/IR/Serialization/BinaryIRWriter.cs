// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2024 ILGPU Project
//                                    www.ilgpu.net
//
// File: BinaryIRWriter.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.IO;
using System.Text;

namespace ILGPU.IR.Serialization
{
    /// <summary>
    /// Wrapper class around <see cref="BinaryWriter"/>
    /// for serializing IR types and values. 
    /// </summary>
    /// <remarks>
    /// Wraps an instance of <see cref="BinaryIRWriter"/>
    /// around a given <see cref="Stream"/>.
    /// </remarks>
    /// <param name="stream">
    /// The <see cref="Stream"/> instance to wrap.
    /// </param>
    /// <param name="encoding">
    /// The <see cref="Encoding"/> to use for
    /// serializing <see cref="string"/> values.
    /// </param>
    public sealed partial class BinaryIRWriter(Stream stream, Encoding encoding) : DisposeBase, IIRWriter
    {
        private readonly BinaryWriter writer = new BinaryWriter(stream, encoding);

        /// <inheritdoc/>
        public void Write(string tag, int value) =>
            writer.Write(value);

        /// <inheritdoc/>
        public void Write(string tag, long value) =>
            writer.Write(value);

        /// <inheritdoc/>
        public void Write(string tag, string value) =>
            writer.Write(value);

        /// <inheritdoc/>
        public void Write<T>(string tag, T value) where T : Enum =>
            writer.Write(Convert.ToInt32(value));

        /// <inheritdoc/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
                writer.Dispose();
            base.Dispose(disposing);
        }
    }
}
