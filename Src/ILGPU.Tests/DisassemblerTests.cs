﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: DisassemblerTests.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Frontend;
using ILGPU.Frontend.DebugInformation;
using ILGPU.Runtime;
using ILGPU.Util;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class DisassemblerTests(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        private const int Length = 32;

        internal interface IPrefixOffsetFunc
        {
            int Apply(int value);
        }

        internal struct IdentityImpl : IPrefixOffsetFunc, IXunitSerializable
        {
            public readonly int Apply(int value)
            {
                return value;
            }


            public readonly void Serialize(IXunitSerializationInfo info) { }

            public readonly void Deserialize(IXunitSerializationInfo info) { }
        }

        public static TheoryData<object> PrefixOffsetsData => new()
        {
            { default(IdentityImpl) }
        };

        internal static void InstructionPrefixOffsetsKernel<TFunc>(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> data)
            where TFunc : struct, IPrefixOffsetFunc
        {
            TFunc func = default;
            data[index] = func.Apply(Group.IsFirstThread ? default : default);
        }

        [Theory]
        [MemberData(nameof(PrefixOffsetsData))]
        [KernelMethod(nameof(InstructionPrefixOffsetsKernel))]
        [SuppressMessage(
            "Usage",
            "xUnit1026:Theory methods should use all of their parameters",
            Justification = "Required to infer generic type argument")]
        public void InstructionPrefixOffsets<T>(T _)
            where T : unmanaged
        {
            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute<Index1D, T>(buffer.IntExtent, buffer.View);

            var expected = Enumerable.Repeat(0, Length).ToArray();
            Verify(buffer.View, expected);
        }

        internal static void StelemLdtokenKernel<T>(
            Index1D index,
            ArrayView1D<T, Stride1D.Dense> input,
            ArrayView1D<T, Stride1D.Dense> output)
            where T : unmanaged
        {
            output[index] = (dynamic)input[index] * input[index];
        }

        /// <summary>
        /// Tests that the <see cref="Disassembler"/> is able to correctly handle
        /// <see cref="ILOpCode.Ldtoken"/> and <see cref="ILOpCode.Stelem_Ref"/>.
        /// </summary>
        [Theory]
        [InlineData(typeof(byte))]
        [InlineData(typeof(sbyte))]
        [InlineData(typeof(ushort))]
        [InlineData(typeof(short))]
        [InlineData(typeof(uint))]
        [InlineData(typeof(int))]
        [InlineData(typeof(ulong))]
        [InlineData(typeof(long))]
        [InlineData(typeof(float))]
        [InlineData(typeof(double))]
        public void StelemLdtoken(Type genericType)
        {
            var methodInfo = typeof(DisassemblerTests).GetMethod(
                nameof(StelemLdtokenKernel),
                BindingFlags.NonPublic | BindingFlags.Static)
                .ThrowIfNull();
            var method = methodInfo.MakeGenericMethod(genericType);

            var disassembler = new Disassembler(method, SequencePointEnumerator.Empty);
            disassembler.Disassemble();
        }
    }
}
