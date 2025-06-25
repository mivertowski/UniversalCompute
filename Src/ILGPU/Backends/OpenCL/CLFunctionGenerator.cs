// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLFunctionGenerator.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR;
using ILGPU.IR.Analyses;
using ILGPU.IR.Values;
using System.Text;

namespace ILGPU.Backends.OpenCL
{
    /// <summary>
    /// Represents a function generator for helper device functions.
    /// </summary>
    /// <remarks>
    /// Creates a new OpenCL function generator.
    /// </remarks>
    /// <param name="args">The generation arguments.</param>
    /// <param name="method">The current method.</param>
    /// <param name="allocas">All local allocas.</param>
    sealed class CLFunctionGenerator(
        in CLCodeGenerator.GeneratorArgs args,
        Method method,
        Allocas allocas) : CLCodeGenerator(args, method, allocas)
    {
        #region Constants

        /// <summary>
        /// Methods with these flags will be skipped during code generation.
        /// </summary>
        private const MethodFlags MethodFlagsToSkip =
            MethodFlags.External |
            MethodFlags.Intrinsic;

        #endregion

        #region Nested Types

        /// <summary>
        /// A specialized function setup logic for parameters.
        /// </summary>
        /// <remarks>
        /// Constructs a new specialized function setup logic.
        /// </remarks>
        /// <param name="typeGenerator">The parent type generator.</param>
        private readonly struct FunctionParameterSetupLogic(CLTypeGenerator typeGenerator) : IParametersSetupLogic
        {

            /// <summary>
            /// Returns the parent type generator.
            /// </summary>
            public CLTypeGenerator TypeGenerator { get; } = typeGenerator;

            /// <summary>
            /// Returns the internal type for the given parameter.
            /// </summary>
            public string GetParameterType(Parameter parameter) =>
                TypeGenerator[parameter.ParameterType];

            /// <summary>
            /// This setup logic does not support intrinsic parameters.
            /// </summary>
            public Variable? HandleIntrinsicParameter(
                int parameterOffset,
                Parameter parameter) =>
                null;
        }

        #endregion
        #region Instance

        #endregion

        #region Methods

        /// <summary>
        /// Generates a header stub for the current method.
        /// </summary>
        /// <param name="builder">The target builder to use.</param>
        private void GenerateHeaderStub(StringBuilder builder)
        {
            builder.Append(TypeGenerator[Method.ReturnType]);
            builder.Append(' ');
            builder.Append(GetMethodName(Method));
            builder.AppendLine("(");
            var setupLogic = new FunctionParameterSetupLogic(TypeGenerator);
            SetupParameters(builder, ref setupLogic, 0);
            builder.AppendLine(")");
        }

        /// <summary>
        /// Generates a function declaration in OpenCL code.
        /// </summary>
        public override void GenerateHeader(StringBuilder builder)
        {
            if (Method.HasFlags(MethodFlagsToSkip))
                return;

            GenerateHeaderStub(builder);
            builder.AppendLine(";");
        }

        /// <summary>
        /// Generates OpenCL code.
        /// </summary>
        public override void GenerateCode()
        {
            if (Method.HasFlags(MethodFlagsToSkip))
                return;

            // Declare function and parameters
            GenerateHeaderStub(Builder);

            // Bind shared-memory allocations
            BindSharedMemoryAllocation(Allocas.SharedAllocations);
            BindSharedMemoryAllocation(Allocas.DynamicSharedAllocations);

            // Generate code
            BeginFunctionBody();
            GenerateCodeInternal();
            FinishFunctionBody();
        }

        #endregion
    }
}
