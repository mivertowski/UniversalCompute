﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: LanguageIntrinsics.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR;
using ILGPU.IR.Values;
using ILGPU.Resources;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;
using System.Collections.Immutable;
using System.Text.RegularExpressions;
using static ILGPU.Util.FormatString;
using FormatArray = System.Collections.Immutable.ImmutableArray<
    ILGPU.Util.FormatString.FormatExpression>;

namespace ILGPU.Frontend.Intrinsic
{
    enum LanguageIntrinsicKind
    {
        EmitPTX,
        EmitRefPTX,
    }

    /// <summary>
    /// Marks inline language methods that are built in.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    sealed class LanguageIntrinsicAttribute(LanguageIntrinsicKind intrinsicKind) : IntrinsicAttribute
    {
        public override IntrinsicType Type => IntrinsicType.Language;

        /// <summary>
        /// Returns the assigned intrinsic kind.
        /// </summary>
        public LanguageIntrinsicKind IntrinsicKind { get; } = intrinsicKind;
    }

    partial class Intrinsics
    {
        /// <summary>
        /// Regex for parsing PTX assembly instructions.
        /// </summary>
        private static readonly Regex PTXExpressionRegex =
            // Escape sequence, %n arguments, singular % detection.
            new("(%%|%\\d+|%)");

        /// <summary>
        /// Handles language operations.
        /// </summary>
        /// <param name="context">The current invocation context.</param>
        /// <param name="attribute">The intrinsic attribute.</param>
        /// <returns>The resulting value.</returns>
        private static ValueReference HandleLanguageOperation(
            ref InvocationContext context,
            LanguageIntrinsicAttribute attribute) =>
            attribute.IntrinsicKind switch
            {
                LanguageIntrinsicKind.EmitPTX =>
                    CreateLanguageEmitPTX(ref context, usingRefParams: false),
                LanguageIntrinsicKind.EmitRefPTX =>
                    CreateLanguageEmitPTX(ref context, usingRefParams: true),
                _ => throw context.Location.GetNotSupportedException(
                    ErrorMessages.NotSupportedLanguageIntrinsic,
                    attribute.IntrinsicKind.ToString()),
            };

        /// <summary>
        /// Creates a new inline PTX instruction.
        /// </summary>
        /// <param name="ptxExpression">The PTX expression string.</param>
        /// <param name="usingRefParams">True, if passing parameters by reference.</param>
        /// <param name="context">The current invocation context.</param>
        private static ValueReference CreateLanguageEmitPTX(
            string ptxExpression,
            bool usingRefParams,
            ref InvocationContext context)
        {
            // Parse PTX expression and ensure valid argument references
            var location = context.Location;
            if (!TryParse(ptxExpression, out var expressions))
            {
                throw location.GetNotSupportedException(
                    ErrorMessages.NotSupportedInlinePTXFormat,
                    ptxExpression);
            }

            // Validate all expressions
            foreach (var expression in expressions)
            {
                if (!expression.HasArgument)
                    continue;
                if (expression.Argument < 0 ||
                    expression.Argument >= context.NumArguments - 1)
                {
                    throw location.GetNotSupportedException(
                        ErrorMessages.NotSupportedInlinePTXFormatArgumentRef,
                        ptxExpression,
                        expression.Argument);
                }
            }

            // Gather all arguments
            // The method parameter at position 0 is the PTX string.
            // The method parameter at position 1 is the first argument to the PTX string.
            var capacity = context.Arguments.Count - 1;
            var arguments = InlineList<ValueReference>.Create(capacity);
            var directions = ImmutableArray.CreateBuilder<CudaEmitParameterDirection>(
                capacity);
            var methodParams = context.Method.GetParameters();
            var genericArgs = context.Method.GetGenericArguments();

            for (int i = 1; i < context.Arguments.Count; i++)
            {
                var argument = context.Arguments[i];
                arguments.Add(argument);

                CudaEmitParameterDirection direction;
                if (usingRefParams)
                {
                    var genericArg = genericArgs[i - 1];
                    var genericArgType = genericArg.GetGenericTypeDefinition();

                    if (typeof(Input<>).IsAssignableFrom(genericArgType))
                    {
                        direction = CudaEmitParameterDirection.In;
                    }
                    else if (typeof(Output<>).IsAssignableFrom(genericArgType))
                    {
                        direction = CudaEmitParameterDirection.Out;
                    }
                    else
                    {
                        direction = typeof(Ref<>).IsAssignableFrom(genericArgType)
                            ? CudaEmitParameterDirection.Both
                            : throw location.GetNotSupportedException(
                                                    ErrorMessages.NotSupportedInlinePTXFormatArgumentType,
                                                    ptxExpression,
                                                    genericArg.ToString());
                    }
                }
                else
                {
                    direction = methodParams[i].IsOut
                        ? CudaEmitParameterDirection.Out
                        : CudaEmitParameterDirection.In;
                }
                directions.Add(direction);
            }

            // Valid all argument types
            foreach (var arg in arguments)
            {
                if (arg.BasicValueType != BasicValueType.None)
                    continue;
                throw location.GetNotSupportedException(
                    ErrorMessages.NotSupportedInlinePTXFormatArgumentType,
                    ptxExpression,
                    arg.Type!.ToString());
            }

            // Create the language statement
            return context.Builder.CreateLanguageEmitPTX(
                location,
                usingRefParams,
                expressions,
                directions.ToImmutable(),
                ref arguments);
        }

        /// <summary>
        /// Creates a new inline PTX instruction to the standard output stream.
        /// </summary>
        /// <param name="context">The current invocation context.</param>
        /// <param name="usingRefParams">True, if passing parameters by reference.</param>
        private static ValueReference CreateLanguageEmitPTX(
            ref InvocationContext context,
            bool usingRefParams) =>
            CreateLanguageEmitPTX(
                GetEmitPTXExpression(ref context),
                usingRefParams,
                ref context);

        /// <summary>
        /// Resolves a PTX expression string.
        /// </summary>
        /// <param name="context">The current invocation context.</param>
        /// <returns>The resolved PTX expression string.</returns>
        private static string GetEmitPTXExpression(ref InvocationContext context)
        {
            var ptxExpression = context[0].ResolveAs<StringValue>();
            return ptxExpression is null
                ? throw context.Location.GetNotSupportedException(
                    ErrorMessages.NotSupportedInlinePTXFormatConstant,
                    context[0].ToString())
                : ptxExpression.String ?? string.Empty;
        }

        /// <summary>
        /// Parses the given PTX expression into an array of format expressions.
        /// </summary>
        /// <param name="ptxExpression">The PTX format expression.</param>
        /// <param name="expressions">The array of managed format expressions.</param>
        /// <returns>True, if all expressions could be parsed successfully.</returns>
        public static bool TryParse(
            string ptxExpression,
            out FormatArray expressions)
        {
            // Search for '%n' format arguments
            var parts = PTXExpressionRegex.Split(ptxExpression);
            var result = ImmutableArray.CreateBuilder<FormatExpression>(parts.Length);

            foreach (var part in parts)
            {
                if (part.Equals("%%", StringComparison.Ordinal))
                {
                    result.Add(new FormatExpression("%"));
                }
                else if (part.StartsWith("%", StringComparison.Ordinal))
                {
                    // Check whether the argument can be resolved to an integer.
                    if (int.TryParse(part[1..], out int argument))
                    {
                        result.Add(new FormatExpression(argument));
                    }
                    else
                    {
                        // Singular % or remaining text was not a number.
                        expressions = FormatArray.Empty;
                        return false;
                    }
                }
                else if (part.Length > 0)
                {
                    result.Add(new FormatExpression(part));
                }
            }

            expressions = result.ToImmutable();
            return true;
        }
    }
}
