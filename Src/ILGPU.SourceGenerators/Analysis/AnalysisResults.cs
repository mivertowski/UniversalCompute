// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.Linq;

namespace ILGPU.SourceGenerators.Analysis
{
    /// <summary>
    /// Result of kernel method analysis for AOT compatibility.
    /// </summary>
    internal sealed class KernelAnalysisResult
    {
        public bool IsValid { get; }
        public string? Error { get; }
        public IMethodSymbol? MethodSymbol { get; }
        public ParameterAnalysisResult? ParameterAnalysis { get; }
        public MethodBodyAnalysisResult? BodyAnalysis { get; }

        private KernelAnalysisResult(
            bool isValid, 
            string? error, 
            IMethodSymbol? methodSymbol,
            ParameterAnalysisResult? parameterAnalysis,
            MethodBodyAnalysisResult? bodyAnalysis)
        {
            IsValid = isValid;
            Error = error;
            MethodSymbol = methodSymbol;
            ParameterAnalysis = parameterAnalysis;
            BodyAnalysis = bodyAnalysis;
        }

        public static KernelAnalysisResult Success(
            IMethodSymbol methodSymbol,
            ParameterAnalysisResult parameterAnalysis,
            MethodBodyAnalysisResult bodyAnalysis)
        {
            return new KernelAnalysisResult(true, null, methodSymbol, parameterAnalysis, bodyAnalysis);
        }

        public static KernelAnalysisResult Failed(string error)
        {
            return new KernelAnalysisResult(false, error, null, null, null);
        }
    }

    /// <summary>
    /// Result of parameter analysis for AOT compatibility.
    /// </summary>
    internal sealed class ParameterAnalysisResult
    {
        public bool IsValid { get; }
        public string? Error { get; }
        public IReadOnlyList<AnalyzedParameter> Parameters { get; }

        private ParameterAnalysisResult(bool isValid, string? error, IReadOnlyList<AnalyzedParameter> parameters)
        {
            IsValid = isValid;
            Error = error;
            Parameters = parameters;
        }

        public static ParameterAnalysisResult Success(IEnumerable<AnalyzedParameter> parameters)
        {
            return new ParameterAnalysisResult(true, null, parameters.ToList());
        }

        public static ParameterAnalysisResult Failed(string error)
        {
            return new ParameterAnalysisResult(false, error, new List<AnalyzedParameter>());
        }
    }

    /// <summary>
    /// Result of method body analysis for AOT compatibility.
    /// </summary>
    internal sealed class MethodBodyAnalysisResult
    {
        public bool IsValid { get; private set; } = true;
        public string? Error { get; private set; }
        public List<string> Warnings { get; } = [];

        public void AddWarning(string warning)
        {
            Warnings.Add(warning);
        }

        public void SetError(string error)
        {
            IsValid = false;
            Error = error;
        }

        public static MethodBodyAnalysisResult Failed(string error)
        {
            var result = new MethodBodyAnalysisResult();
            result.SetError(error);
            return result;
        }
    }

    /// <summary>
    /// Represents an analyzed kernel parameter.
    /// </summary>
    internal sealed class AnalyzedParameter(IParameterSymbol symbol, ParameterKind kind, ITypeSymbol type, ITypeSymbol? elementType = null)
    {
        public IParameterSymbol Symbol { get; } = symbol;
        public ParameterKind Kind { get; } = kind;
        public ITypeSymbol Type { get; } = type;
        public ITypeSymbol? ElementType { get; } = elementType;
    }

    /// <summary>
    /// Categories of kernel parameters for code generation.
    /// </summary>
    internal enum ParameterKind
    {
        /// <summary>
        /// Primitive value types (int, float, etc.)
        /// </summary>
        Primitive,

        /// <summary>
        /// ILGPU ArrayView<T> or similar view types
        /// </summary>
        ArrayView,

        /// <summary>
        /// User-defined struct types
        /// </summary>
        Struct
    }
}
