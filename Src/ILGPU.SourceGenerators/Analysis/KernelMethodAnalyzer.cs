// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
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
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ILGPU.SourceGenerators.Analysis
{
    /// <summary>
    /// Analyzes kernel methods for AOT compatibility and code generation requirements.
    /// </summary>
    internal sealed class KernelMethodAnalyzer(Compilation compilation, SemanticModel semanticModel)
    {
        private readonly Compilation compilation = compilation;
        private readonly SemanticModel semanticModel = semanticModel;

        /// <summary>
        /// Analyzes a method to determine if it's a valid kernel method for AOT compilation.
        /// </summary>
        public KernelAnalysisResult AnalyzeKernelMethod(MethodDeclarationSyntax methodSyntax)
        {
            var methodSymbol = semanticModel.GetDeclaredSymbol(methodSyntax);
            if (methodSymbol == null)
                return KernelAnalysisResult.Failed("Method symbol not found");

            // Check if method is static
            if (!methodSymbol.IsStatic)
                return KernelAnalysisResult.Failed("Kernel methods must be static");

            // Check return type (should be void)
            if (!methodSymbol.ReturnsVoid)
                return KernelAnalysisResult.Failed("Kernel methods must return void");

            // Analyze parameters for AOT compatibility
            var parameterAnalysis = AnalyzeParameters(methodSymbol.Parameters);
            if (!parameterAnalysis.IsValid)
                return KernelAnalysisResult.Failed(parameterAnalysis.Error ?? "Parameter analysis failed");

            // Analyze method body for unsupported features
            var bodyAnalysis = AnalyzeMethodBody(methodSyntax);
            if (!bodyAnalysis.IsValid)
                return KernelAnalysisResult.Failed(bodyAnalysis.Error ?? "Method body analysis failed");

            return KernelAnalysisResult.Success(methodSymbol, parameterAnalysis, bodyAnalysis);
        }

        /// <summary>
        /// Analyzes kernel method parameters for AOT compatibility.
        /// </summary>
        private ParameterAnalysisResult AnalyzeParameters(IEnumerable<IParameterSymbol> parameters)
        {
            var analyzedParameters = new List<AnalyzedParameter>();

            foreach (var parameter in parameters)
            {
                var parameterType = parameter.Type;
                
                // Check for supported parameter types
                if (IsArrayViewType(parameterType))
                {
                    var elementType = GetArrayViewElementType(parameterType);
                    analyzedParameters.Add(new AnalyzedParameter(
                        parameter, 
                        ParameterKind.ArrayView, 
                        parameterType,
                        elementType));
                }
                else if (IsPrimitiveType(parameterType))
                {
                    analyzedParameters.Add(new AnalyzedParameter(
                        parameter, 
                        ParameterKind.Primitive, 
                        parameterType));
                }
                else if (IsStructType(parameterType))
                {
                    analyzedParameters.Add(new AnalyzedParameter(
                        parameter, 
                        ParameterKind.Struct, 
                        parameterType));
                }
                else
                {
                    return ParameterAnalysisResult.Failed(
                        $"Unsupported parameter type '{parameterType}' for parameter '{parameter.Name}'");
                }
            }

            return ParameterAnalysisResult.Success(analyzedParameters);
        }

        /// <summary>
        /// Analyzes the method body for AOT-incompatible features.
        /// </summary>
        private static MethodBodyAnalysisResult AnalyzeMethodBody(MethodDeclarationSyntax methodSyntax)
        {
            var bodyAnalysis = new MethodBodyAnalysisResult();

            // Check for dynamic allocations (new keyword)
            var dynamicAllocations = methodSyntax.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .ToList();

            if (dynamicAllocations.Count != 0)
            {
                bodyAnalysis.AddWarning("Dynamic object allocation found - may not be AOT compatible");
            }

            // Check for reflection usage
            var memberAccess = methodSyntax.DescendantNodes()
                .OfType<MemberAccessExpressionSyntax>()
                .Where(ma => IsReflectionAPI(ma))
                .ToList();

            if (memberAccess.Count != 0)
            {
                return MethodBodyAnalysisResult.Failed("Reflection API usage not allowed in AOT kernels");
            }

            // Check for delegate creation
            var delegateCreations = methodSyntax.DescendantNodes()
                .OfType<AnonymousFunctionExpressionSyntax>()
                .ToList();

            if (delegateCreations.Count != 0)
            {
                return MethodBodyAnalysisResult.Failed("Anonymous functions not supported in AOT kernels");
            }

            return bodyAnalysis;
        }

        private static bool IsArrayViewType(ITypeSymbol type)
        {
            // Check if type is ArrayView<T> or similar ILGPU view types
            return type.Name.Contains("ArrayView") || type.Name.Contains("MemoryBuffer");
        }

        private static ITypeSymbol? GetArrayViewElementType(ITypeSymbol arrayViewType)
        {
            // Extract T from ArrayView<T>
            if (arrayViewType is INamedTypeSymbol namedType && namedType.TypeArguments.Length > 0)
                return namedType.TypeArguments[0];
            return null;
        }

        private static bool IsPrimitiveType(ITypeSymbol type)
        {
            return type.SpecialType switch
            {
                SpecialType.System_Boolean => true,
                SpecialType.System_Byte => true,
                SpecialType.System_SByte => true,
                SpecialType.System_Int16 => true,
                SpecialType.System_UInt16 => true,
                SpecialType.System_Int32 => true,
                SpecialType.System_UInt32 => true,
                SpecialType.System_Int64 => true,
                SpecialType.System_UInt64 => true,
                SpecialType.System_Single => true,
                SpecialType.System_Double => true,
                _ => false
            };
        }

        private static bool IsStructType(ITypeSymbol type)
        {
            return type.TypeKind == TypeKind.Struct && !IsPrimitiveType(type);
        }

        private static bool IsReflectionAPI(MemberAccessExpressionSyntax memberAccess)
        {
            var memberName = memberAccess.Name.Identifier.ValueText;
            return memberName.Contains("GetType") || 
                   memberName.Contains("GetMethod") || 
                   memberName.Contains("GetField") ||
                   memberName.Contains("CreateInstance");
        }
    }
}
