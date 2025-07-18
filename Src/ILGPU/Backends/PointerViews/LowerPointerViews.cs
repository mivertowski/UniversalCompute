﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2020-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: LowerPointerViews.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR;
using ILGPU.IR.Rewriting;
using ILGPU.IR.Transformations;
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using ILGPU.Util;

namespace ILGPU.Backends.PointerViews
{
    /// <summary>
    /// Lowers view instances into pointer view implementations.
    /// </summary>
    public sealed class LowerPointerViews : LowerViews
    {
        #region Type Lowering

        /// <summary>
        /// Converts view types into pointer-based structure types.
        /// </summary>
        private sealed class PointerViewLowering(Method.Builder builder) : ViewTypeLowering(builder)
        {

            /// <summary>
            /// Returns the number of fields per view type.
            /// </summary>
            protected override int GetNumFields(ViewType type) => 2;

            /// <summary>
            /// Converts the given view type into a structure with two elements.
            /// </summary>
            protected override TypeNode ConvertType<TTypeContext>(
                TTypeContext typeContext,
                ViewType type)
            {
                var builder = typeContext.CreateStructureType(2);
                builder.Add(typeContext.CreatePointerType(
                    type.ElementType,
                    type.AddressSpace));
                builder.Add(typeContext.GetPrimitiveType(
                    BasicValueType.Int64));
                return builder.Seal();
            }
        }

        #endregion

        #region Rewriter Methods

        /// <summary>
        /// Lowers a new view.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> _,
            NewView value)
        {
            var builder = context.Builder;
            var longLength = builder.CreateConvertToInt64(
                value.Location,
                value.Length);
            var viewInstance = builder.CreateDynamicStructure(
                value.Location,
                value.Pointer,
                longLength);
            context.ReplaceAndRemove(value, viewInstance);
        }

        /// <summary>
        /// Lowers get-view-length property.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> _,
            GetViewLength value)
        {
            var builder = context.Builder;
            var length = builder.CreateGetField(
                value.Location,
                value.View,
                new FieldSpan(1));

            // Convert to a 32bit length value
            if (value.Is32BitProperty)
            {
                length = builder.CreateConvertToInt32(
                    value.Location,
                    length);
            }
            context.ReplaceAndRemove(value, length);
        }

        /// <summary>
        /// Lowers a sub-view value.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> _,
            SubViewValue value)
        {
            var builder = context.Builder;
            var location = value.Location;
            var pointer = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(0));
            var newPointer = builder.CreateLoadElementAddress(
                location,
                pointer,
                value.Offset);

            var length = value.Length;
            if (length.BasicValueType != BasicValueType.Int64)
            {
                length = builder.CreateConvertToInt64(
                    value.Location,
                    length);
            }
            var subView = builder.CreateDynamicStructure(
                location,
                newPointer,
                length);
            context.ReplaceAndRemove(value, subView);
        }

        /// <summary>
        /// Lowers an address-space cast.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> _,
            AddressSpaceCast value)
        {
            var builder = context.Builder;
            var location = value.Location;
            var pointer = builder.CreateGetField(
                location,
                value.Value,
                new FieldSpan(0));
            var length = builder.CreateGetField(
                location,
                value.Value,
                new FieldSpan(1));

            var newPointer = builder.CreateAddressSpaceCast(
                location,
                pointer,
                value.TargetAddressSpace);
            var newInstance = builder.CreateDynamicStructure(
                location,
                newPointer,
                length);
            context.ReplaceAndRemove(value, newInstance);
        }

        /// <summary>
        /// Lowers a view cast.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> typeLowering,
            ViewCast value)
        {
            var builder = context.Builder;
            var location = value.Location;
            var pointer = builder.CreateGetField(
                location,
                value.Value,
                new FieldSpan(0));
            var length = builder.CreateGetField(
                location,
                value.Value,
                new FieldSpan(1));

            // New pointer
            var newPointer = builder.CreatePointerCast(
                location,
                pointer,
                value.TargetElementType);

            // Compute new length:
            // newLength = length * sourceElementSize / targetElementSize;
            var sourceElementType =
                typeLowering[value].AsNotNullCast<ViewType>().ElementType;
            var sourceElementSize = builder.CreateLongSizeOf(
                location,
                sourceElementType);
            var targetElementSize = builder.CreateLongSizeOf(
                location,
                value.TargetElementType);
            var newLength = builder.CreateArithmetic(
                location,
                builder.CreateArithmetic(
                    location,
                    length,
                    sourceElementSize,
                    BinaryArithmeticKind.Mul),
                targetElementSize, BinaryArithmeticKind.Div);

            var newInstance = builder.CreateDynamicStructure(
                location,
                newPointer,
                newLength);
            context.ReplaceAndRemove(value, newInstance);
        }

        /// <summary>
        /// Lowers a lea operation.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> _,
            LoadElementAddress value)
        {
            var builder = context.Builder;
            var location = value.Location;
            var pointer = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(0));
            var newLea = builder.CreateLoadElementAddress(
                location,
                pointer,
                value.Offset);
            context.ReplaceAndRemove(value, newLea);
        }

        /// <summary>
        /// Lowers an align-view-to operation.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> typeLowering,
            AlignTo value)
        {
            var builder = context.Builder;
            var location = value.Location;

            // Extract basic view information from the converted structure
            var pointer = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(0));
            var length = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(1));

            // Build the final result structure instance
            var resultBuilder = builder.CreateDynamicStructure(location);

            // Convert the current input pointer to a 64-bit integer value
            var pointerAsInt = builder.CreatePointerAsIntCast(
                location,
                pointer,
                BasicValueType.Int64);

            // Compute the aligned pointer and convert it to an integer value
            var aligned = builder.CreateAlignTo(
                location,
                pointer,
                value.AlignmentInBytes);
            var alignedAsInt = builder.CreatePointerAsIntCast(
                location,
                aligned,
                BasicValueType.Int64);

            // Compute the number of elements to skip:
            // Min((aligned - ptr) / SizeOf(ElementType), length)
            var viewType = typeLowering[value].As<ViewType>(location);
            var elementsToSkip = builder.CreateArithmetic(
                location,
                builder.CreateArithmetic(
                    location,
                    builder.CreateArithmetic(
                        location,
                        alignedAsInt,
                        pointerAsInt,
                        BinaryArithmeticKind.Sub),
                    builder.CreateSizeOf(location, viewType.ElementType),
                    BinaryArithmeticKind.Div),
                length,
                BinaryArithmeticKind.Min);

            // Create the prefix view that starts at the original pointer offset and
            // includes elementsToSkip many elements.
            {
                resultBuilder.Add(pointer);
                resultBuilder.Add(elementsToSkip);
            }

            // Create the main view that starts at the aligned pointer offset and has a
            // length of remainingLength
            {
                resultBuilder.Add(aligned);
                var remainingLength = builder.CreateArithmetic(
                    location,
                    length,
                    elementsToSkip,
                    BinaryArithmeticKind.Sub);
                resultBuilder.Add(remainingLength);
            }

            var result = resultBuilder.Seal();
            context.ReplaceAndRemove(value, result);
        }

        /// <summary>
        /// Lowers an as-aligned-view operation.
        /// </summary>
        private static void Lower(
            RewriterContext context,
            TypeLowering<ViewType> typeLowering,
            AsAligned value)
        {
            var builder = context.Builder;
            var location = value.Location;

            // Extract basic view information from the converted structure
            var pointer = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(0));
            var length = builder.CreateGetField(
                location,
                value.Source,
                new FieldSpan(1));

            // Ensure that the underyling pointer is aligned
            var aligned = builder.CreateAsAligned(
                location,
                pointer,
                value.AlignmentInBytes);

            // Create a new wrapped instance
            var newInstance = builder.CreateDynamicStructure(
                location,
                aligned,
                length);
            context.ReplaceAndRemove(value, newInstance);
        }

        #endregion

        #region Rewriter

        /// <summary>
        /// The internal rewriter.
        /// </summary>
        private static readonly Rewriter<TypeLowering<ViewType>> Rewriter =
            new();

        /// <summary>
        /// Initializes all rewriter patterns.
        /// </summary>
        static LowerPointerViews()
        {
            AddRewriters(
                Rewriter,
                Lower,
                Lower,
                Lower,
                Lower,
                Lower,
                Lower,
                Lower,
                Lower);
        }

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new pointer view lowering transformation.
        /// </summary>
        public LowerPointerViews() { }

        #endregion

        #region Methods

        /// <summary>
        /// Creates a new <see cref="PointerViewLowering"/> converter.
        /// </summary>
        protected override TypeLowering<ViewType> CreateLoweringConverter(
            Method.Builder builder) =>
            new PointerViewLowering(builder);

        /// <summary>
        /// Applies the pointer view lowering transformation.
        /// </summary>
        protected override bool PerformTransformation(Method.Builder builder) =>
            PerformTransformation(builder, Rewriter);

        #endregion
    }
}
