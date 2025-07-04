// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2020-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: AnalysisValue.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Types;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace ILGPU.IR.Analyses
{
    /// <summary>
    /// An abstract analysis value context.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    public interface IAnalysisValueSourceContext<T>
        where T : IEquatable<T>
    {
        /// <summary>
        /// Returns the analysis value associated with the given value.
        /// </summary>
        /// <param name="value">The source value to lookup.</param>
        /// <returns>The parent value.</returns>
        AnalysisValue<T> this[Value value] { get; }
    }

    /// <summary>
    /// A default implementation of an <see cref="IAnalysisValueSourceContext{T}"/>
    /// that always returns a specific constant value.
    /// </summary>
    /// <typeparam name="T">The value type.</typeparam>
    /// <remarks>
    /// Constructs a new source context.
    /// </remarks>
    /// <param name="value">The constant value to use for all nodes.</param>
    public readonly struct ConstAnalysisValueSourceContext<T>(T value) :
        IAnalysisValueSourceContext<T>
        where T : IEquatable<T>
    {

        /// <summary>
        /// Returns the constant value to use for all nodes.
        /// </summary>
        public T Value { get; } = value;

        /// <summary>
        /// Returns the value of <see cref="Value"/> for all input nodes.
        /// </summary>
        public readonly AnalysisValue<T> this[Value value] =>
            AnalysisValue.Create(Value, value.Type.AsNotNull());
    }

    /// <summary>
    /// An abstract analysis value context.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    public interface IAnalysisValueContext<T> : IAnalysisValueSourceContext<T>
        where T : IEquatable<T>
    {
        /// <summary>
        /// Returns the analysis value associated with the given the method.
        /// </summary>
        /// <param name="method">The source method to lookup.</param>
        /// <returns>The parent value.</returns>
        AnalysisValue<T> this[Method method] { get; }
    }

    /// <summary>
    /// An analysis value to encapsulate static program analysis values.
    /// </summary>
    /// <typeparam name="T">The underlying element type.</typeparam>
    /// <remarks>
    /// This type encapsulates a general <see cref="Data"/> element that represents
    /// accumulated analysis information for the whole object. Furthermore, it stores
    /// additional fine-grained information about each child element in the case of
    /// structure values. This improves the overall program analysis precision.
    /// </remarks>
    /// <remarks>
    /// Constructs a new analysis value with different data values for each child.
    /// </remarks>
    /// <param name="data">The accumulated data value.</param>
    /// <param name="childArray">All child data values.</param>
    public readonly struct AnalysisValue<T>(T data, T[] childArray) :
        IEquatable<AnalysisValue<T>>
        where T : IEquatable<T>
    {
        #region Instance

        private readonly T[] childData = childArray ?? [];

        /// <summary>
        /// Constructs a new analysis value with the given data value.
        /// </summary>
        /// <param name="data">The accumulated data value.</param>
        public AnalysisValue(T data)
            : this(data, [])
        { }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the underlying data value.
        /// </summary>
        public T Data { get; } = data;

        /// <summary>
        /// Returns the number of child elements.
        /// </summary>
        public readonly int NumFields => childData.Length;

        /// <summary>
        /// Returns the i-th child data element.
        /// </summary>
        /// <param name="index">The child index.</param>
        /// <returns></returns>
        public readonly T this[int index] => childData[index];

        #endregion

        #region Methods

        /// <summary>
        /// Clones the internal child-data array into a new one.
        /// </summary>
        /// <returns>The cloned child-data array.</returns>
        public readonly T[] CloneChildData()
        {
            var newChildData = new T[NumFields];
            Array.Copy(childData, newChildData, NumFields);
            return newChildData;
        }

        #endregion

        #region IEquatable

        /// <summary>
        /// Returns true if the given value is equal to the current one.
        /// </summary>
        /// <param name="other">The other value.</param>
        /// <returns>True, if the given value is equal to the current one.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly bool Equals(AnalysisValue<T> other)
        {
            if (!Data.Equals(other.Data) || NumFields != other.NumFields)
                return false;
            for (int i = 0, e = NumFields; i < e; ++i)
            {
                if (!childData[i].Equals(other.childData[i]))
                    return false;
            }
            return true;
        }

        #endregion

        #region Object

        /// <summary>
        /// Returns true if the given object is equal to the current value.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True, if the given object is equal to the current value.</returns>
        public readonly override bool Equals(object? obj) =>
            obj is AnalysisValue<T> value && Equals(value);

        /// <summary>
        /// Returns the hash code of this value.
        /// </summary>
        /// <returns>The hash code of this value.</returns>
        public readonly override int GetHashCode() => Data.GetHashCode() ^ NumFields;

        /// <summary>
        /// Returns the string representation of this value.
        /// </summary>
        /// <returns>The string representation of this value.</returns>
        public readonly override string? ToString() =>
            childData.Length > 0
            ? $"{Data} [{string.Join(", ", childData)}]"
            : Data.ToString();

        #endregion

        #region Operators

        /// <summary>
        /// Returns true if the first and second value are the same.
        /// </summary>
        /// <param name="first">The first value.</param>
        /// <param name="second">The second value.</param>
        /// <returns>True, if the first and second value are the same.</returns>
        public static bool operator ==(
            AnalysisValue<T> first,
            AnalysisValue<T> second) =>
            first.Equals(second);

        /// <summary>
        /// Returns true if the first and second value are not the same.
        /// </summary>
        /// <param name="first">The first value.</param>
        /// <param name="second">The second value.</param>
        /// <returns>True, if the first and second value are not the same.</returns>
        public static bool operator !=(
            AnalysisValue<T> first,
            AnalysisValue<T> second) =>
            !(first == second);

        #endregion
    }

    /// <summary>
    /// Helper methods for the structure <see cref="AnalysisValue{T}"/>.
    /// </summary>
    public static class AnalysisValue
    {
        /// <summary>
        /// Creates a new analysis value for the given type node.
        /// </summary>
        /// <param name="data">The data value.</param>
        /// <param name="type">The type node.</param>
        /// <returns>The created analysis value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static AnalysisValue<T> Create<T>(T data, TypeNode type)
            where T : IEquatable<T>
        {
            if (type is StructureType structureType)
            {
                var childData = new T[structureType.NumFields];
                for (int i = 0, e = childData.Length; i < e; ++i)
                    childData[i] = data;
                return new AnalysisValue<T>(data, childData);
            }
            return new AnalysisValue<T>(data);
        }
    }

    /// <summary>
    /// Maps <see cref="Value"/> instances to <see cref="AnalysisValue{T}"/> instances
    /// specialized using the user-defined type <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The target mapping type.</typeparam>
    /// <remarks>
    /// Constructs a new value mapping using the given dictionary.
    /// </remarks>
    /// <param name="data">The underlying dictionary to use.</param>
    public readonly struct AnalysisValueMapping<T>(Dictionary<Value, AnalysisValue<T>> data)
        where T : IEquatable<T>
    {
        #region Instance

        private readonly Dictionary<Value, AnalysisValue<T>> mapping = data ?? throw new ArgumentNullException(nameof(data));

        #endregion

        #region Properties

        /// <summary>
        /// Lookups the given key in this map.
        /// </summary>
        /// <param name="key">The key to lookup.</param>
        /// <returns>The resolved analysis value.</returns>
        public AnalysisValue<T> this[Value key]
        {
            readonly get => mapping[key];
            internal set => mapping[key] = value;
        }

        /// <summary>
        /// Returns the number of elements in this mapping.
        /// </summary>
        public readonly int Count => mapping?.Count ?? 0;

        #endregion

        #region Methods

        /// <summary>
        /// Returns true if the given key is contained in this map.
        /// </summary>
        /// <param name="key">The key to lookup.</param>
        /// <returns>True, if the given key is contained in this map.</returns>
        public readonly bool ContainsKey(Value key) =>
            mapping.ContainsKey(key);

        /// <summary>
        /// Tries to get map the given key to a stored value.
        /// </summary>
        /// <param name="key">The key to lookup.</param>
        /// <param name="value">The resolved value (if any).</param>
        /// <returns>True, if the given key could be found.</returns>
        public readonly bool TryGetValue(Value key, out AnalysisValue<T> value) =>
            mapping.TryGetValue(key, out value);

        #endregion

        #region IEnumerable

        /// <summary>
        /// Returns an enumerator to enumerate all items in this mapping.
        /// </summary>
        /// <returns>The enumerator.</returns>
        public readonly Dictionary<Value, AnalysisValue<T>>.Enumerator GetEnumerator() =>
            mapping.GetEnumerator();

        #endregion
    }

    /// <summary>
    /// Maps <see cref="Method"/> instances to <see cref="AnalysisValue{T}"/> instances
    /// specialized using the user-defined type <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The target mapping type.</typeparam>
    /// <remarks>
    /// Constructs a new value mapping using the given dictionary.
    /// </remarks>
    /// <param name="data">The underlying dictionary to use.</param>
    public readonly struct AnalysisReturnValueMapping<T>(Dictionary<Method, AnalysisValue<T>> data)
        where T : IEquatable<T>
    {
        #region Instance

        private readonly Dictionary<Method, AnalysisValue<T>> mapping = data ?? throw new ArgumentNullException(nameof(data));

        #endregion

        #region Properties

        /// <summary>
        /// Lookups the given key in this map.
        /// </summary>
        /// <param name="key">The key to lookup.</param>
        /// <returns>The resolved analysis value.</returns>
        public AnalysisValue<T> this[Method key]
        {
            readonly get => mapping[key];
            internal set => mapping[key] = value;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Tries to get map the given key to a stored value.
        /// </summary>
        /// <param name="key">The key to lookup.</param>
        /// <param name="value">The resolved value (if any).</param>
        /// <returns>True, if the given key could be found.</returns>
        public readonly bool TryGetValue(Method key, out AnalysisValue<T> value) =>
            mapping.TryGetValue(key, out value);

        #endregion
    }

    /// <summary>
    /// Helper methods for the structure <see cref="AnalysisValueMapping{T}"/>.
    /// </summary>
    public static class AnalysisValueMapping
    {
        /// <summary>
        /// Creates a new analysis mapping instance.
        /// </summary>
        /// <typeparam name="T">The target mapping type.</typeparam>
        /// <returns>The initialized analysis mapping instance.</returns>
        public static AnalysisValueMapping<T> Create<T>()
            where T : struct, IEquatable<T> =>
            new(
                []);
    }

    /// <summary>
    /// Helper methods for the structure <see cref="AnalysisReturnValueMapping{T}"/>.
    /// </summary>
    public static class AnalysisReturnValueMapping
    {
        /// <summary>
        /// Creates a new analysis return mapping instance.
        /// </summary>
        /// <typeparam name="T">The target mapping type.</typeparam>
        /// <returns>The initialized analysis mapping instance.</returns>
        public static AnalysisReturnValueMapping<T> Create<T>()
            where T : struct, IEquatable<T> =>
            new(
                []);
    }
}
