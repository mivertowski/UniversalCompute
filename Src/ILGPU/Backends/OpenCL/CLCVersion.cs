﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLCVersion.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System.Text.RegularExpressions;

namespace ILGPU.Backends.OpenCL
{
    /// <summary>
    /// Represents an OpenCL C version.
    /// </summary>
    /// <remarks>
    /// Constructs a new OpenCL C version.
    /// </remarks>
    /// <param name="major">The major version.</param>
    /// <param name="minor">The minor version.</param>
    public readonly struct CLCVersion(int major, int minor)
    {
        #region Static

        /// <summary>
        /// The OpenCL C version 1.0.
        /// </summary>
        public static readonly CLCVersion CL10 = new(1, 0);

        /// <summary>
        /// The OpenCL C version 1.1.
        /// </summary>
        public static readonly CLCVersion CL11 = new(1, 1);

        /// <summary>
        /// The OpenCL C version 1.2.
        /// </summary>
        public static readonly CLCVersion CL12 = new(1, 2);

        /// <summary>
        /// The OpenCL C version 2.0.
        /// </summary>
        public static readonly CLCVersion CL20 = new(2, 0);

        /// <summary>
        /// The OpenCL C version 3.0.
        /// </summary>
        public static readonly CLCVersion CL30 = new(3, 0);

        /// <summary>
        /// The internal regex that is used to parse OpenCL C versions.
        /// </summary>
        private static readonly Regex VersionRegex =
            new("\\s*(CL|OpenCL C)?\\s*([0-9]+).([0-9]+)");

        /// <summary>
        /// Tries to parse the given string expression into an OpenCL C version.
        /// </summary>
        /// <param name="expression">The expression to parse.</param>
        /// <param name="version">The parsed version (if any).</param>
        /// <returns>
        /// True, if the given expression could be parsed into an OpenCL C version.
        /// </returns>
        public static bool TryParse(string expression, out CLCVersion version)
        {
            version = default;
            var match = VersionRegex.Match(expression);
            if (!match.Success)
                return false;
            version = new CLCVersion(
                int.Parse(match.Groups[2].Value),
                int.Parse(match.Groups[3].Value));
            return true;
        }

        #endregion
        #region Instance

        #endregion

        #region Properties

        /// <summary>
        /// The major OpenCL C Version.
        /// </summary>
        public int Major { get; } = major;

        /// <summary>
        /// The minor OpenCL C Version.
        /// </summary>
        public int Minor { get; } = minor;

        #endregion

        #region Object

        /// <summary>
        /// Returns the OpenCL C string representation that is compatible
        /// with the OpenCL API.
        /// </summary>
        /// <returns>The string representation of this OpenCL C version.</returns>
        public override string ToString() => $"CL{Major}.{Minor}";

        #endregion

        #region Operators

        /// <summary>
        /// Returns true if the first version is smaller than the second one.
        /// </summary>
        /// <param name="first">The first version.</param>
        /// <param name="second">The second version.</param>
        /// <returns>True, if the first version is smaller than the second one.</returns>
        public static bool operator <(CLCVersion first, CLCVersion second) =>
            first.Major < second.Major ||
            first.Major == second.Major && first.Minor < second.Minor;

        /// <summary>
        /// Returns true if the first version is greater than the second one.
        /// </summary>
        /// <param name="first">The first version.</param>
        /// <param name="second">The second version.</param>
        /// <returns>True, if the first version is greater than the second one.</returns>
        public static bool operator >(CLCVersion first, CLCVersion second) =>
            first.Major > second.Major ||
            first.Major == second.Major && first.Minor > second.Minor;

        /// <summary>
        /// Returns true if the first version is smaller than or equal to the second one.
        /// </summary>
        /// <param name="first">The first version.</param>
        /// <param name="second">The second version.</param>
        /// <returns>True, if the first version is smaller than the second one.</returns>
        public static bool operator <=(CLCVersion first, CLCVersion second) =>
            first.Major <= second.Major ||
            first.Major == second.Major && first.Minor <= second.Minor;

        /// <summary>
        /// Returns true if the first version is greater than or equal to the second one.
        /// </summary>
        /// <param name="first">The first version.</param>
        /// <param name="second">The second version.</param>
        /// <returns>True, if the first version is greater than the second one.</returns>
        public static bool operator >=(CLCVersion first, CLCVersion second) =>
            first.Major >= second.Major ||
            first.Major == second.Major && first.Minor >= second.Minor;

        #endregion
    }
}
