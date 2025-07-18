﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2018-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: DebugInformationManager.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Reflection;
using System.Threading;

namespace ILGPU.Frontend.DebugInformation
{
    /// <summary>
    /// Represents a debug-information manager.
    /// </summary>
    public sealed class DebugInformationManager : DisposeBase, ICache
    {
        #region Constants

        /// <summary>
        /// The PDB file extension (.pdb).
        /// </summary>
        public const string PDBFileExtensions = ".pdb";

        /// <summary>
        /// The PDB file-search extension (*.pdb).
        /// </summary>
        public const string PDBFileSearchPattern = "*" + PDBFileExtensions;

        #endregion

        #region Nested Types

        /// <summary>
        /// Represents a custom PDB loader.
        /// </summary>
        private interface ILoader
        {
            /// <summary>
            /// Executes the actual loader logic.
            /// </summary>
            /// <param name="assembly">The current assembly.</param>
            /// <param name="assemblyDebugInformation">
            /// The loaded debug-information instance.
            /// </param>
            /// <returns>
            /// True, if the requested debug information could be loaded.
            /// </returns>
            bool Load(
                Assembly assembly,
                [NotNullWhen(true)]
                out AssemblyDebugInformation? assemblyDebugInformation);
        }

        /// <summary>
        /// Represents a file loader for PDB files.
        /// </summary>
        /// <remarks>
        /// Constructs a new file loader.
        /// </remarks>
        /// <param name="parent">The parent manager.</param>
        /// <param name="pdbFileName">The file name to load.</param>
        readonly struct FileLoader(DebugInformationManager parent, string pdbFileName) : ILoader
        {

            /// <summary>
            /// Returns the parent debug-information manager.
            /// </summary>
            public DebugInformationManager Parent { get; } = parent;

            /// <summary>
            /// Returns the file name to load.
            /// </summary>
            public string PDBFileName { get; } = pdbFileName;


            /// <summary cref="ILoader.Load(Assembly, out AssemblyDebugInformation)"/>
            [SuppressMessage(
                "Reliability",
                "CA2000:Dispose objects before losing scope",
                Justification = "The PDB FileStream instance ownership will be " +
                "transfered to the AssemblyDebugInformation")]
            [SuppressMessage(
                "Design",
                "CA1031:Do not catch general exception types",
                Justification = "This method catches all exceptions and redirects them " +
                "to the debug out stream since this method can easily fail in the case " +
                "of an invalid/incompatible/broken PDB file.")]
            public bool Load(
                Assembly assembly,
                out AssemblyDebugInformation assemblyDebugInformation)
            {
                if (Parent.pdbFiles.TryGetValue(PDBFileName, out string? fileName))
                {
                    try
                    {
                        var pdbFileStream = new FileStream(
                            fileName,
                            FileMode.Open,
                            FileAccess.Read);
                        assemblyDebugInformation =
                            new AssemblyDebugInformation(assembly, pdbFileStream);
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine("Error loading PDB file: " + PDBFileName);
                        Debug.WriteLine(ex.ToString());
                        assemblyDebugInformation =
                            new AssemblyDebugInformation(assembly);
                    }
                }
                else
                {
                    assemblyDebugInformation = new AssemblyDebugInformation(assembly);
                }

                Parent.assemblies.Add(assembly, assemblyDebugInformation);
                return assemblyDebugInformation.IsValid;
            }
        }

        /// <summary>
        /// Represents a automatic file loader for PDB files.
        /// </summary>
        /// <remarks>
        /// Constructs a new automatic file loader.
        /// </remarks>
        /// <param name="parent">The parent manager.</param>
        readonly struct AutoFileLoader(DebugInformationManager parent) : ILoader
        {

            /// <summary>
            /// Returns the parent debug-information manager.
            /// </summary>
            public DebugInformationManager Parent { get; } = parent;

            /// <summary cref="ILoader.Load(Assembly, out AssemblyDebugInformation?)"/>
            public bool Load(
                Assembly assembly,
                [NotNullWhen(true)]
                out AssemblyDebugInformation? assemblyDebugInformation)
            {
                assemblyDebugInformation = null;
                if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location))
                    return false;

                var debugDir = Path.GetDirectoryName(assembly.Location);
                if (debugDir != null && !Parent.lookupDirectories.Contains(debugDir))
                    Parent.RegisterLookupDirectory(debugDir);

                var pdbFileName = assembly.GetName().Name.AsNotNull();
                var loader = new FileLoader(Parent, pdbFileName);
                return loader.Load(assembly, out assemblyDebugInformation);
            }
        }

        /// <summary>
        /// Represents a stream loader for PDB files.
        /// </summary>
        /// <remarks>
        /// Constructs a new stream loader.
        /// </remarks>
        /// <param name="parent">The parent manager.</param>
        /// <param name="pdbStream">
        /// The stream to load from (must be left open).
        /// </param>
        readonly struct StreamLoader(DebugInformationManager parent, Stream pdbStream) : ILoader
        {

            /// <summary>
            /// Returns the parent debug-information manager.
            /// </summary>
            public DebugInformationManager Parent { get; } = parent;

            /// <summary>
            /// Returns the stream to load from.
            /// </summary>
            public Stream PDBStream { get; } = pdbStream;

            /// <summary cref="ILoader.Load(Assembly, out AssemblyDebugInformation?)"/>
            [SuppressMessage(
                "Design",
                "CA1031:Do not catch general exception types",
                Justification = "This method catches all exceptions and redirects them " +
                "to the debug out stream since this method can easily fail in the case " +
                "of an invalid/incompatible/broken PDB file.")]
            public bool Load(
                Assembly assembly,
                [NotNullWhen(true)]
                out AssemblyDebugInformation? assemblyDebugInformation)
            {
                try
                {
                    assemblyDebugInformation =
                        new AssemblyDebugInformation(assembly, PDBStream);
                    if (!assemblyDebugInformation.IsValid)
                        return false;

                    Parent.assemblies.Add(assembly, assemblyDebugInformation);
                    return true;
                }
                catch (Exception ex)
                {
                    assemblyDebugInformation = null;
                    Debug.WriteLine($"Error reading from PDB stream for assembly " +
                        $"'{assembly}'");
                    Debug.WriteLine(ex.ToString());
                    return false;
                }
            }
        }

        #endregion

        #region Instance

        private readonly ReaderWriterLockSlim cacheLock =
            new(LockRecursionPolicy.SupportsRecursion);
        private readonly Dictionary<string, string> pdbFiles =
            [];
        private readonly HashSet<string> lookupDirectories = [];
        private readonly Dictionary<Assembly, AssemblyDebugInformation> assemblies =
            [];

        /// <summary>
        /// Constructs a new debug-information manager.
        /// </summary>
        public DebugInformationManager()
        {
            // Register the current application path
            var currentDirectory = Directory.GetCurrentDirectory();
            if (Directory.Exists(currentDirectory))
                RegisterLookupDirectory(currentDirectory);
        }

        #endregion

        #region Methods

        /// <summary>
        /// Tries to load symbols for the given assembly.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="assemblyDebugInformation">
        /// The loaded debug information (or null).
        /// </param>
        /// <returns>True, if the debug information could be loaded.</returns>
        public bool TryLoadSymbols(
            Assembly assembly,
            [NotNullWhen(true)] out AssemblyDebugInformation? assemblyDebugInformation) =>
            TryLoadSymbolsInternal(
                assembly,
                new AutoFileLoader(this),
                out assemblyDebugInformation);

        /// <summary>
        /// Tries to load symbols for the given assembly based on the given
        /// debug-information file.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="pdbFileName">The name of the debug-information file.</param>
        /// <param name="assemblyDebugInformation">
        /// The loaded debug information (or null).
        /// </param>
        /// <returns>True, if the debug information could be loaded.</returns>
        public bool TryLoadSymbols(
            Assembly assembly,
            string pdbFileName,
            [NotNullWhen(true)] out AssemblyDebugInformation? assemblyDebugInformation)
        {
            if (pdbFileName == null)
                throw new ArgumentNullException(nameof(pdbFileName));
            return !File.Exists(pdbFileName)
                ? throw new FileNotFoundException()
                : TryLoadSymbolsInternal(
                assembly,
                new FileLoader(this, pdbFileName),
                out assemblyDebugInformation);
        }

        /// <summary>
        /// Tries to load symbols for the given assembly based on the given PDB stream.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="pdbStream">The source PDB stream (must be left open).</param>
        /// <param name="assemblyDebugInformation">
        /// The loaded debug information (or null).
        /// </param>
        /// <returns>True, if the debug information could be loaded.</returns>
        public bool TryLoadSymbols(
            Assembly assembly,
            Stream pdbStream,
            [NotNullWhen(true)] out AssemblyDebugInformation? assemblyDebugInformation) => pdbStream == null
                ? throw new ArgumentNullException(nameof(pdbStream))
                : TryLoadSymbolsInternal(
                assembly,
                new StreamLoader(this, pdbStream),
                out assemblyDebugInformation);

        /// <summary>
        /// Tries to load symbols for the given assembly based on the given
        /// debug-information file.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="loader">The internal loader.</param>
        /// <param name="assemblyDebugInformation">
        /// The loaded debug information (or null).
        /// </param>
        /// <returns>True, if the debug information could be loaded.</returns>
        private bool TryLoadSymbolsInternal<TLoader>(
            Assembly assembly,
            in TLoader loader,
            [NotNullWhen(true)] out AssemblyDebugInformation? assemblyDebugInformation)
            where TLoader : struct, ILoader
        {
            if (assembly == null)
                throw new ArgumentNullException(nameof(assembly));

            cacheLock.EnterUpgradeableReadLock();
            try
            {
                if (assemblies.TryGetValue(assembly, out assemblyDebugInformation))
                    return true;

                cacheLock.EnterWriteLock();
                try
                {
                    return loader.Load(assembly, out assemblyDebugInformation);
                }
                catch (BadImageFormatException)
                {
                    // Ignore unsuccessful loads here
                    return false;
                }
                finally
                {
                    cacheLock.ExitWriteLock();
                }
            }
            finally
            {
                cacheLock.ExitUpgradeableReadLock();
            }
        }

        /// <summary>
        /// Tries to find a debug-information file with the name
        /// <paramref name="pdbFileName"/>.
        /// </summary>
        /// <param name="pdbFileName">The name of the debug-information file.</param>
        /// <param name="fileName">The resolved filename (or null).</param>
        /// <returns>True, if the given debug-information file could be found.</returns>
        public bool TryFindPbdFile(
            string pdbFileName,
            [NotNullWhen(true)] out string? fileName)
        {
            cacheLock.EnterReadLock();
            try
            {
                return pdbFiles.TryGetValue(pdbFileName, out fileName);
            }
            finally
            {
                cacheLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Registers the given directory as a source directory for
        /// debug-information files.
        /// </summary>
        /// <param name="directory">The directory to register.</param>
        public void RegisterLookupDirectory(string directory)
        {
            if (directory == null)
                throw new ArgumentNullException(nameof(directory));
            if (!Directory.Exists(directory))
                throw new DirectoryNotFoundException();

            cacheLock.EnterUpgradeableReadLock();
            try
            {
                if (lookupDirectories.Contains(directory))
                    return;

                cacheLock.EnterWriteLock();
                try
                {
                    var files = Directory.GetFiles(directory, PDBFileSearchPattern);
                    foreach (var file in files)
                        pdbFiles[Path.GetFileNameWithoutExtension(file)] = file;

                    lookupDirectories.Add(directory);
                }
                finally
                {
                    cacheLock.ExitWriteLock();
                }
            }
            finally
            {
                cacheLock.ExitUpgradeableReadLock();
            }
        }

        /// <summary>
        /// Tries to load debug information for the given method.
        /// </summary>
        /// <param name="methodBase">The method.</param>
        /// <param name="methodDebugInformation">
        /// Loaded debug information (or null).
        /// </param>
        /// <returns>True, if debug information could be loaded.</returns>
        public bool TryLoadDebugInformation(
            MethodBase methodBase,
            [NotNullWhen(true)] out MethodDebugInformation? methodDebugInformation)
        {
            Debug.Assert(methodBase != null, "Invalid method");
            methodDebugInformation = null;
            return TryLoadSymbols(
                methodBase.Module.Assembly,
                out var assemblyDebugInformation) &&
                assemblyDebugInformation.TryLoadDebugInformation(
                    methodBase,
                    out methodDebugInformation);
        }

        /// <summary>
        /// Loads the sequence points of the given method.
        /// </summary>
        /// <param name="methodBase">The method base.</param>
        /// <returns>
        /// A sequence-point enumerator that targets the given method.
        /// </returns>
        /// <remarks>
        /// If no debug information could be loaded for the given method, an empty
        /// <see cref="SequencePointEnumerator"/> will be returned.
        /// </remarks>
        public SequencePointEnumerator LoadSequencePoints(MethodBase methodBase) =>
            TryLoadDebugInformation(
                methodBase,
                out MethodDebugInformation? methodDebugInformation)
            ? methodDebugInformation.CreateSequencePointEnumerator()
            : SequencePointEnumerator.Empty;

        /// <summary>
        /// Clears cached debug information.
        /// </summary>
        /// <param name="mode">The clear mode.</param>
        public void ClearCache(ClearCacheMode mode)
        {
            cacheLock.EnterWriteLock();
            try
            {
                if (mode == ClearCacheMode.Everything)
                {
                    assemblies.Clear();
                }
                else
                {
                    // Do not dispose system assemblies
                    var assembliesToRemove = new List<Assembly>(assemblies.Count);
                    foreach (Assembly assembly in assemblies.Keys)
                    {
                        if (assembly.FullName != null &&
                            assembly.FullName.StartsWith(
                                Context.AssemblyName,
                                StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }
                        assembliesToRemove.Add(assembly);
                    }
                    foreach (var assemblyToRemove in assembliesToRemove)
                        assemblies.Remove(assemblyToRemove);
                }
            }
            finally
            {
                cacheLock.ExitWriteLock();
            }
        }

        #endregion

        #region IDisposable

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
                cacheLock.Dispose();
            base.Dispose(disposing);
        }

        #endregion
    }
}
