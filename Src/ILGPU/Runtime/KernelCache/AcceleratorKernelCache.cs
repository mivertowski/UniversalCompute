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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace ILGPU.Runtime.KernelCache
{
    /// <summary>
    /// Integrates kernel caching with ILGPU accelerators.
    /// </summary>
    public static class AcceleratorKernelCache
    {
        private static readonly Dictionary<Accelerator, IKernelCache> acceleratorCaches = [];
        private static readonly object lockObject = new();

        /// <summary>
        /// Gets or creates a kernel cache for the specified accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="options">Cache options (optional).</param>
        /// <returns>The kernel cache for the accelerator.</returns>
        public static IKernelCache GetOrCreateCache(Accelerator accelerator, KernelCacheOptions? options = null)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            lock (lockObject)
            {
                if (acceleratorCaches.TryGetValue(accelerator, out var existingCache))
                {
                    return existingCache;
                }

                var cache = new KernelCacheManager(options);
                acceleratorCaches[accelerator] = cache;
                return cache;
            }
        }

        /// <summary>
        /// Loads a kernel with caching support.
        /// </summary>
        /// <typeparam name="TDelegate">The kernel delegate type.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="action">The kernel action.</param>
        /// <param name="specialization">Optional kernel specialization.</param>
        /// <returns>The loaded kernel with caching.</returns>
        public static CachedKernel<TDelegate> LoadKernelCached<TDelegate>(
            this Accelerator accelerator,
            TDelegate action,
            KernelSpecialization? specialization = null)
            where TDelegate : Delegate
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            var cache = GetOrCreateCache(accelerator);
            return new CachedKernel<TDelegate>(accelerator, action, cache, specialization);
        }

        /// <summary>
        /// Clears all caches for all accelerators.
        /// </summary>
        public static void ClearAllCaches()
        {
            lock (lockObject)
            {
                foreach (var cache in acceleratorCaches.Values)
                {
                    cache.Clear();
                }
            }
        }

        /// <summary>
        /// Gets statistics for all accelerator caches.
        /// </summary>
        /// <returns>A dictionary of accelerator to cache statistics.</returns>
        public static Dictionary<string, KernelCacheStatistics> GetAllStatistics()
        {
            var result = new Dictionary<string, KernelCacheStatistics>();
            
            lock (lockObject)
            {
                foreach (var kvp in acceleratorCaches)
                {
                    var acceleratorName = $"{kvp.Key.AcceleratorType}_{kvp.Key.DeviceId}";
                    result[acceleratorName] = kvp.Value.GetStatistics();
                }
            }

            return result;
        }

        /// <summary>
        /// Disposes all accelerator caches.
        /// </summary>
        internal static void DisposeAllCaches()
        {
            lock (lockObject)
            {
                foreach (var cache in acceleratorCaches.Values)
                {
                    cache.Dispose();
                }
                acceleratorCaches.Clear();
            }
        }
    }

    /// <summary>
    /// Represents a cached kernel with automatic cache management.
    /// </summary>
    /// <typeparam name="TDelegate">The kernel delegate type.</typeparam>
    public sealed class CachedKernel<TDelegate> : IDisposable
        where TDelegate : Delegate
    {
        #region Fields

        private readonly Accelerator accelerator;
        private readonly TDelegate action;
        private readonly IKernelCache cache;
        private readonly KernelSpecialization? specialization;
        private TDelegate? cachedKernel;
        private bool disposed;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the <see cref="CachedKernel{TDelegate}"/> class.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="action">The kernel action.</param>
        /// <param name="cache">The kernel cache.</param>
        /// <param name="specialization">Optional kernel specialization.</param>
        internal CachedKernel(
            Accelerator accelerator,
            TDelegate action,
            IKernelCache cache,
            KernelSpecialization? specialization = null)
        {
            this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            this.action = action ?? throw new ArgumentNullException(nameof(action));
            this.cache = cache ?? throw new ArgumentNullException(nameof(cache));
            this.specialization = specialization;

            // Create cache key and version
            CacheKey = CreateCacheKey();
            Version = CreateVersion();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the underlying kernel, loading it if necessary.
        /// </summary>
        public TDelegate Kernel
        {
            get
            {
                if (disposed)
                    throw new ObjectDisposedException(nameof(CachedKernel<TDelegate>));

                return cachedKernel != null ? cachedKernel : LoadKernel();
            }
        }

        /// <summary>
        /// Gets the cache key for this kernel.
        /// </summary>
        public string CacheKey { get; }

        /// <summary>
        /// Gets the version string for this kernel.
        /// </summary>
        public string Version { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Invokes the kernel with the specified parameters.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="extent">The kernel extent.</param>
        /// <param name="args">The kernel arguments.</param>
        public void Invoke(AcceleratorStream stream, KernelConfig extent, params object[] args)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(CachedKernel<TDelegate>));

            var kernel = Kernel;
            var invokeMethod = typeof(TDelegate).GetMethod("Invoke");
            
            var parameters = new object[args.Length + 2];
            parameters[0] = stream;
            parameters[1] = extent;
            Array.Copy(args, 0, parameters, 2, args.Length);
            
            invokeMethod?.Invoke(kernel, parameters);
        }

        /// <summary>
        /// Gets information about this cached kernel.
        /// </summary>
        /// <returns>Kernel information including cache status.</returns>
        public CachedKernelInfo GetInfo()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(CachedKernel<TDelegate>));

            var isInCache = cache.ContainsKey(CacheKey);
            var kernelLoaded = cachedKernel != null;
            
            return new CachedKernelInfo(
                CacheKey,
                Version,
                isInCache,
                kernelLoaded,
                action.Method.Name,
                accelerator.AcceleratorType.ToString());
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Loads the kernel from cache or compiles it if not cached.
        /// </summary>
        /// <returns>The loaded kernel.</returns>
        private TDelegate LoadKernel()
        {
            // Try to get from cache first
            if (cache.TryGet(CacheKey, Version, out var cacheEntry))
            {
                if (cacheEntry!.Kernel is TDelegate kernel)
                {
                    cachedKernel = kernel;
                    return kernel;
                }
            }

            // Not in cache or invalid - compile new kernel
            // For simplicity, create a basic kernel loading approach
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Use reflection to call LoadKernel method from KernelLoaders
                var loadKernelMethod = typeof(KernelLoaders).GetMethods()
                    .FirstOrDefault(m => m.Name == "LoadKernel" && m.GetParameters().Length == 2 && m.IsGenericMethodDefinition);
                
                if (loadKernelMethod != null && action.Method.GetParameters().Length > 0)
                {
                    var parameterType = action.Method.GetParameters().First().ParameterType;
                    var genericMethod = loadKernelMethod.MakeGenericMethod(parameterType);
                    var newKernel = (TDelegate)genericMethod.Invoke(null, [accelerator, action])!;
                
                    // Store in cache
                    var kernelMetadata = new Dictionary<string, object>
                    {
                        ["MethodName"] = action.Method.Name,
                        ["AcceleratorType"] = accelerator.AcceleratorType.ToString(),
                        ["CompilationTime"] = DateTime.UtcNow
                    };

                    cache.Put(CacheKey, newKernel, Version, kernelMetadata);
                    
                    cachedKernel = newKernel;
                    return newKernel;
                }
            }
            catch (Exception)
            {
                // Fall through to error
            }
#pragma warning restore CA1031 // Do not catch general exception types
            
            throw new InvalidOperationException($"Unable to load kernel for action {action.Method.Name}");
        }

        /// <summary>
        /// Creates a unique cache key for this kernel.
        /// </summary>
        /// <returns>The cache key.</returns>
        private string CreateCacheKey()
        {
            var methodInfo = action.Method;
            var parameters = methodInfo.GetParameters();
            
            var keyBuilder = new System.Text.StringBuilder();
            keyBuilder.Append(methodInfo.Name);
            keyBuilder.Append('|');
            keyBuilder.Append(accelerator.AcceleratorType);
            keyBuilder.Append('|');
            keyBuilder.Append(accelerator.Device.Name);
            keyBuilder.Append('|');
            
            foreach (var param in parameters)
            {
                keyBuilder.Append(param.ParameterType.FullName);
                keyBuilder.Append(',');
            }

            if (specialization != null)
            {
                keyBuilder.Append('|');
                keyBuilder.Append(specialization.GetHashCode());
            }

            return keyBuilder.ToString();
        }

        /// <summary>
        /// Creates a version string for this kernel.
        /// </summary>
        /// <returns>The version string.</returns>
        private string CreateVersion()
        {
            var assemblyVersion = Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "unknown";
            var targetFramework = Environment.Version.ToString();
            
            return $"{assemblyVersion}_{targetFramework}_{accelerator.AcceleratorType}";
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Releases all resources used by the <see cref="CachedKernel{TDelegate}"/>.
        /// </summary>
        public void Dispose()
        {
            if (!disposed)
            {
                // Kernel delegates don't need explicit disposal
                cachedKernel = null;
                disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// Information about a cached kernel.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the <see cref="CachedKernelInfo"/> class.
    /// </remarks>
    /// <param name="cacheKey">The cache key.</param>
    /// <param name="version">The version string.</param>
    /// <param name="isInCache">Whether the kernel is cached.</param>
    /// <param name="isLoaded">Whether the kernel is loaded in memory.</param>
    /// <param name="methodName">The kernel method name.</param>
    /// <param name="acceleratorType">The accelerator type.</param>
    public sealed class CachedKernelInfo(
        string cacheKey,
        string version,
        bool isInCache,
        bool isLoaded,
        string methodName,
        string acceleratorType)
    {

        /// <summary>
        /// Gets the cache key.
        /// </summary>
        public string CacheKey { get; } = cacheKey;

        /// <summary>
        /// Gets the version string.
        /// </summary>
        public string Version { get; } = version;

        /// <summary>
        /// Gets a value indicating whether the kernel is in cache.
        /// </summary>
        public bool IsInCache { get; } = isInCache;

        /// <summary>
        /// Gets a value indicating whether the kernel is loaded in memory.
        /// </summary>
        public bool IsLoaded { get; } = isLoaded;

        /// <summary>
        /// Gets the kernel method name.
        /// </summary>
        public string MethodName { get; } = methodName;

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public string AcceleratorType { get; } = acceleratorType;
    }
}
