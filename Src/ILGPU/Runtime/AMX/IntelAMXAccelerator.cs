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

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime.AMX.Native;
using ILGPU.Runtime.CPU;
using ILGPU.Util;
using System;
using System.Collections.Immutable;
using System.Reflection;
using System.Threading.Tasks;

namespace ILGPU.Runtime.AMX
{
    /// <summary>
    /// Intel AMX (Advanced Matrix Extensions) accelerator for AI workload acceleration.
    /// </summary>
    public sealed class IntelAMXAccelerator : KernelAccelerator<AMXCompiledKernel, AMXKernel>
    {
        #region Instance

        /// <summary>
        /// The associated Intel AMX device.
        /// </summary>
        public new IntelAMXDevice Device { get; }

        /// <summary>
        /// Gets whether this accelerator supports BF16 operations.
        /// </summary>
        public bool SupportsBF16 => Device.SupportsBF16;

        /// <summary>
        /// Gets whether this accelerator supports INT8 operations.
        /// </summary>
        public bool SupportsINT8 => Device.SupportsINT8;

        /// <summary>
        /// Gets whether this accelerator supports mixed precision operations.
        /// </summary>
        public bool SupportsMixedPrecision => Device.SupportsMixedPrecision;

        /// <summary>
        /// Gets the maximum tile size supported.
        /// </summary>
        public int MaxTileSize => Device.MaxTileSize;

        /// <summary>
        /// Gets the number of available tiles.
        /// </summary>
        public int TileCount => Device.TileCount;

        /// <summary>
        /// Initializes a new Intel AMX accelerator.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The Intel AMX device.</param>
        internal IntelAMXAccelerator(Context context, IntelAMXDevice device)
            : base(context, device)
        {
            Device = device;

            try
            {
                // Verify AMX support
                if (!AMXNative.IsAMXSupported())
                    throw new NotSupportedException("Intel AMX not supported on this processor");

                // Initialize AMX subsystem
                Init();
            }
            catch (Exception ex)
            {
                throw new AMXException("Failed to initialize Intel AMX accelerator", ex);
            }
        }

        /// <summary>
        /// Initializes the accelerator properties.
        /// </summary>
        private void Init()
        {
            DefaultStream = CreateStreamInternal();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public new AcceleratorType AcceleratorType => AcceleratorType.CPU;

        /// <summary>
        /// Gets the accelerator name.
        /// </summary>
        public new string Name => $"Intel AMX ({Device.ProcessorName})";

        /// <summary>
        /// Gets the memory size in bytes.
        /// </summary>
        public new long MemorySize => Device.MemorySize;

        /// <summary>
        /// Gets the maximum grid size.
        /// </summary>
        public new Index3D MaxGridSize => Device.MaxGridSize;

        /// <summary>
        /// Gets the maximum group size.
        /// </summary>
        public new Index3D MaxGroupSize => Device.MaxGroupSize;

        /// <summary>
        /// Gets the warp size.
        /// </summary>
        public new int WarpSize => 1; // AMX operates on single thread

        /// <summary>
        /// Gets the number of multiprocessors.
        /// </summary>
        public new int NumMultiprocessors => Device.NumCores;

        #endregion

        #region Abstract Method Implementations

        /// <summary>
        /// Called when the accelerator is bound to the current thread.
        /// </summary>
        protected override void OnBind()
        {
            // AMX-specific binding logic if needed
        }

        /// <summary>
        /// Called when the accelerator is unbound from the current thread.
        /// </summary>
        protected override void OnUnbind()
        {
            // AMX-specific unbinding logic if needed
        }


        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates a chunk of memory.
        /// </summary>
        /// <param name="length">The length in elements.</param>
        /// <param name="elementSize">The element size in bytes.</param>
        /// <returns>The allocated memory buffer.</returns>
        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new AMXMemoryBuffer(this, length, elementSize);
        }

        /// <summary>
        /// Creates a page-lock scope for the given array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="pinned">The pinned array.</param>
        /// <param name="numElements">The number of elements.</param>
        /// <returns>The page-lock scope.</returns>
        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(
            IntPtr pinned,
            long numElements) =>
            new AMXPageLockScope<T>(this, pinned, numElements);

        #endregion

        #region Kernel Management

        /// <summary>
        /// Generates a kernel launcher method for the given compiled kernel.
        /// </summary>
        /// <param name="kernel">The compiled kernel.</param>
        /// <param name="customGroupSize">The custom group size.</param>
        /// <returns>The kernel launcher method.</returns>
        protected override MethodInfo GenerateKernelLauncherMethod(
            AMXCompiledKernel kernel,
            int customGroupSize)
        {
            // For AMX accelerator, use default launcher generation
            // In a real implementation, this would generate optimized AMX-specific launchers
            return typeof(IntelAMXAccelerator).GetMethod(nameof(DefaultLauncher), 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                ?? throw new InvalidOperationException("Default launcher method not found");
        }

        /// <summary>
        /// Default kernel launcher for AMX kernels.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.DoesNotReturn]
        private static void DefaultLauncher()
        {
            // Default launcher implementation - this is a placeholder for future AMX kernel execution
            throw new NotImplementedException("AMX kernel launcher not implemented");
        }

        /// <summary>
        /// Creates a kernel from the given compiled kernel.
        /// </summary>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <returns>The created kernel.</returns>
        protected override AMXKernel CreateKernel(AMXCompiledKernel compiledKernel) =>
            new AMXKernel(this, compiledKernel);

        /// <summary>
        /// Creates a kernel from the given compiled kernel with launcher method.
        /// </summary>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <param name="launcher">The launcher method.</param>
        /// <returns>The created kernel.</returns>
        protected override AMXKernel CreateKernel(AMXCompiledKernel compiledKernel, MethodInfo launcher) =>
            new AMXKernel(this, compiledKernel);

        #endregion

        #region Stream Management

        /// <summary>
        /// Creates a new AMX stream.
        /// </summary>
        /// <returns>The created stream.</returns>
        protected override AcceleratorStream CreateStreamInternal() =>
            new AMXStream(this);

        /// <summary>
        /// Synchronizes all pending operations.
        /// </summary>
        protected override void SynchronizeInternal()
        {
            // AMX operations are synchronous on CPU
        }

        #endregion

        #region Peer Access

        /// <summary>
        /// Checks whether this accelerator can access the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        /// <returns>True if peer access is possible.</returns>
        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            // AMX can access CPU memory directly
            return otherAccelerator is IntelAMXAccelerator || otherAccelerator is CPUAccelerator;
        }

        /// <summary>
        /// Enables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // No explicit peer access needed for CPU-based accelerators
        }

        /// <summary>
        /// Disables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // No explicit peer access to disable for CPU-based accelerators
        }

        #endregion

        #region Kernel Estimation

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor.
        /// </summary>
        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            // For AMX, each core can handle one group at a time
            return 1;
        }

        /// <summary>
        /// Estimates the group size for the given kernel.
        /// </summary>
        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = NumMultiprocessors;
            return Math.Min(maxGroupSize, NumMultiprocessors);
        }

        /// <summary>
        /// Estimates the group size for the given kernel.
        /// </summary>
        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            return EstimateGroupSizeInternal(
                kernel,
                _ => dynamicSharedMemorySizeInBytes,
                maxGroupSize,
                out minGridSize);
        }

        #endregion

        #region Extensions

        /// <summary>
        /// Creates an accelerator extension.
        /// </summary>
        /// <typeparam name="TExtension">The extension type.</typeparam>
        /// <typeparam name="TExtensionProvider">The extension provider type.</typeparam>
        /// <param name="provider">The provider instance.</param>
        /// <returns>The created extension.</returns>
        public override TExtension CreateExtension<TExtension, TExtensionProvider>(
            TExtensionProvider provider) =>
            throw new NotSupportedException($"Extension {typeof(TExtension)} is not supported by AMX accelerator");

        #endregion

        #region AMX Operations

        /// <summary>
        /// Executes matrix multiplication using Intel AMX acceleration.
        /// </summary>
        /// <param name="a">Matrix A data pointer.</param>
        /// <param name="b">Matrix B data pointer.</param>
        /// <param name="c">Result matrix C data pointer.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="dataType">Data type for computation.</param>
        /// <param name="stream">AMX stream.</param>
        internal unsafe void ExecuteMatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            AMXDataType dataType,
            AMXStream? stream = null)
        {
            try
            {
                // Execute using Intel AMX hardware acceleration
                AMXNative.ExecuteAMXMatMul(
                    a.ToPointer(), b.ToPointer(), c.ToPointer(),
                    m, k, n, dataType);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel AMX runtime libraries not found. Install Intel AMX runtime for optimal performance.");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel AMX functions not found. Check AMX runtime installation.");
            }
        }

        /// <summary>
        /// Executes BF16 matrix multiplication with AMX acceleration.
        /// </summary>
        /// <param name="a">Matrix A (BF16).</param>
        /// <param name="b">Matrix B (BF16).</param>
        /// <param name="c">Result matrix C (FP32).</param>
        /// <param name="m">Matrix dimensions.</param>
        /// <param name="k">Matrix dimensions.</param>
        /// <param name="n">Matrix dimensions.</param>
        /// <param name="stream">AMX stream.</param>
        public void ExecuteBF16MatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            AMXStream? stream = null)
        {
            ExecuteMatMul(a, b, c, m, k, n, AMXDataType.BF16, stream);
        }

        /// <summary>
        /// Executes INT8 matrix multiplication with AMX acceleration.
        /// </summary>
        /// <param name="a">Matrix A (INT8).</param>
        /// <param name="b">Matrix B (INT8).</param>
        /// <param name="c">Result matrix C (INT32).</param>
        /// <param name="m">Matrix dimensions.</param>
        /// <param name="k">Matrix dimensions.</param>
        /// <param name="n">Matrix dimensions.</param>
        /// <param name="stream">AMX stream.</param>
        public void ExecuteINT8MatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            AMXStream? stream = null)
        {
            ExecuteMatMul(a, b, c, m, k, n, AMXDataType.INT8, stream);
        }

        /// <summary>
        /// Executes AI inference acceleration using AMX tiles.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="weights">Weight tensor data.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="inputSize">Input feature size.</param>
        /// <param name="outputSize">Output feature size.</param>
        /// <param name="dataType">Data type for computation.</param>
        /// <param name="stream">AMX stream.</param>
        internal async Task ExecuteAIInferenceAsync(
            IntPtr input, IntPtr weights, IntPtr output,
            int batchSize, int inputSize, int outputSize,
            AMXDataType dataType,
            AMXStream? stream = null)
        {
            await Task.Run(() =>
            {
                try
                {
                    // Execute matrix multiplication for neural network layer
                    ExecuteMatMul(input, weights, output, 
                        batchSize, inputSize, outputSize, dataType, stream);
                }
                catch (Exception ex)
                {
                    throw new AMXException("Failed to execute AI inference with AMX", ex);
                }
            });
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this Intel AMX accelerator.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                try
                {
                    // Release any AMX-specific resources
                    AMXNative.ReleaseTiles();
                }
                catch
                {
                    // Ignore errors during disposal
                }
            }
        }

        #endregion

        #region AMX Information

        /// <summary>
        /// Prints detailed AMX capability information.
        /// </summary>
        public void PrintAMXInformation()
        {
            Console.WriteLine($"Intel AMX Accelerator Information:");
            Console.WriteLine($"  Name: {Name}");
            Console.WriteLine($"  Processor: {Device.ProcessorName}");
            Console.WriteLine($"  AMX Support: {AMXNative.IsAMXSupported()}");
            Console.WriteLine($"  BF16 Support: {SupportsBF16}");
            Console.WriteLine($"  INT8 Support: {SupportsINT8}");
            Console.WriteLine($"  Mixed Precision: {SupportsMixedPrecision}");
            Console.WriteLine($"  Max Tile Size: {MaxTileSize}x{MaxTileSize}");
            Console.WriteLine($"  Tile Count: {TileCount}");
            Console.WriteLine($"  Memory Size: {MemorySize / (1024 * 1024)} MB");
            Console.WriteLine($"  CPU Cores: {NumMultiprocessors}");
            Console.WriteLine($"  Cache Size: {Device.CacheSize / (1024 * 1024)} MB");
        }

        #endregion
    }
}