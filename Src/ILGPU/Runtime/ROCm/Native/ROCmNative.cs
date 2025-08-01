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
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.ROCm.Native
{
    /// <summary>
    /// Native AMD ROCm/HIP API bindings for GPU compute acceleration.
    /// </summary>
    /// <remarks>
    /// These bindings interface with AMD's ROCm platform and HIP runtime API
    /// for GPU compute operations on AMD Radeon and Instinct hardware.
    /// 
    /// Requirements:
    /// - AMD ROCm 5.0+ runtime
    /// - AMD GPU with GCN 3.0+ or RDNA architecture
    /// - HIP runtime library
    /// - Linux (primary), Windows (experimental)
    /// </remarks>
    internal static partial class ROCmNative
    {
        #region Constants

#if WINDOWS
        private const string HipLibrary = "hip64_530"; // HIP version 5.30+
        private const string ROCBlasLibrary = "rocblas";
#else
        private const string HipLibrary = "libamdhip64.so.5";
        private const string ROCBlasLibrary = "librocblas.so.3";
#endif

        #endregion

        #region HIP Runtime API

        /// <summary>
        /// Initializes HIP runtime.
        /// </summary>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipInit")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError Initialize(uint flags);

        /// <summary>
        /// Gets the number of HIP devices.
        /// </summary>
        /// <param name="count">Number of devices.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipGetDeviceCount")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError GetDeviceCount(out int count);

        /// <summary>
        /// Gets device properties.
        /// </summary>
        /// <param name="props">Device properties structure.</param>
        /// <param name="device">Device index.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipGetDeviceProperties")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError GetDeviceProperties(out HipDeviceProperties props, int device);

        /// <summary>
        /// Sets the current device.
        /// </summary>
        /// <param name="device">Device index.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipSetDevice")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError SetDevice(int device);

        /// <summary>
        /// Gets the current device.
        /// </summary>
        /// <param name="device">Current device index.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipGetDevice")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError GetDevice(out int device);

        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates device memory.
        /// </summary>
        /// <param name="ptr">Pointer to allocated memory.</param>
        /// <param name="size">Size in bytes.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipMalloc")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError Malloc(out IntPtr ptr, ulong size);

        /// <summary>
        /// Frees device memory.
        /// </summary>
        /// <param name="ptr">Pointer to free.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipFree")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError Free(IntPtr ptr);

        /// <summary>
        /// Copies memory between host and device.
        /// </summary>
        /// <param name="dst">Destination pointer.</param>
        /// <param name="src">Source pointer.</param>
        /// <param name="sizeBytes">Size in bytes.</param>
        /// <param name="kind">Copy direction.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipMemcpy")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError Memcpy(IntPtr dst, IntPtr src, ulong sizeBytes, HipMemcpyKind kind);

        /// <summary>
        /// Asynchronous memory copy.
        /// </summary>
        /// <param name="dst">Destination pointer.</param>
        /// <param name="src">Source pointer.</param>
        /// <param name="sizeBytes">Size in bytes.</param>
        /// <param name="kind">Copy direction.</param>
        /// <param name="stream">HIP stream.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipMemcpyAsync")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError MemcpyAsync(IntPtr dst, IntPtr src, ulong sizeBytes, HipMemcpyKind kind, IntPtr stream);

        /// <summary>
        /// Sets device memory to a specific value.
        /// </summary>
        /// <param name="devPtr">Device pointer.</param>
        /// <param name="value">Value to set.</param>
        /// <param name="count">Number of bytes.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipMemset")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError Memset(IntPtr devPtr, int value, ulong count);

        #endregion

        #region Stream Management

        /// <summary>
        /// Creates a HIP stream.
        /// </summary>
        /// <param name="stream">Stream handle.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipStreamCreate")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError StreamCreate(out IntPtr stream);

        /// <summary>
        /// Destroys a HIP stream.
        /// </summary>
        /// <param name="stream">Stream handle.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipStreamDestroy")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError StreamDestroy(IntPtr stream);

        /// <summary>
        /// Synchronizes a HIP stream.
        /// </summary>
        /// <param name="stream">Stream handle.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipStreamSynchronize")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError StreamSynchronize(IntPtr stream);

        #endregion

        #region Kernel Execution

        /// <summary>
        /// Launches a kernel.
        /// </summary>
        /// <param name="function">Kernel function.</param>
        /// <param name="gridDimX">Grid dimension X.</param>
        /// <param name="gridDimY">Grid dimension Y.</param>
        /// <param name="gridDimZ">Grid dimension Z.</param>
        /// <param name="blockDimX">Block dimension X.</param>
        /// <param name="blockDimY">Block dimension Y.</param>
        /// <param name="blockDimZ">Block dimension Z.</param>
        /// <param name="sharedMemBytes">Shared memory size.</param>
        /// <param name="stream">HIP stream.</param>
        /// <param name="kernelParams">Kernel parameters.</param>
        /// <param name="extra">Extra parameters.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipLaunchKernel")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError LaunchKernel(
            IntPtr function,
            uint gridDimX, uint gridDimY, uint gridDimZ,
            uint blockDimX, uint blockDimY, uint blockDimZ,
            uint sharedMemBytes, IntPtr stream,
            IntPtr[] kernelParams, IntPtr[] extra);

        #endregion

        #region Module Management

        /// <summary>
        /// Loads a module from binary data.
        /// </summary>
        /// <param name="module">Module handle.</param>
        /// <param name="image">Binary image data.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipModuleLoadData")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError ModuleLoadData(out IntPtr module, byte[] image);

        /// <summary>
        /// Gets a function from a module.
        /// </summary>
        /// <param name="function">Function handle.</param>
        /// <param name="module">Module handle.</param>
        /// <param name="name">Function name.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipModuleGetFunction")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError ModuleGetFunction(out IntPtr function, IntPtr module, [MarshalAs(UnmanagedType.LPWStr)] string name);

        /// <summary>
        /// Unloads a module.
        /// </summary>
        /// <param name="module">Module handle.</param>
        /// <returns>Hip error code.</returns>
        [LibraryImport(HipLibrary, EntryPoint = "hipModuleUnload")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial HipError ModuleUnload(IntPtr module);

        #endregion

        #region ROCm/HIP Detection and Initialization

        /// <summary>
        /// Checks if ROCm/HIP is supported on this system.
        /// </summary>
        /// <returns>True if ROCm is supported; otherwise, false.</returns>
        internal static bool IsROCmSupported()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var result = GetDeviceCount(out int count);
                return result == HipError.Success && count > 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Initializes ROCm runtime.
        /// </summary>
        /// <returns>True if initialization succeeded; otherwise, false.</returns>
        internal static bool InitializeROCm()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var result = Initialize(0);
                if (result != HipError.Success)
                    return false;

                // Verify we have at least one device
                result = GetDeviceCount(out int count);
                return result == HipError.Success && count > 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Gets device count safely.
        /// </summary>
        /// <returns>Number of ROCm devices.</returns>
        internal static int GetDeviceCountSafe()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var result = GetDeviceCount(out int count);
                return result == HipError.Success ? count : 0;
            }
            catch
            {
                return 0;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Gets device properties safely.
        /// </summary>
        /// <param name="deviceIndex">Device index.</param>
        /// <returns>Device properties or null if failed.</returns>
        internal static HipDeviceProperties? GetDevicePropertiesSafe(int deviceIndex)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var result = GetDeviceProperties(out var props, deviceIndex);
                return result == HipError.Success ? props : null;
            }
            catch
            {
                return null;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region ROCBlas Integration

        /// <summary>
        /// Executes matrix multiplication using ROCBlas.
        /// </summary>
        /// <param name="a">Matrix A data pointer.</param>
        /// <param name="b">Matrix B data pointer.</param>
        /// <param name="c">Matrix C data pointer.</param>
        /// <param name="m">Matrix dimension M.</param>
        /// <param name="k">Matrix dimension K.</param>
        /// <param name="n">Matrix dimension N.</param>
        /// <param name="stream">HIP stream.</param>
        internal static unsafe void ExecuteROCBlasMatMul(
            void* a, void* b, void* c,
            int m, int k, int n, IntPtr stream)
        {
            try
            {
                // Try to use ROCBlas for hardware acceleration
                // TODO: ExecuteROCBlasMatMulNative parameters incomplete - requires handle, transpose flags, leading dimensions, alpha/beta
                ExecuteROCBlasMatMulNative(
                    stream, // handle 
                    0, 0,   // transA, transB (no transpose)
                    m, n, k,
                    1.0f,   // alpha
                    (IntPtr)a, k,   // a, lda
                    (IntPtr)b, n,   // b, ldb  
                    0.0f,   // beta
                    (IntPtr)c, n);  // c, ldc
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU implementation if ROCBlas is not available
                ExecuteCPUMatMulFallback(a, b, c, m, k, n);
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU implementation if ROCBlas functions are not found
                ExecuteCPUMatMulFallback(a, b, c, m, k, n);
            }
        }

        /// <summary>
        /// Native ROCBlas matrix multiplication implementation.
        /// </summary>
        [LibraryImport(ROCBlasLibrary, EntryPoint = "rocblas_sgemm")]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        private static partial HipError ExecuteROCBlasMatMulNative(
            IntPtr handle, int transA, int transB,
            int m, int n, int k,
            float alpha, IntPtr a, int lda,
            IntPtr b, int ldb,
            float beta, IntPtr c, int ldc);

        /// <summary>
        /// CPU fallback for matrix multiplication.
        /// </summary>
        private static unsafe void ExecuteCPUMatMulFallback(
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            var aPtr = (float*)a;
            var bPtr = (float*)b;
            var cPtr = (float*)c;

            // Basic matrix multiplication fallback
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0.0f;
                    for (int ki = 0; ki < k; ki++)
                        sum += aPtr[i * k + ki] * bPtr[ki * n + j];
                    cPtr[i * n + j] = sum;
                }
            }
        }

        #endregion
    }

    #region Enums and Structures

    /// <summary>
    /// HIP error codes.
    /// </summary>
    internal enum HipError : int
    {
        Success = 0,
        ErrorInvalidValue = 1,
        ErrorOutOfMemory = 2,
        ErrorNotInitialized = 3,
        ErrorDeinitialized = 4,
        ErrorProfilerDisabled = 5,
        ErrorProfilerNotInitialized = 6,
        ErrorProfilerAlreadyStarted = 7,
        ErrorProfilerAlreadyStopped = 8,
        ErrorNoDevice = 100,
        ErrorInvalidDevice = 101,
        ErrorInvalidImage = 200,
        ErrorInvalidContext = 201,
        ErrorContextAlreadyCurrent = 202,
        ErrorMapFailed = 205,
        ErrorUnmapFailed = 206,
        ErrorArrayIsMapped = 207,
        ErrorAlreadyMapped = 208,
        ErrorNoBinaryForGpu = 209,
        ErrorAlreadyAcquired = 210,
        ErrorNotMapped = 211,
        ErrorNotMappedAsArray = 212,
        ErrorNotMappedAsPointer = 213,
        ErrorEccUncorrectable = 214,
        ErrorUnsupportedLimit = 215,
        ErrorContextAlreadyInUse = 216,
        ErrorPeerAccessUnsupported = 217,
        ErrorInvalidPtx = 218,
        ErrorInvalidGraphicsContext = 219,
        ErrorNvlinkUncorrectable = 220,
        ErrorJitCompilerNotFound = 221,
        ErrorInvalidSource = 300,
        ErrorFileNotFound = 301,
        ErrorSharedObjectSymbolNotFound = 302,
        ErrorSharedObjectInitFailed = 303,
        ErrorOperatingSystem = 304,
        ErrorInvalidHandle = 400,
        ErrorNotFound = 500,
        ErrorNotReady = 600,
        ErrorIllegalAddress = 700,
        ErrorLaunchOutOfResources = 701,
        ErrorLaunchTimeOut = 702,
        ErrorPeerAccessAlreadyEnabled = 704,
        ErrorPeerAccessNotEnabled = 705,
        ErrorSetOnActiveProcess = 708,
        ErrorAssert = 710,
        ErrorHostMemoryAlreadyRegistered = 712,
        ErrorHostMemoryNotRegistered = 713,
        ErrorLaunchFailure = 719,
        ErrorCooperativeLaunchTooLarge = 720,
        ErrorNotSupported = 801,
        ErrorUnknown = 999
    }

    /// <summary>
    /// HIP memory copy kinds.
    /// </summary>
    internal enum HipMemcpyKind : int
    {
        HostToHost = 0,
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3,
        Default = 4
    }

    /// <summary>
    /// HIP device properties.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct HipDeviceProperties
    {
        public fixed byte Name[256];
        
        public ulong TotalGlobalMem;
        public ulong SharedMemPerBlock;
        public int RegsPerBlock;
        public int WarpSize;
        public ulong MemPitch;
        public int MaxThreadsPerBlock;
        public fixed int MaxThreadsDim[3];
        public fixed int MaxGridSize[3];
        public int ClockRate;
        public ulong TotalConstMem;
        public int Major;
        public int Minor;
        public ulong TextureAlignment;
        public ulong TexturePitchAlignment;
        public int DeviceOverlap;
        public int MultiProcessorCount;
        public int KernelExecTimeoutEnabled;
        public int Integrated;
        public int CanMapHostMemory;
        public int ComputeMode;
        public int MaxTexture1D;
        public fixed int MaxTexture2D[2];
        public fixed int MaxTexture3D[3];
        public int ConcurrentKernels;
        public int EccEnabled;
        public int PciBusId;
        public int PciDeviceId;
        public int PciDomainId;
        public int TccDriver;
        public int AsyncEngineCount;
        public int UnifiedAddressing;
        public int MemoryClockRate;
        public int MemoryBusWidth;
        public int L2CacheSize;
        public int MaxThreadsPerMultiProcessor;
        public int StreamPrioritiesSupported;
        public int GlobalL1CacheSupported;
        public int LocalL1CacheSupported;
        public ulong SharedMemPerMultiprocessor;
        public int RegsPerMultiprocessor;
        public int ManagedMemory;
        public int IsMultiGpuBoard;
        public int MultiGpuBoardGroupId;
        public int HostNativeAtomicSupported;
        public int SingleToDoublePrecisionPerfRatio;
        public int PageableMemoryAccess;
        public int ConcurrentManagedAccess;
        public int ComputePreemptionSupported;
        public int CanUseHostPointerForRegisteredMem;
        public int CooperativeLaunch;
        public int CooperativeMultiDeviceLaunch;
        public ulong SharedMemPerBlockOptin;
        public int PageableMemoryAccessUsesHostPageTables;
        public int DirectManagedMemAccessFromHost;
    }

    #endregion
}
