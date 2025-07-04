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

namespace ILGPU.Runtime.OneAPI.Native
{
    /// <summary>
    /// Native Intel OneAPI/SYCL bindings for Intel GPU acceleration.
    /// </summary>
    /// <remarks>
    /// These bindings interface with Intel's OneAPI toolkit and SYCL runtime
    /// for GPU compute operations on Intel Arc, Iris Xe, and data center GPUs.
    /// 
    /// Requirements:
    /// - Intel OneAPI Base Toolkit 2024.0+
    /// - Intel GPU drivers (100.x.x.x series)
    /// - Level Zero or OpenCL runtime
    /// - SYCL runtime library
    /// - Intel Arc, Iris Xe, or Xe-HP/HPC GPU
    /// </remarks>
    internal static partial class SYCLNative
    {
        #region Constants

#if WINDOWS
        private const string SYCLLibrary = "sycl7"; // Intel OneAPI SYCL runtime
        private const string LevelZeroLibrary = "ze_loader"; // Intel Level Zero
        private const string OpenCLLibrary = "OpenCL"; // OpenCL fallback
        private const string MKLLibrary = "mkl_sycl"; // Intel MKL SYCL
        private const string DPCPPLibrary = "sycl7"; // Intel DPC++ compiler runtime
#else
        private const string SYCLLibrary = "libsycl.so.7";
        private const string LevelZeroLibrary = "libze_loader.so.1";
        private const string OpenCLLibrary = "libOpenCL.so.1";
        private const string MKLLibrary = "libmkl_sycl.so.2";
        private const string DPCPPLibrary = "libsycl.so.7";
#endif

        #endregion

        #region SYCL Platform and Device Management

        /// <summary>
        /// Gets SYCL platforms.
        /// </summary>
        /// <param name="platformCount">Number of platforms.</param>
        /// <param name="platforms">Platform handles.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_get_platforms", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult GetPlatforms(out uint platformCount, IntPtr[] platforms);

        /// <summary>
        /// Gets SYCL devices for a platform.
        /// </summary>
        /// <param name="platform">Platform handle.</param>
        /// <param name="deviceType">Device type filter.</param>
        /// <param name="deviceCount">Number of devices.</param>
        /// <param name="devices">Device handles.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_get_devices", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult GetDevices(
            IntPtr platform, 
            SYCLDeviceType deviceType, 
            out uint deviceCount, 
            IntPtr[] devices);

        /// <summary>
        /// Gets device information.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="infoType">Information type.</param>
        /// <param name="valueSize">Value buffer size.</param>
        /// <param name="value">Value buffer.</param>
        /// <param name="retSize">Returned size.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_get_device_info", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult GetDeviceInfo(
            IntPtr device,
            SYCLDeviceInfo infoType,
            UIntPtr valueSize,
            IntPtr value,
            out UIntPtr retSize);

        /// <summary>
        /// Creates a SYCL context.
        /// </summary>
        /// <param name="deviceCount">Number of devices.</param>
        /// <param name="devices">Device handles.</param>
        /// <param name="context">Created context handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_create_context", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult CreateContext(
            uint deviceCount,
            IntPtr[] devices,
            out IntPtr context);

        /// <summary>
        /// Releases a SYCL context.
        /// </summary>
        /// <param name="context">Context handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_release_context", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult ReleaseContext(IntPtr context);

        #endregion

        #region SYCL Queue Management

        /// <summary>
        /// Creates a SYCL queue.
        /// </summary>
        /// <param name="context">Context handle.</param>
        /// <param name="device">Device handle.</param>
        /// <param name="properties">Queue properties.</param>
        /// <param name="queue">Created queue handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_create_queue", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult CreateQueue(
            IntPtr context,
            IntPtr device,
            SYCLQueueProperties properties,
            out IntPtr queue);

        /// <summary>
        /// Releases a SYCL queue.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_release_queue", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult ReleaseQueue(IntPtr queue);

        /// <summary>
        /// Waits for queue to complete.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_queue_wait", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult QueueWait(IntPtr queue);

        #endregion

        #region SYCL Memory Management

        /// <summary>
        /// Allocates device memory.
        /// </summary>
        /// <param name="size">Memory size in bytes.</param>
        /// <param name="device">Device handle.</param>
        /// <param name="context">Context handle.</param>
        /// <returns>Device memory pointer.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_malloc_device", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr MallocDevice(UIntPtr size, IntPtr device, IntPtr context);

        /// <summary>
        /// Allocates shared memory.
        /// </summary>
        /// <param name="size">Memory size in bytes.</param>
        /// <param name="device">Device handle.</param>
        /// <param name="context">Context handle.</param>
        /// <returns>Shared memory pointer.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_malloc_shared", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr MallocShared(UIntPtr size, IntPtr device, IntPtr context);

        /// <summary>
        /// Allocates host memory.
        /// </summary>
        /// <param name="size">Memory size in bytes.</param>
        /// <param name="context">Context handle.</param>
        /// <returns>Host memory pointer.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_malloc_host", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr MallocHost(UIntPtr size, IntPtr context);

        /// <summary>
        /// Frees SYCL memory.
        /// </summary>
        /// <param name="ptr">Memory pointer.</param>
        /// <param name="context">Context handle.</param>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_free", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Free(IntPtr ptr, IntPtr context);

        /// <summary>
        /// Copies memory.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <param name="dest">Destination pointer.</param>
        /// <param name="src">Source pointer.</param>
        /// <param name="size">Size in bytes.</param>
        /// <returns>Event handle.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_memcpy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Memcpy(IntPtr queue, IntPtr dest, IntPtr src, UIntPtr size);

        /// <summary>
        /// Sets memory to a value.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <param name="ptr">Memory pointer.</param>
        /// <param name="value">Value to set.</param>
        /// <param name="size">Size in bytes.</param>
        /// <returns>Event handle.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_memset", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Memset(IntPtr queue, IntPtr ptr, int value, UIntPtr size);

        #endregion

        #region SYCL Kernel Execution

        /// <summary>
        /// Creates a kernel from SPIR-V binary.
        /// </summary>
        /// <param name="context">Context handle.</param>
        /// <param name="devices">Device handles.</param>
        /// <param name="deviceCount">Number of devices.</param>
        /// <param name="spirvBinary">SPIR-V binary data.</param>
        /// <param name="binarySize">Binary size.</param>
        /// <param name="kernelName">Kernel entry point name.</param>
        /// <param name="kernel">Created kernel handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_create_kernel_from_spirv", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult CreateKernelFromSPIRV(
            IntPtr context,
            IntPtr[] devices,
            uint deviceCount,
            byte[] spirvBinary,
            UIntPtr binarySize,
            [MarshalAs(UnmanagedType.LPStr)] string kernelName,
            out IntPtr kernel);

        /// <summary>
        /// Releases a kernel.
        /// </summary>
        /// <param name="kernel">Kernel handle.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_release_kernel", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult ReleaseKernel(IntPtr kernel);

        /// <summary>
        /// Sets kernel argument.
        /// </summary>
        /// <param name="kernel">Kernel handle.</param>
        /// <param name="argIndex">Argument index.</param>
        /// <param name="argSize">Argument size.</param>
        /// <param name="argValue">Argument value.</param>
        /// <returns>SYCL result code.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_set_kernel_arg", CallingConvention = CallingConvention.Cdecl)]
        internal static extern SYCLResult SetKernelArg(
            IntPtr kernel,
            uint argIndex,
            UIntPtr argSize,
            IntPtr argValue);

        /// <summary>
        /// Submits kernel for execution.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <param name="kernel">Kernel handle.</param>
        /// <param name="workDim">Work dimensions.</param>
        /// <param name="globalWorkSize">Global work size.</param>
        /// <param name="localWorkSize">Local work size.</param>
        /// <param name="eventWaitList">Wait events.</param>
        /// <param name="numEvents">Number of wait events.</param>
        /// <returns>Event handle.</returns>
        [DllImport(SYCLLibrary, EntryPoint = "sycl_submit_kernel", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SubmitKernel(
            IntPtr queue,
            IntPtr kernel,
            uint workDim,
            UIntPtr[] globalWorkSize,
            UIntPtr[] localWorkSize,
            IntPtr[] eventWaitList,
            uint numEvents);

        #endregion

        #region Intel GPU Architecture Detection

        /// <summary>
        /// Checks if Intel OneAPI/SYCL is supported on this system.
        /// </summary>
        /// <returns>True if SYCL is supported; otherwise, false.</returns>
        internal static bool IsSYCLSupported()
        {
            try
            {
                // Try to get platforms to verify SYCL is available
                var result = GetPlatforms(out uint platformCount, new IntPtr[0]);
                return result == SYCLResult.Success;
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
        }

        /// <summary>
        /// Gets Intel GPU devices using SYCL.
        /// </summary>
        /// <returns>Array of Intel GPU device handles.</returns>
        internal static IntPtr[] GetIntelGPUDevices()
        {
            try
            {
                // Get platforms
                var result = GetPlatforms(out uint platformCount, new IntPtr[0]);
                if (result != SYCLResult.Success || platformCount == 0)
                    return Array.Empty<IntPtr>();

                var platforms = new IntPtr[platformCount];
                result = GetPlatforms(out _, platforms);
                if (result != SYCLResult.Success)
                    return Array.Empty<IntPtr>();

                var allDevices = new System.Collections.Generic.List<IntPtr>();

                foreach (var platform in platforms)
                {
                    // Get GPU devices for this platform
                    result = GetDevices(platform, SYCLDeviceType.GPU, out uint deviceCount, new IntPtr[0]);
                    if (result == SYCLResult.Success && deviceCount > 0)
                    {
                        var devices = new IntPtr[deviceCount];
                        result = GetDevices(platform, SYCLDeviceType.GPU, out _, devices);
                        if (result == SYCLResult.Success)
                        {
                            // Filter for Intel devices
                            foreach (var device in devices)
                            {
                                if (IsIntelDevice(device))
                                    allDevices.Add(device);
                            }
                        }
                    }
                }

                return allDevices.ToArray();
            }
            catch
            {
                return Array.Empty<IntPtr>();
            }
        }

        /// <summary>
        /// Checks if a device is an Intel device.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <returns>True if Intel device; otherwise, false.</returns>
        private static bool IsIntelDevice(IntPtr device)
        {
            try
            {
                var vendorName = GetDeviceInfoString(device, SYCLDeviceInfo.Vendor);
                return vendorName.Contains("Intel", StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets device information as string.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="infoType">Information type.</param>
        /// <returns>Device information string.</returns>
        internal static string GetDeviceInfoString(IntPtr device, SYCLDeviceInfo infoType)
        {
            try
            {
                // Get required size
                var result = GetDeviceInfo(device, infoType, UIntPtr.Zero, IntPtr.Zero, out var retSize);
                if (result != SYCLResult.Success)
                    return string.Empty;

                // Get actual value
                var buffer = new byte[retSize.ToUInt32()];
                var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    result = GetDeviceInfo(device, infoType, retSize, handle.AddrOfPinnedObject(), out _);
                    if (result != SYCLResult.Success)
                        return string.Empty;

                    return System.Text.Encoding.UTF8.GetString(buffer, 0, buffer.Length - 1);
                }
                finally
                {
                    handle.Free();
                }
            }
            catch
            {
                return string.Empty;
            }
        }

        /// <summary>
        /// Gets device information as uint32.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="infoType">Information type.</param>
        /// <returns>Device information value.</returns>
        internal static uint GetDeviceInfoUInt32(IntPtr device, SYCLDeviceInfo infoType)
        {
            try
            {
                var buffer = new byte[4];
                var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    var result = GetDeviceInfo(device, infoType, new UIntPtr(4), handle.AddrOfPinnedObject(), out _);
                    if (result != SYCLResult.Success)
                        return 0;

                    return BitConverter.ToUInt32(buffer, 0);
                }
                finally
                {
                    handle.Free();
                }
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// Gets device information as uint64.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="infoType">Information type.</param>
        /// <returns>Device information value.</returns>
        internal static ulong GetDeviceInfoUInt64(IntPtr device, SYCLDeviceInfo infoType)
        {
            try
            {
                var buffer = new byte[8];
                var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    var result = GetDeviceInfo(device, infoType, new UIntPtr(8), handle.AddrOfPinnedObject(), out _);
                    if (result != SYCLResult.Success)
                        return 0;

                    return BitConverter.ToUInt64(buffer, 0);
                }
                finally
                {
                    handle.Free();
                }
            }
            catch
            {
                return 0;
            }
        }

        #endregion

        #region Intel MKL Integration

        /// <summary>
        /// Executes matrix multiplication using Intel MKL SYCL.
        /// </summary>
        /// <param name="queue">SYCL queue handle.</param>
        /// <param name="a">Matrix A data pointer.</param>
        /// <param name="b">Matrix B data pointer.</param>
        /// <param name="c">Matrix C data pointer.</param>
        /// <param name="m">Matrix dimension M.</param>
        /// <param name="k">Matrix dimension K.</param>
        /// <param name="n">Matrix dimension N.</param>
        internal static unsafe void ExecuteMKLSYCLMatMul(
            IntPtr queue,
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            try
            {
                // Try to use Intel MKL SYCL for hardware acceleration
                ExecuteMKLSYCLGEMM(queue, a, b, c, m, k, n);
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU implementation if MKL SYCL is not available
                ExecuteCPUMatMulFallback(a, b, c, m, k, n);
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU implementation if MKL SYCL functions are not found
                ExecuteCPUMatMulFallback(a, b, c, m, k, n);
            }
        }

        /// <summary>
        /// Native Intel MKL SYCL GEMM implementation.
        /// </summary>
        [DllImport(MKLLibrary, EntryPoint = "oneapi_mkl_blas_sgemm", CallingConvention = CallingConvention.Cdecl)]
        private static extern SYCLResult ExecuteMKLSYCLGEMM(
            IntPtr queue,
            int layout, int transA, int transB,
            int m, int n, int k,
            float alpha, IntPtr a, int lda,
            IntPtr b, int ldb,
            float beta, IntPtr c, int ldc);

        /// <summary>
        /// Wrapper for MKL SYCL GEMM with proper parameters.
        /// </summary>
        private static unsafe void ExecuteMKLSYCLGEMM(
            IntPtr queue,
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            var result = ExecuteMKLSYCLGEMM(
                queue,
                101, // CblasRowMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m, n, k,
                1.0f, // alpha
                new IntPtr(a), k, // matrix A
                new IntPtr(b), n, // matrix B
                0.0f, // beta
                new IntPtr(c), n); // matrix C

            if (result != SYCLResult.Success)
                throw new InvalidOperationException($"MKL SYCL GEMM failed with result: {result}");
        }

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

        #region Intel GPU Architecture Detection

        /// <summary>
        /// Detects Intel GPU architecture from device information.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <returns>Intel GPU architecture.</returns>
        internal static IntelGPUArchitecture DetectIntelArchitecture(IntPtr device)
        {
            try
            {
                var deviceName = GetDeviceInfoString(device, SYCLDeviceInfo.Name).ToUpperInvariant();
                var vendorName = GetDeviceInfoString(device, SYCLDeviceInfo.Vendor).ToUpperInvariant();

                if (!vendorName.Contains("INTEL", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.Unknown;

                // Intel Arc GPUs (Xe-HPG)
                if (deviceName.Contains("ARC A", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("DG2", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.XeHPG;

                // Intel Data Center GPUs (Xe-HP/HPC)
                if (deviceName.Contains("DATA CENTER", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("PONTE VECCHIO", StringComparison.OrdinalIgnoreCase) || 
                    deviceName.Contains("XE-HP", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("XE-HPC", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.XeHPC;

                // Intel Iris Xe GPUs (Gen12/Xe-LP)
                if (deviceName.Contains("IRIS XE", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("XE GRAPHICS", StringComparison.OrdinalIgnoreCase) || 
                    deviceName.Contains("GEN12", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("TGL", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("DG1", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.XeLP;

                // Intel UHD Graphics (Gen11/Gen9)
                if (deviceName.Contains("UHD", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("GEN11", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("ICL", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.Gen11;

                if (deviceName.Contains("HD", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("GEN9", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("SKL", StringComparison.OrdinalIgnoreCase))
                    return IntelGPUArchitecture.Gen9;

                return IntelGPUArchitecture.Unknown;
            }
            catch
            {
                return IntelGPUArchitecture.Unknown;
            }
        }

        #endregion
    }

    #region SYCL Enums and Structures

    /// <summary>
    /// SYCL result codes.
    /// </summary>
    internal enum SYCLResult : int
    {
        Success = 0,
        InvalidValue = -30,
        OutOfResources = -5,
        OutOfHostMemory = -6,
        DeviceNotFound = -1,
        DeviceNotAvailable = -2,
        CompilerNotAvailable = -3,
        BuildProgramFailure = -11,
        InvalidDevice = -33,
        InvalidContext = -34,
        InvalidQueue = -36,
        InvalidKernel = -48,
        InvalidArgIndex = -49,
        InvalidArgValue = -50,
        InvalidMemObject = -38,
        InvalidEventWaitList = -57,
        MisalignedSubBufferOffset = -13,
        ExecStatusErrorForEventsInWaitList = -14,
        CompileProgramFailure = -15,
        LinkerNotAvailable = -16,
        LinkProgramFailure = -17,
        DevicePartitionFailed = -18,
        KernelArgInfoNotAvailable = -19
    }

    /// <summary>
    /// SYCL device types.
    /// </summary>
    internal enum SYCLDeviceType : uint
    {
        Default = 1 << 0,
        CPU = 1 << 1,
        GPU = 1 << 2,
        Accelerator = 1 << 3,
        Custom = 1 << 4,
        All = 0xFFFFFFFF
    }

    /// <summary>
    /// SYCL device information types.
    /// </summary>
    internal enum SYCLDeviceInfo : uint
    {
        Type = 0x1000,
        VendorID = 0x1001,
        MaxComputeUnits = 0x1002,
        MaxWorkItemDimensions = 0x1003,
        MaxWorkGroupSize = 0x1004,
        MaxWorkItemSizes = 0x1005,
        PreferredVectorWidthChar = 0x1006,
        PreferredVectorWidthShort = 0x1007,
        PreferredVectorWidthInt = 0x1008,
        PreferredVectorWidthLong = 0x1009,
        PreferredVectorWidthFloat = 0x100A,
        PreferredVectorWidthDouble = 0x100B,
        MaxClockFrequency = 0x100C,
        AddressBits = 0x100D,
        MaxReadImageArgs = 0x100E,
        MaxWriteImageArgs = 0x100F,
        MaxMemAllocSize = 0x1010,
        Image2DMaxWidth = 0x1011,
        Image2DMaxHeight = 0x1012,
        Image3DMaxWidth = 0x1013,
        Image3DMaxHeight = 0x1014,
        Image3DMaxDepth = 0x1015,
        ImageSupport = 0x1016,
        MaxParameterSize = 0x1017,
        MaxSamplers = 0x1018,
        MemBaseAddrAlign = 0x1019,
        MinDataTypeAlignSize = 0x101A,
        SingleFpConfig = 0x101B,
        GlobalMemCacheType = 0x101C,
        GlobalMemCachelineSize = 0x101D,
        GlobalMemCacheSize = 0x101E,
        GlobalMemSize = 0x101F,
        MaxConstantBufferSize = 0x1020,
        MaxConstantArgs = 0x1021,
        LocalMemType = 0x1022,
        LocalMemSize = 0x1023,
        ErrorCorrectionSupport = 0x1024,
        ProfilingTimerResolution = 0x1025,
        EndianLittle = 0x1026,
        Available = 0x1027,
        CompilerAvailable = 0x1028,
        ExecutionCapabilities = 0x1029,
        QueueProperties = 0x102A,
        Name = 0x102B,
        Vendor = 0x102C,
        DriverVersion = 0x102D,
        Profile = 0x102E,
        Version = 0x102F,
        Extensions = 0x1030,
        Platform = 0x1031,
        DoubleFpConfig = 0x1032,
        HalfFpConfig = 0x1033,
        PreferredVectorWidthHalf = 0x1034,
        HostUnifiedMemory = 0x1035,
        NativeVectorWidthChar = 0x1036,
        NativeVectorWidthShort = 0x1037,
        NativeVectorWidthInt = 0x1038,
        NativeVectorWidthLong = 0x1039,
        NativeVectorWidthFloat = 0x103A,
        NativeVectorWidthDouble = 0x103B,
        NativeVectorWidthHalf = 0x103C,
        OpenCLCVersion = 0x103D
    }

    /// <summary>
    /// SYCL queue properties.
    /// </summary>
    [Flags]
    internal enum SYCLQueueProperties : ulong
    {
        None = 0,
        OutOfOrderExecModeEnable = 1 << 0,
        ProfilingEnable = 1 << 1,
        OnDevice = 1 << 2,
        OnDeviceDefault = 1 << 3
    }

    /// <summary>
    /// Intel GPU architecture enumeration.
    /// </summary>
    public enum IntelGPUArchitecture
    {
        Unknown = 0,
        Gen9 = 9,       // Intel HD Graphics (Skylake)
        Gen11 = 11,     // Intel UHD Graphics (Ice Lake)
        XeLP = 12,      // Intel Iris Xe (Tiger Lake, DG1)
        XeHPG = 13,     // Intel Arc GPUs (Alchemist, Battlemage)
        XeHP = 14,      // Intel Data Center GPU Max (Ponte Vecchio)
        XeHPC = 15,     // Intel Exascale GPUs
        Xe2 = 20        // Future Intel Xe2 architecture
    }

    /// <summary>
    /// Intel GPU device information structure.
    /// </summary>
    internal struct IntelGPUDeviceInfo
    {
        public string Name;
        public string Vendor;
        public string Version;
        public string DriverVersion;
        public IntelGPUArchitecture Architecture;
        public uint ComputeUnits;
        public uint MaxWorkGroupSize;
        public ulong GlobalMemSize;
        public ulong LocalMemSize;
        public uint MaxClockFrequency;
        public uint SubgroupSize;
        public bool SupportsFloat64;
        public bool SupportsInt64;
        public bool SupportsUnifiedMemory;
    }

    #endregion
}