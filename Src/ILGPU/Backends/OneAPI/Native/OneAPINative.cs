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
using System.Runtime.InteropServices;
using System.Text;

namespace ILGPU.Backends.OneAPI.Native
{
    /// <summary>
    /// OneAPI device types for device selection.
    /// </summary>
    internal enum OneAPIDeviceType
    {
        Default = 0,
        CPU = 1,
        GPU = 2,
        FPGA = 3,
        Accelerator = 4
    }

    /// <summary>
    /// Native Intel OneAPI/SYCL API bindings.
    /// </summary>
    internal static partial class OneAPINative
    {
        #region Constants

        private const string OpenCLLibrary = "OpenCL";

        #endregion

        #region Platform and Device Management

        /// <summary>
        /// Gets all available platforms.
        /// </summary>
        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial int clGetPlatformIDs(
            uint numEntries,
            [Out] IntPtr[]? platforms,
            out uint numPlatforms);

        /// <summary>
        /// Gets all devices for a platform.
        /// </summary>
        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial int clGetDeviceIDs(
            IntPtr platform,
            ulong deviceType,
            uint numEntries,
            [Out] IntPtr[]? devices,
            out uint numDevices);

        /// <summary>
        /// Gets device information.
        /// </summary>
        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static unsafe partial int clGetDeviceInfo(
            IntPtr device,
            uint paramName,
            nuint paramValueSize,
            void* paramValue,
            out nuint paramValueSizeRet);

        /// <summary>
        /// Creates a device handle.
        /// </summary>
        internal static IntPtr CreateDevice(IntPtr deviceId, OneAPIDeviceType deviceType) =>
            // In OneAPI/SYCL, device handles are typically managed through platform APIs
            deviceId;

        /// <summary>
        /// Releases a device handle.
        /// </summary>
        internal static void ReleaseDevice(IntPtr device)
        {
            // Device handles are reference counted in OpenCL/SYCL
            int hresult = clReleaseDevice(device);
            if (hresult != 0)
            {
                // Log warning but don't throw in release methods to avoid masking original exceptions
                System.Diagnostics.Debug.WriteLine($"Warning: clReleaseDevice failed with HRESULT: 0x{hresult:X8}");
            }
        }

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clReleaseDevice(IntPtr device);

        #endregion

        #region Context and Queue Management

        /// <summary>
        /// Creates a context for the device.
        /// </summary>
        internal static IntPtr CreateContext(IntPtr device)
        {
            IntPtr[] devices = [device];
            var context = clCreateContext(
                IntPtr.Zero,
                1,
                devices,
                IntPtr.Zero,
                IntPtr.Zero,
                out var errCode);

            return errCode != 0 ? throw new InvalidOperationException($"Failed to create context: {errCode}") : context;
        }

        /// <summary>
        /// Releases a context.
        /// </summary>
        internal static void ReleaseContext(IntPtr context)
        {
            int hresult = clReleaseContext(context);
            if (hresult != 0)
            {
                // Log warning but don't throw in release methods to avoid masking original exceptions
                System.Diagnostics.Debug.WriteLine($"Warning: clReleaseContext failed with HRESULT: 0x{hresult:X8}");
            }
        }

        /// <summary>
        /// Creates a command queue for the device.
        /// </summary>
        internal static IntPtr CreateQueue(IntPtr context, IntPtr device)
        {
            var queue = clCreateCommandQueueWithProperties(
                context,
                device,
                IntPtr.Zero,
                out var errCode);

            return errCode != 0 ? throw new InvalidOperationException($"Failed to create queue: {errCode}") : queue;
        }

        /// <summary>
        /// Releases a command queue.
        /// </summary>
        internal static void ReleaseQueue(IntPtr queue)
        {
            int hresult = clReleaseCommandQueue(queue);
            if (hresult != 0)
            {
                // Log warning but don't throw in release methods to avoid masking original exceptions
                System.Diagnostics.Debug.WriteLine($"Warning: clReleaseCommandQueue failed with HRESULT: 0x{hresult:X8}");
            }
        }

        /// <summary>
        /// Waits for all commands in the queue to complete.
        /// </summary>
        internal static void QueueWait(IntPtr queue)
        {
            int hresult = clFinish(queue);
            if (hresult != 0)
            {
                throw new InvalidOperationException($"Queue wait operation failed with HRESULT: 0x{hresult:X8}");
            }
        }

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial IntPtr clCreateContext(
            IntPtr properties,
            uint numDevices,
            [In] IntPtr[] devices,
            IntPtr pfnNotify,
            IntPtr userData,
            out int errCodeRet);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clReleaseContext(IntPtr context);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial IntPtr clCreateCommandQueueWithProperties(
            IntPtr context,
            IntPtr device,
            IntPtr properties,
            out int errCodeRet);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clReleaseCommandQueue(IntPtr commandQueue);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clFinish(IntPtr commandQueue);

        #endregion

        #region Kernel Compilation

        /// <summary>
        /// Compiles a kernel from source.
        /// </summary>
        internal static IntPtr CompileKernel(IntPtr context, IntPtr device, string source, string options)
        {
            var sourceBytes = Encoding.UTF8.GetBytes(source);
            var sourcePtr = Marshal.AllocHGlobal(sourceBytes.Length);
            Marshal.Copy(sourceBytes, 0, sourcePtr, sourceBytes.Length);

            try
            {
                IntPtr[] sources = [sourcePtr];
                nuint[] lengths = [(nuint)sourceBytes.Length];
                
                var program = clCreateProgramWithSource(
                    context,
                    1,
                    sources,
                    lengths,
                    out var errCode);
                
                if (errCode != 0)
                    throw new InvalidOperationException($"Failed to create program: {errCode}");
                
                // Build the program
                IntPtr[] devices = [device];
                var buildResult = clBuildProgram(
                    program,
                    1,
                    devices,
                    options,
                    IntPtr.Zero,
                    IntPtr.Zero);
                
                if (buildResult != 0)
                {
                    var log = GetBuildLog(program, device);
                    int releaseResult = clReleaseProgram(program);
                    if (releaseResult != 0)
                    {
                        System.Diagnostics.Debug.WriteLine($"Warning: clReleaseProgram failed with HRESULT: 0x{releaseResult:X8}");
                    }
                    throw new InvalidOperationException($"Failed to build program: {log}");
                }
                
                return program;
            }
            finally
            {
                Marshal.FreeHGlobal(sourcePtr);
            }
        }

        private static string GetBuildLog(IntPtr program, IntPtr device)
        {
            int hresult = clGetProgramBuildInfo(
                program,
                device,
                0x1183, // CL_PROGRAM_BUILD_LOG
                0,
                IntPtr.Zero,
                out var logSize);
            
            if (hresult != 0)
                return $"Failed to get build log size: HRESULT 0x{hresult:X8}";
            
            if (logSize == 0)
                return "No build log available";
            
            var logBuffer = Marshal.AllocHGlobal((int)logSize);
            try
            {
                hresult = clGetProgramBuildInfo(
                    program,
                    device,
                    0x1183, // CL_PROGRAM_BUILD_LOG
                    logSize,
                    logBuffer,
                    out _);

                return hresult != 0
                    ? $"Failed to get build log: HRESULT 0x{hresult:X8}"
                    : Marshal.PtrToStringAnsi(logBuffer) ?? "Failed to read build log";
            }
            finally
            {
                Marshal.FreeHGlobal(logBuffer);
            }
        }

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial IntPtr clCreateProgramWithSource(
            IntPtr context,
            uint count,
            [In] IntPtr[] strings,
            [In] nuint[] lengths,
            out int errCodeRet);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clBuildProgram(
            IntPtr program,
            uint numDevices,
            [In] IntPtr[] deviceList,
            [MarshalAs(UnmanagedType.LPWStr)] string options,
            IntPtr pfnNotify,
            IntPtr userData);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clReleaseProgram(IntPtr program);

        [LibraryImport(OpenCLLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static partial int clGetProgramBuildInfo(
            IntPtr program,
            IntPtr device,
            uint paramName,
            nuint paramValueSize,
            IntPtr paramValue,
            out nuint paramValueSizeRet);

        #endregion

        #region Device Capabilities

        /// <summary>
        /// Gets device information of a specific type.
        /// </summary>
        internal static unsafe T GetDeviceInfo<T>(IntPtr device, OneAPIDeviceInfo info)
        {
            var paramName = (uint)info;
            
            if (typeof(T) == typeof(string))
            {
                var buffer = stackalloc byte[1024];
                int hresult = clGetDeviceInfo(device, paramName, 1024, buffer, out var actualSize);
                if (hresult != 0)
                    throw new InvalidOperationException($"Failed to get device info (string): HRESULT 0x{hresult:X8}");
                var str = Marshal.PtrToStringAnsi((IntPtr)buffer, (int)actualSize - 1);
                return (T)(object)str;
            }
            else if (typeof(T) == typeof(int))
            {
                int value = 0;
                int hresult = clGetDeviceInfo(device, paramName, sizeof(int), &value, out _);
                return hresult != 0
                    ? throw new InvalidOperationException($"Failed to get device info (int): HRESULT 0x{hresult:X8}")
                    : (T)(object)value;
            }
            else if (typeof(T) == typeof(long))
            {
                long value = 0;
                int hresult = clGetDeviceInfo(device, paramName, sizeof(long), &value, out _);
                return hresult != 0
                    ? throw new InvalidOperationException($"Failed to get device info (long): HRESULT 0x{hresult:X8}")
                    : (T)(object)value;
            }
            else if (typeof(T) == typeof(long[]))
            {
                var values = new long[3];
                fixed (long* ptr = values)
                {
                    int hresult = clGetDeviceInfo(device, paramName, sizeof(long) * 3, ptr, out _);
                    if (hresult != 0)
                        throw new InvalidOperationException($"Failed to get device info (long[]): HRESULT 0x{hresult:X8}");
                }
                return (T)(object)values;
            }
            else if (typeof(T) == typeof(OneAPIDeviceType))
            {
                ulong value = 0;
                int hresult = clGetDeviceInfo(device, paramName, sizeof(ulong), &value, out _);
                return hresult != 0
                    ? throw new InvalidOperationException($"Failed to get device info (OneAPIDeviceType): HRESULT 0x{hresult:X8}")
                    : (T)(object)ConvertToDeviceType(value);
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported for device info query");
        }

        private static OneAPIDeviceType ConvertToDeviceType(ulong clDeviceType) => clDeviceType switch
        {
            0x01 => OneAPIDeviceType.Default,
            0x02 => OneAPIDeviceType.CPU,
            0x04 => OneAPIDeviceType.GPU,
            0x08 => OneAPIDeviceType.Accelerator,
            _ => OneAPIDeviceType.Default
        };

        /// <summary>
        /// Checks if the device supports Unified Shared Memory.
        /// </summary>
        internal static bool SupportsUSM(IntPtr device)
        {
            // Check for Intel USM extension
            var extensions = GetDeviceInfo<string>(device, OneAPIDeviceInfo.Extensions);
            return extensions.Contains("cl_intel_unified_shared_memory", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Checks if the device supports FP16.
        /// </summary>
        internal static bool SupportsFP16(IntPtr device)
        {
            var extensions = GetDeviceInfo<string>(device, OneAPIDeviceInfo.Extensions);
            return extensions.Contains("cl_khr_fp16", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Checks if the device supports FP64.
        /// </summary>
        internal static bool SupportsFP64(IntPtr device)
        {
            var extensions = GetDeviceInfo<string>(device, OneAPIDeviceInfo.Extensions);
            return extensions.Contains("cl_khr_fp64", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Checks if the device supports subgroups.
        /// </summary>
        internal static bool SupportsSubgroups(IntPtr device)
        {
            var extensions = GetDeviceInfo<string>(device, OneAPIDeviceInfo.Extensions);
            return extensions.Contains("cl_intel_subgroups", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Gets the subgroup size.
        /// </summary>
        internal static int GetSubgroupSize(IntPtr device)
        {
            // Intel GPUs typically use subgroup size of 8, 16, or 32
            if (SupportsSubgroups(device))
            {
                // This would query the actual subgroup size
                // For now, return a typical value
                return 16;
            }
            return 1;
        }

        /// <summary>
        /// Gets the number of execution units (Intel-specific).
        /// </summary>
        internal static int GetNumExecutionUnits(IntPtr device)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Query Intel-specific extension for EU count
                return GetDeviceInfo<int>(device, (OneAPIDeviceInfo)0x4050); // CL_DEVICE_EU_COUNT_INTEL
            }
            catch
            {
                // Fallback: estimate from max compute units
                return GetDeviceInfo<int>(device, OneAPIDeviceInfo.MaxComputeUnits);
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Gets the maximum threads per EU (Intel-specific).
        /// </summary>
        internal static int GetMaxThreadsPerEU(IntPtr device) =>
            // Intel GPUs typically have 7 threads per EU
            7;

        /// <summary>
        /// Gets the preferred work group size multiple.
        /// </summary>
        internal static int GetPreferredWorkGroupSizeMultiple(IntPtr device) => GetSubgroupSize(device);

        #endregion

        #region Memory Management

        /// <summary>
        /// Checks if two devices can share memory.
        /// </summary>
        internal static bool CanShareMemory(IntPtr device1, IntPtr device2) =>
            // Check if devices are on the same platform
            // In OneAPI/SYCL, USM enables sharing between devices
            SupportsUSM(device1) && SupportsUSM(device2);

        /// <summary>
        /// Enables peer access between contexts.
        /// </summary>
        internal static void EnablePeerAccess(IntPtr context1, IntPtr context2)
        {
            // In OneAPI/SYCL, peer access is handled through USM
            // No explicit enable needed
        }

        /// <summary>
        /// Disables peer access between contexts.
        /// </summary>
        internal static void DisablePeerAccess(IntPtr context1, IntPtr context2)
        {
            // In OneAPI/SYCL, peer access is handled through USM
            // No explicit disable needed
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets all available platforms.
        /// </summary>
        internal static List<IntPtr> GetPlatforms()
        {
            int hresult = clGetPlatformIDs(0, null, out var numPlatforms);
            if (hresult != 0)
                throw new InvalidOperationException($"Failed to get platform count: HRESULT 0x{hresult:X8}");
            
            if (numPlatforms == 0)
                return [];
            
            var platforms = new IntPtr[numPlatforms];
            hresult = clGetPlatformIDs(numPlatforms, platforms, out _);
            return hresult != 0 ? throw new InvalidOperationException($"Failed to get platforms: HRESULT 0x{hresult:X8}") : [.. platforms];
        }

        /// <summary>
        /// Gets all devices for a platform.
        /// </summary>
        internal static List<IntPtr> GetDevices(IntPtr platform)
        {
            const ulong CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
            
            int hresult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, out var numDevices);
            if (hresult != 0)
                throw new InvalidOperationException($"Failed to get device count: HRESULT 0x{hresult:X8}");
            
            if (numDevices == 0)
                return [];
            
            var devices = new IntPtr[numDevices];
            hresult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, out _);
            return hresult != 0 ? throw new InvalidOperationException($"Failed to get devices: HRESULT 0x{hresult:X8}") : [.. devices];
        }

        /// <summary>
        /// Sets the current device.
        /// </summary>
        internal static void SetCurrentDevice(IntPtr device)
        {
            // In OneAPI/SYCL, device selection is handled through queue creation
            // No explicit device set needed
        }

        #endregion
    }

    /// <summary>
    /// OneAPI device information parameters.
    /// </summary>
    internal enum OneAPIDeviceInfo : uint
    {
        Type = 0x1000,
        VendorId = 0x1001,
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
        SingleFPConfig = 0x101B,
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
        Name = 0x102B,
        Vendor = 0x102C,
        DriverVersion = 0x102D,
        Profile = 0x102E,
        Version = 0x102F,
        Extensions = 0x1030,
        Platform = 0x1031,
        DoubleFPConfig = 0x1032,
        GlobalMemCacheBandwidth = 0x1036,
        NumericVersion = 0x105E
    }
}
