// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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

namespace ILGPU.Backends.Metal.Native
{
    /// <summary>
    /// Apple GPU families for capability checking.
    /// </summary>
    internal enum AppleGPUFamily
    {
        Apple1 = 1,
        Apple2 = 2,
        Apple3 = 3,
        Apple4 = 4,
        Apple5 = 5,
        Apple6 = 6,
        Apple7 = 7,
        Apple8 = 8,
        Apple9 = 9,
        Apple10 = 10
    }

    /// <summary>
    /// Native Metal API bindings for Apple platforms.
    /// </summary>
    internal static partial class MetalNative
    {
        #region Constants

        private const string MetalLibrary = "Metal.framework/Metal";
        private const string CoreFoundationLibrary = "CoreFoundation.framework/CoreFoundation";

        #endregion

        #region Device Management

        /// <summary>
        /// Checks if Metal is supported on this system.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool MTLCreateSystemDefaultDevice();

        /// <summary>
        /// Gets the number of available Metal devices.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial int MTLCopyAllDevices();

        /// <summary>
        /// Gets a Metal device by index.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLCopyAllDevicesGetDevice(int index);

        /// <summary>
        /// Gets the device name.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceGetName(IntPtr device);

        /// <summary>
        /// Checks if the device is a discrete GPU.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool MTLDeviceIsLowPower(IntPtr device);

        /// <summary>
        /// Gets the recommended maximum working set size.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial ulong MTLDeviceRecommendedMaxWorkingSetSize(IntPtr device);

        /// <summary>
        /// Gets the recommended maximum working set size.
        /// </summary>
        internal static ulong GetRecommendedMaxWorkingSetSize(IntPtr device)
        {
            return MTLDeviceRecommendedMaxWorkingSetSize(device);
        }

        /// <summary>
        /// Checks ray tracing support.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool MTLDeviceSupportsRaytracing(IntPtr device);

        /// <summary>
        /// Gets the GPU family.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial int MTLDeviceSupportsFamily(IntPtr device, int family);

        /// <summary>
        /// Releases a Metal device.
        /// </summary>
        [LibraryImport(CoreFoundationLibrary)]
        internal static partial void CFRelease(IntPtr obj);

        #endregion

        #region Command Queue Management

        /// <summary>
        /// Creates a command queue.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewCommandQueue(IntPtr device);

        /// <summary>
        /// Creates a command buffer.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLCommandQueueCommandBuffer(IntPtr queue);

        /// <summary>
        /// Commits a command buffer for execution.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLCommandBufferCommit(IntPtr commandBuffer);

        /// <summary>
        /// Waits for command buffer completion.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLCommandBufferWaitUntilCompleted(IntPtr commandBuffer);

        #endregion

        #region Buffer Management

        /// <summary>
        /// Creates a Metal buffer.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewBuffer(
            IntPtr device, nuint length, int options);

        /// <summary>
        /// Creates a Metal buffer with data.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewBufferWithBytes(
            IntPtr device, IntPtr bytes, nuint length, int options);

        /// <summary>
        /// Gets buffer contents pointer.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLBufferContents(IntPtr buffer);

        /// <summary>
        /// Gets buffer length.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial nuint MTLBufferLength(IntPtr buffer);

        #endregion

        #region Library and Function Management

        /// <summary>
        /// Creates a library from compiled data.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewLibraryWithData(
            IntPtr device, IntPtr data, nuint length, out IntPtr error);

        /// <summary>
        /// Compiles a library from source.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewLibraryWithSource(
            IntPtr device, IntPtr source, IntPtr options, out IntPtr error);

        /// <summary>
        /// Creates a function from library.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLLibraryNewFunctionWithName(
            IntPtr library, IntPtr name);

        /// <summary>
        /// Creates a compute pipeline state.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLDeviceNewComputePipelineStateWithFunction(
            IntPtr device, IntPtr function, out IntPtr error);

        #endregion

        #region Compute Encoder

        /// <summary>
        /// Creates a compute command encoder.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial IntPtr MTLCommandBufferComputeCommandEncoder(
            IntPtr commandBuffer);

        /// <summary>
        /// Sets compute pipeline state.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLComputeCommandEncoderSetComputePipelineState(
            IntPtr encoder, IntPtr pipelineState);

        /// <summary>
        /// Sets buffer for compute encoder.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLComputeCommandEncoderSetBuffer(
            IntPtr encoder, IntPtr buffer, nuint offset, nuint index);

        /// <summary>
        /// Dispatches compute threads.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLComputeCommandEncoderDispatchThreadgroups(
            IntPtr encoder, MTLSize threadgroupsPerGrid, MTLSize threadsPerThreadgroup);

        /// <summary>
        /// Ends encoding.
        /// </summary>
        [LibraryImport(MetalLibrary)]
        internal static partial void MTLComputeCommandEncoderEndEncoding(IntPtr encoder);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks if Metal is supported.
        /// </summary>
        internal static bool IsMetalSupported()
        {
            try
            {
                return MTLCreateSystemDefaultDevice();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the number of Metal devices.
        /// </summary>
        internal static int GetDeviceCount()
        {
            return MTLCopyAllDevices();
        }

        /// <summary>
        /// Gets a Metal device.
        /// </summary>
        internal static IntPtr GetDevice(int index)
        {
            return MTLCopyAllDevicesGetDevice(index);
        }

        /// <summary>
        /// Gets device name as string.
        /// </summary>
        internal static string GetDeviceName(IntPtr device)
        {
            var namePtr = MTLDeviceGetName(device);
            return Marshal.PtrToStringAnsi(namePtr) ?? "Unknown Metal Device";
        }

        /// <summary>
        /// Checks if device is discrete GPU.
        /// </summary>
        internal static bool IsDiscreteGPU(IntPtr device)
        {
            return !MTLDeviceIsLowPower(device);
        }

        /// <summary>
        /// Checks ray tracing support.
        /// </summary>
        internal static bool SupportsRayTracing(IntPtr device)
        {
            return MTLDeviceSupportsRaytracing(device);
        }

        /// <summary>
        /// Gets GPU family.
        /// </summary>
        internal static AppleGPUFamily GetGPUFamily(IntPtr device)
        {
            // Check from newest to oldest
            for (int family = 10; family >= 1; family--)
            {
                if (MTLDeviceSupportsFamily(device, family) != 0)
                {
                    return (AppleGPUFamily)family;
                }
            }
            return AppleGPUFamily.Apple1;
        }

        /// <summary>
        /// Creates command queue.
        /// </summary>
        internal static IntPtr CreateCommandQueue(IntPtr device)
        {
            return MTLDeviceNewCommandQueue(device);
        }

        /// <summary>
        /// Creates library from data.
        /// </summary>
        internal static IntPtr CreateLibrary(IntPtr device, byte[] data, int length)
        {
            IntPtr dataPtr = Marshal.AllocHGlobal(length);
            try
            {
                Marshal.Copy(data, 0, dataPtr, length);
                var library = MTLDeviceNewLibraryWithData(device, dataPtr, (nuint)length, out var error);
                if (error != IntPtr.Zero)
                {
                    CFRelease(error);
                    return IntPtr.Zero;
                }
                return library;
            }
            finally
            {
                Marshal.FreeHGlobal(dataPtr);
            }
        }

        /// <summary>
        /// Compiles library from source.
        /// </summary>
        internal static IntPtr CompileLibrary(IntPtr device, string source, IntPtr options)
        {
            var sourcePtr = Marshal.StringToHGlobalAnsi(source);
            try
            {
                var library = MTLDeviceNewLibraryWithSource(device, sourcePtr, options, out var error);
                if (error != IntPtr.Zero)
                {
                    CFRelease(error);
                    return IntPtr.Zero;
                }
                return library;
            }
            finally
            {
                Marshal.FreeHGlobal(sourcePtr);
            }
        }

        /// <summary>
        /// Releases device.
        /// </summary>
        internal static void ReleaseDevice(IntPtr device)
        {
            if (device != IntPtr.Zero)
                CFRelease(device);
        }

        #endregion
    }

    /// <summary>
    /// Metal 3D size structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct MTLSize
    {
        public nuint Width;
        public nuint Height;
        public nuint Depth;

        public MTLSize(nuint width, nuint height, nuint depth)
        {
            Width = width;
            Height = height;
            Depth = depth;
        }
    }
}