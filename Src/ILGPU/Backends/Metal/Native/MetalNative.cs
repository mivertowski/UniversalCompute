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
    internal static class MetalNative
    {
        #region Constants

        private const string MetalLibrary = "Metal.framework/Metal";
        private const string CoreFoundationLibrary = "CoreFoundation.framework/CoreFoundation";

        #endregion

        #region Device Management

        /// <summary>
        /// Checks if Metal is supported on this system.
        /// </summary>
        [DllImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool MTLCreateSystemDefaultDevice();

        /// <summary>
        /// Gets the number of available Metal devices.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern int MTLCopyAllDevices();

        /// <summary>
        /// Gets a Metal device by index.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLCopyAllDevicesGetDevice(int index);

        /// <summary>
        /// Gets the device name.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceGetName(IntPtr device);

        /// <summary>
        /// Checks if the device is a discrete GPU.
        /// </summary>
        [DllImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool MTLDeviceIsLowPower(IntPtr device);

        /// <summary>
        /// Gets the recommended maximum working set size.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern ulong MTLDeviceRecommendedMaxWorkingSetSize(IntPtr device);

        /// <summary>
        /// Gets the recommended maximum working set size.
        /// </summary>
        internal static ulong GetRecommendedMaxWorkingSetSize(IntPtr device) => MTLDeviceRecommendedMaxWorkingSetSize(device);

        /// <summary>
        /// Checks ray tracing support.
        /// </summary>
        [DllImport(MetalLibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool MTLDeviceSupportsRaytracing(IntPtr device);

        /// <summary>
        /// Gets the GPU family.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern int MTLDeviceSupportsFamily(IntPtr device, int family);

        /// <summary>
        /// Releases a Metal device.
        /// </summary>
        [DllImport(CoreFoundationLibrary)]
        internal static extern void CFRelease(IntPtr obj);

        #endregion

        #region Command Queue Management

        /// <summary>
        /// Creates a command queue.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewCommandQueue(IntPtr device);

        /// <summary>
        /// Creates a command buffer.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLCommandQueueCommandBuffer(IntPtr queue);

        /// <summary>
        /// Commits a command buffer for execution.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLCommandBufferCommit(IntPtr commandBuffer);

        /// <summary>
        /// Waits for command buffer completion.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLCommandBufferWaitUntilCompleted(IntPtr commandBuffer);

        #endregion

        #region Buffer Management

        /// <summary>
        /// Creates a Metal buffer.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewBuffer(
            IntPtr device, nuint length, int options);

        /// <summary>
        /// Creates a Metal buffer with data.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewBufferWithBytes(
            IntPtr device, IntPtr bytes, nuint length, int options);

        /// <summary>
        /// Gets buffer contents pointer.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLBufferContents(IntPtr buffer);

        /// <summary>
        /// Gets buffer length.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern nuint MTLBufferLength(IntPtr buffer);

        #endregion

        #region Library and Function Management

        /// <summary>
        /// Creates a library from compiled data.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewLibraryWithData(
            IntPtr device, IntPtr data, nuint length, out IntPtr error);

        /// <summary>
        /// Compiles a library from source.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewLibraryWithSource(
            IntPtr device, IntPtr source, IntPtr options, out IntPtr error);

        /// <summary>
        /// Creates a function from library.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLLibraryNewFunctionWithName(
            IntPtr library, IntPtr name);

        /// <summary>
        /// Creates a compute pipeline state.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLDeviceNewComputePipelineStateWithFunction(
            IntPtr device, IntPtr function, out IntPtr error);

        #endregion

        #region Compute Encoder

        /// <summary>
        /// Creates a compute command encoder.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern IntPtr MTLCommandBufferComputeCommandEncoder(
            IntPtr commandBuffer);

        /// <summary>
        /// Sets compute pipeline state.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLComputeCommandEncoderSetComputePipelineState(
            IntPtr encoder, IntPtr pipelineState);

        /// <summary>
        /// Sets buffer for compute encoder.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLComputeCommandEncoderSetBuffer(
            IntPtr encoder, IntPtr buffer, nuint offset, nuint index);

        /// <summary>
        /// Dispatches compute threads.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLComputeCommandEncoderDispatchThreadgroups(
            IntPtr encoder, MTLSize threadgroupsPerGrid, MTLSize threadsPerThreadgroup);

        /// <summary>
        /// Ends encoding.
        /// </summary>
        [DllImport(MetalLibrary)]
        internal static extern void MTLComputeCommandEncoderEndEncoding(IntPtr encoder);

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
        internal static int GetDeviceCount() => MTLCopyAllDevices();

        /// <summary>
        /// Gets a Metal device.
        /// </summary>
        internal static IntPtr GetDevice(int index) => MTLCopyAllDevicesGetDevice(index);

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
        internal static bool IsDiscreteGPU(IntPtr device) => !MTLDeviceIsLowPower(device);

        /// <summary>
        /// Checks ray tracing support.
        /// </summary>
        internal static bool SupportsRayTracing(IntPtr device) => MTLDeviceSupportsRaytracing(device);

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
        internal static IntPtr CreateCommandQueue(IntPtr device) => MTLDeviceNewCommandQueue(device);

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
        /// Creates system default Metal device.
        /// </summary>
        internal static IntPtr CreateSystemDefaultDevice()
        {
            try
            {
                if (MTLCreateSystemDefaultDevice())
                {
                    // Get the first available device
                    return MTLCopyAllDevicesGetDevice(0);
                }
                return IntPtr.Zero;
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Copies all available Metal devices.
        /// </summary>
        internal static IntPtr[] CopyAllDevices()
        {
            try
            {
                var deviceCount = GetDeviceCount();
                var devices = new IntPtr[deviceCount];
                
                for (int i = 0; i < deviceCount; i++)
                {
                    devices[i] = GetDevice(i);
                }
                
                return devices;
            }
            catch
            {
                return Array.Empty<IntPtr>();
            }
        }

        /// <summary>
        /// Creates a new command queue.
        /// </summary>
        internal static IntPtr NewCommandQueue(IntPtr device) => MTLDeviceNewCommandQueue(device);

        /// <summary>
        /// Releases command queue.
        /// </summary>
        internal static void ReleaseCommandQueue(IntPtr commandQueue)
        {
            if (commandQueue != IntPtr.Zero)
                CFRelease(commandQueue);
        }

        /// <summary>
        /// Synchronizes command queue.
        /// </summary>
        internal static void SynchronizeCommandQueue(IntPtr commandQueue)
        {
            // Create a command buffer and wait for completion
            var commandBuffer = MTLCommandQueueCommandBuffer(commandQueue);
            if (commandBuffer != IntPtr.Zero)
            {
                MTLCommandBufferCommit(commandBuffer);
                MTLCommandBufferWaitUntilCompleted(commandBuffer);
                CFRelease(commandBuffer);
            }
        }

        /// <summary>
        /// Allocates Metal buffer memory.
        /// </summary>
        internal static IntPtr AllocateMemory(ulong sizeInBytes) =>
            // This would typically be done through the device
            // For now, return a placeholder
            IntPtr.Zero;

        /// <summary>
        /// Frees Metal buffer memory.
        /// </summary>
        internal static void FreeMemory(IntPtr buffer)
        {
            if (buffer != IntPtr.Zero)
                CFRelease(buffer);
        }

        /// <summary>
        /// Creates an MPS Graph.
        /// </summary>
        internal static IntPtr CreateMPSGraph(IntPtr device) =>
            // This would create an MPSGraph instance
            // Placeholder for actual MPS Graph creation
            IntPtr.Zero;

        /// <summary>
        /// Creates a compute pipeline state.
        /// </summary>
        internal static IntPtr CreateComputePipelineState(IntPtr device, string shaderSource)
        {
            // Compile shader source and create pipeline state
            var library = CompileLibrary(device, shaderSource, IntPtr.Zero);
            if (library == IntPtr.Zero) return IntPtr.Zero;

            // Get the main function (assuming function name is "main0")
            var functionName = Marshal.StringToHGlobalAnsi("main0");
            try
            {
                var function = MTLLibraryNewFunctionWithName(library, functionName);
                if (function == IntPtr.Zero) return IntPtr.Zero;

                var pipelineState = MTLDeviceNewComputePipelineStateWithFunction(
                    device, function, out var error);
                
                CFRelease(function);
                if (error != IntPtr.Zero)
                {
                    CFRelease(error);
                    return IntPtr.Zero;
                }
                
                return pipelineState;
            }
            finally
            {
                Marshal.FreeHGlobal(functionName);
                CFRelease(library);
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
    internal struct MTLSize(nuint width, nuint height, nuint depth)
    {
        public nuint Width = width;
        public nuint Height = height;
        public nuint Depth = depth;
    }
}