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

#if ENABLE_ONEAPI_ACCELERATOR
namespace ILGPU.Backends.OneAPI
{
    /// <summary>
    /// OneAPI kernel implementation.
    /// </summary>
    public sealed class OneAPIKernel : IDisposable
    {
        private readonly OneAPIAccelerator _accelerator;
        private readonly IntPtr _program;
        private readonly IntPtr _kernel;
        private readonly string _kernelName;
        private readonly Dictionary<int, object> _arguments;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the OneAPIKernel class.
        /// </summary>
        /// <param name="accelerator">The OneAPI accelerator.</param>
        /// <param name="program">The compiled program handle.</param>
        /// <param name="kernelName">The kernel function name.</param>
        public OneAPIKernel(OneAPIAccelerator accelerator, IntPtr program, string kernelName)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _program = program;
            _kernelName = kernelName ?? throw new ArgumentNullException(nameof(kernelName));
            _arguments = new Dictionary<int, object>();

            // Create kernel from program
            _kernel = OneAPIKernelNative.CreateKernel(program, kernelName);
            if (_kernel == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create kernel: {kernelName}");
        }

        /// <summary>
        /// Gets the kernel name.
        /// </summary>
        public string Name => _kernelName;

        /// <summary>
        /// Sets a kernel argument.
        /// </summary>
        /// <param name="index">The argument index.</param>
        /// <param name="value">The argument value.</param>
        public void SetArgument(int index, object value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            _arguments[index] = value;
            
            // Set the argument in the native kernel
            unsafe
            {
                if (value is MemoryBuffer buffer)
                {
                    var ptr = buffer.NativePtr;
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, (nuint)IntPtr.Size, &ptr);
                }
                else if (value is int intValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(int), &intValue);
                }
                else if (value is float floatValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(float), &floatValue);
                }
                else if (value is long longValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(long), &longValue);
                }
                else if (value is double doubleValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(double), &doubleValue);
                }
                else if (value is Index3D index3d)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, (nuint)Marshal.SizeOf<Index3D>(), &index3d);
                }
                else if (value is IntPtr ptrValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, (nuint)IntPtr.Size, &ptrValue);
                }
                else if (value is byte byteValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(byte), &byteValue);
                }
                else if (value is short shortValue)
                {
                    OneAPIKernelNative.SetKernelArg(_kernel, (uint)index, sizeof(short), &shortValue);
                }
                else
                {
                    throw new NotSupportedException($"Kernel argument type {value.GetType()} not supported");
                }
            }
        }

        /// <summary>
        /// Launches the kernel with the specified configuration.
        /// </summary>
        /// <param name="queue">The command queue.</param>
        /// <param name="globalSize">The global work size.</param>
        /// <param name="localSize">The local work size.</param>
        public void Launch(IntPtr queue, Index3D globalSize, Index3D localSize)
        {
            // Convert Index3D to size arrays
            nuint[] global = { (nuint)globalSize.X, (nuint)globalSize.Y, (nuint)globalSize.Z };
            nuint[] local = { (nuint)localSize.X, (nuint)localSize.Y, (nuint)localSize.Z };
            
            // Determine work dimensions
            uint dimensions = 1;
            if (globalSize.Z > 1) dimensions = 3;
            else if (globalSize.Y > 1) dimensions = 2;

            unsafe
            {
                fixed (nuint* globalPtr = global)
                fixed (nuint* localPtr = local)
                {
                    var result = OneAPIKernelNative.EnqueueNDRangeKernel(
                        queue,
                        _kernel,
                        dimensions,
                        null, // global offset
                        globalPtr,
                        localPtr,
                        0,
                        null,
                        IntPtr.Zero);
                    
                    if (result != 0)
                        throw new InvalidOperationException($"Failed to launch kernel: {result}");
                }
            }
        }

        /// <summary>
        /// Gets the preferred work group size for this kernel.
        /// </summary>
        /// <param name="device">The device handle.</param>
        /// <returns>The preferred work group size.</returns>
        public int GetPreferredWorkGroupSize(IntPtr device)
        {
            return OneAPIKernelNative.GetKernelWorkGroupInfo<int>(
                _kernel,
                device,
                KernelWorkGroupInfo.PreferredWorkGroupSizeMultiple);
        }

        /// <summary>
        /// Gets the maximum work group size for this kernel.
        /// </summary>
        /// <param name="device">The device handle.</param>
        /// <returns>The maximum work group size.</returns>
        public int GetMaxWorkGroupSize(IntPtr device)
        {
            return OneAPIKernelNative.GetKernelWorkGroupInfo<int>(
                _kernel,
                device,
                KernelWorkGroupInfo.WorkGroupSize);
        }

        /// <summary>
        /// Disposes the kernel.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_kernel != IntPtr.Zero)
                    OneAPIKernelNative.ReleaseKernel(_kernel);
                if (_program != IntPtr.Zero)
                    OneAPIKernelNative.ReleaseProgram(_program);
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Adapter to wrap OneAPI kernel as ILGPU kernel.
    /// </summary>
    public sealed class OneAPIKernelAdapter : Kernel
    {
        private readonly OneAPIKernel _oneapiKernel;
        private readonly IntPtr _queue;

        /// <summary>
        /// Initializes a new instance of the OneAPIKernelAdapter class.
        /// </summary>
        /// <param name="accelerator">The OneAPI accelerator.</param>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <param name="oneapiKernel">The OneAPI kernel.</param>
        public OneAPIKernelAdapter(
            OneAPIAccelerator accelerator,
            CompiledKernel compiledKernel,
            OneAPIKernel oneapiKernel)
            : base(accelerator, compiledKernel)
        {
            _oneapiKernel = oneapiKernel ?? throw new ArgumentNullException(nameof(oneapiKernel));
            
            // Get queue from current stream
            var stream = accelerator.DefaultStream as OneAPIStream;
            _queue = stream != null ? GetQueueFromStream(stream) : IntPtr.Zero;
        }

        /// <summary>
        /// Launches the kernel with the specified configuration.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="extent">The kernel configuration.</param>
        /// <param name="runtimeKernelConfig">The runtime kernel configuration.</param>
        protected override void LaunchInternal(
            AcceleratorStream stream,
            KernelConfig extent,
            RuntimeKernelConfig runtimeKernelConfig)
        {
            var oneapiStream = stream as OneAPIStream;
            if (oneapiStream == null)
                throw new InvalidOperationException("Stream must be a OneAPI stream");

            var queue = GetQueueFromStream(oneapiStream);
            
            // Set kernel arguments from runtime config
            SetKernelArguments(runtimeKernelConfig);
            
            // Launch kernel
            _oneapiKernel.Launch(queue, extent.GridDim, extent.GroupDim);
        }

        private void SetKernelArguments(RuntimeKernelConfig config)
        {
            try
            {
                // Set kernel arguments based on runtime configuration
                var args = config.ToArguments();
                for (int i = 0; i < args.Length; i++)
                {
                    if (args[i] != null)
                    {
                        _oneapiKernel.SetArgument(i, args[i]);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to set kernel arguments: {ex.Message}", ex);
            }
        }

        private static IntPtr GetQueueFromStream(OneAPIStream stream)
        {
            // Use reflection to access the private _queue field
            var queueField = typeof(OneAPIStream).GetField("_queue", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return queueField != null ? (IntPtr)queueField.GetValue(stream) : IntPtr.Zero;
        }

        /// <summary>
        /// Disposes the kernel adapter.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                _oneapiKernel?.Dispose();
            }
        }
    }

    /// <summary>
    /// Native OneAPI kernel operations.
    /// </summary>
    internal static extern class OneAPIKernelNative
    {
        [DllImport("OpenCL")]
        internal static extern IntPtr clCreateKernel(
            IntPtr program,
            [MarshalAs(UnmanagedType.LPStr)] string kernelName,
            out int errCodeRet);

        [DllImport("OpenCL")]
        internal static extern int clReleaseKernel(IntPtr kernel);

        [DllImport("OpenCL")]
        internal static extern int clReleaseProgram(IntPtr program);

        [DllImport("OpenCL")]
        internal static unsafe partial int clSetKernelArg(
            IntPtr kernel,
            uint argIndex,
            nuint argSize,
            void* argValue);

        [DllImport("OpenCL")]
        internal static unsafe partial int clEnqueueNDRangeKernel(
            IntPtr commandQueue,
            IntPtr kernel,
            uint workDim,
            nuint* globalWorkOffset,
            nuint* globalWorkSize,
            nuint* localWorkSize,
            uint numEventsInWaitList,
            IntPtr* eventWaitList,
            IntPtr @event);

        internal static unsafe int EnqueueNDRangeKernel(
            IntPtr queue,
            IntPtr kernel,
            uint dimensions,
            nuint* globalOffset,
            nuint* globalSize,
            nuint* localSize,
            uint numEvents,
            IntPtr* events,
            IntPtr @event)
        {
            return clEnqueueNDRangeKernel(queue, kernel, dimensions, globalOffset, globalSize, localSize, numEvents, events, @event);
        }

        [DllImport("OpenCL")]
        internal static unsafe partial int clGetKernelWorkGroupInfo(
            IntPtr kernel,
            IntPtr device,
            uint paramName,
            nuint paramValueSize,
            void* paramValue,
            out nuint paramValueSizeRet);

        internal static IntPtr CreateKernel(IntPtr program, string kernelName)
        {
            var kernel = clCreateKernel(program, kernelName, out var errCode);
            if (errCode != 0)
                return IntPtr.Zero;
            return kernel;
        }

        internal static void ReleaseKernel(IntPtr kernel) => clReleaseKernel(kernel);
        
        internal static void ReleaseProgram(IntPtr program) => clReleaseProgram(program);

        internal static unsafe void SetKernelArg(IntPtr kernel, uint index, nuint size, void* value)
        {
            clSetKernelArg(kernel, index, size, value);
        }

        internal static unsafe T GetKernelWorkGroupInfo<T>(IntPtr kernel, IntPtr device, KernelWorkGroupInfo info)
        {
            var paramName = (uint)info;
            
            if (typeof(T) == typeof(int))
            {
                int value = 0;
                clGetKernelWorkGroupInfo(kernel, device, paramName, sizeof(int), &value, out _);
                return (T)(object)value;
            }
            else if (typeof(T) == typeof(long))
            {
                long value = 0;
                clGetKernelWorkGroupInfo(kernel, device, paramName, sizeof(long), &value, out _);
                return (T)(object)value;
            }
            
            throw new NotSupportedException($"Type {typeof(T)} not supported for kernel work group info query");
        }
    }

    /// <summary>
    /// Kernel work group information parameters.
    /// </summary>
    internal enum KernelWorkGroupInfo : uint
    {
        WorkGroupSize = 0x11B0,
        CompileWorkGroupSize = 0x11B1,
        LocalMemSize = 0x11B2,
        PreferredWorkGroupSizeMultiple = 0x11B3,
        PrivateMemSize = 0x11B4
    }
}
#endif