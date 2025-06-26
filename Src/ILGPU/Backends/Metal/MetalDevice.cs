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

#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Represents an Apple Metal device.
    /// </summary>
    public sealed class MetalDevice : Device
    {
        #region Instance

        /// <summary>
        /// The native Metal device handle.
        /// </summary>
        internal IntPtr NativeDevice { get; }

        /// <summary>
        /// Gets the Metal device capabilities.
        /// </summary>
        public MetalCapabilities Capabilities { get; }

        /// <summary>
        /// Gets whether this is a discrete or integrated GPU.
        /// </summary>
        public bool IsDiscrete { get; }

        /// <summary>
        /// Gets whether this device supports ray tracing.
        /// </summary>
        public bool SupportsRayTracing { get; }

        /// <summary>
        /// Gets the device family (Apple Silicon generation).
        /// </summary>
        public AppleGPUFamily GPUFamily { get; }

        internal MetalDevice(IntPtr nativeDevice, int deviceId)
        {
            NativeDevice = nativeDevice;
            DeviceId = deviceId;
            
            // Query device properties
            Name = MetalNative.GetDeviceName(nativeDevice);
            Capabilities = MetalCapabilities.Query(nativeDevice);
            IsDiscrete = MetalNative.IsDiscreteGPU(nativeDevice);
            SupportsRayTracing = MetalNative.SupportsRayTracing(nativeDevice);
            GPUFamily = MetalNative.GetGPUFamily(nativeDevice);
            
            // Initialize memory info
            var memorySize = MetalNative.GetRecommendedMaxWorkingSetSize(nativeDevice);
            MemoryInfo = new MemoryInfo(memorySize, memorySize); // Unified memory
            
            Status = DeviceStatus.Available;
        }

        #endregion

        #region Properties

        public override int DeviceId { get; }
        public override string Name { get; }
        public override AcceleratorType Type => AcceleratorType.Metal;
        public override MemoryInfo MemoryInfo { get; }
        public override DeviceStatus Status { get; }

        /// <summary>
        /// Apple Silicon has native unified memory.
        /// </summary>
        public override bool SupportsUnifiedMemory => true;

        /// <summary>
        /// Apple devices support memory pools.
        /// </summary>
        public override bool SupportsMemoryPools => true;

        #endregion

        #region Device Discovery

        /// <summary>
        /// Gets all available Metal devices synchronously.
        /// </summary>
        public static IReadOnlyList<MetalDevice> GetDevices()
        {
            var devices = new List<MetalDevice>();
            
            if (!MetalNative.IsMetalSupported())
                return devices;

            var deviceCount = MetalNative.GetDeviceCount();
            for (int i = 0; i < deviceCount; i++)
            {
                var nativeDevice = MetalNative.GetDevice(i);
                if (nativeDevice != IntPtr.Zero)
                {
                    devices.Add(new MetalDevice(nativeDevice, i));
                }
            }

            return devices;
        }

        /// <summary>
        /// Discovers all available Metal devices.
        /// </summary>
        public static async Task<IReadOnlyList<MetalDevice>> DiscoverDevicesAsync()
        {
            return await Task.Run(() =>
            {
                var devices = new List<MetalDevice>();
                
                if (!MetalNative.IsMetalSupported())
                    return devices;

                var deviceCount = MetalNative.GetDeviceCount();
                for (int i = 0; i < deviceCount; i++)
                {
                    var nativeDevice = MetalNative.GetDevice(i);
                    if (nativeDevice != IntPtr.Zero)
                    {
                        devices.Add(new MetalDevice(nativeDevice, i));
                    }
                }

                return devices;
            });
        }

        #endregion

        #region Command Queue Management

        /// <summary>
        /// Creates a Metal command queue for submitting GPU commands.
        /// </summary>
        internal MetalCommandQueue CreateCommandQueue()
        {
            var queue = MetalNative.CreateCommandQueue(NativeDevice);
            if (queue == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Metal command queue");

            return new MetalCommandQueue(queue);
        }

        #endregion

        #region Library Management

        /// <summary>
        /// Creates a Metal library from compiled metallib data.
        /// </summary>
        internal async Task<MetalLibrary> CreateLibraryAsync(
            byte[] metallibData,
            CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                var libraryHandle = MetalNative.CreateLibrary(NativeDevice, metallibData, metallibData.Length);
                if (libraryHandle == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to create Metal library");

                return new MetalLibrary(libraryHandle);
            }, cancellationToken);
        }

        /// <summary>
        /// Compiles Metal source code to a library.
        /// </summary>
        internal async Task<MetalLibrary> CompileLibraryAsync(
            string source,
            MetalCompileOptions options,
            CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                var libraryHandle = MetalNative.CompileLibrary(
                    NativeDevice, source, options.ToNativeOptions());
                    
                if (libraryHandle == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to compile Metal library");

                return new MetalLibrary(libraryHandle);
            }, cancellationToken);
        }

        #endregion

        #region Accelerator Creation

        /// <summary>
        /// Creates a Metal accelerator for this device.
        /// </summary>
        public override Accelerator CreateAccelerator(Context context)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            var mps = new MetalPerformanceShaders(NativeDevice);
            return new MetalAccelerator(context, this, mps);
        }

        #endregion

        #region Disposal

        public override void Dispose()
        {
            if (NativeDevice != IntPtr.Zero)
            {
                MetalNative.ReleaseDevice(NativeDevice);
            }
        }

        #endregion
    }

    /// <summary>
    /// Apple GPU family classifications.
    /// </summary>
    public enum AppleGPUFamily
    {
        Unknown = 0,
        Apple1 = 1,     // A7
        Apple2 = 2,     // A8
        Apple3 = 3,     // A9
        Apple4 = 4,     // A10
        Apple5 = 5,     // A11
        Apple6 = 6,     // A12
        Apple7 = 7,     // A13, M1
        Apple8 = 8,     // A14, M1 Pro/Max
        Apple9 = 9,     // A15, M2
        Apple10 = 10    // A16, M2 Pro/Max/Ultra
    }
}
#endif