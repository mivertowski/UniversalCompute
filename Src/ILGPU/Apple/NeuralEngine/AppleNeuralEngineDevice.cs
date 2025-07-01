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

using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Represents an Apple Neural Engine device.
    /// </summary>
    [DeviceType(AcceleratorType.AppleNeuralEngine)]
    public sealed class AppleNeuralEngineDevice : Device
    {
        #region Static

        /// <summary>
        /// The default Apple Neural Engine device for Apple Silicon systems.
        /// </summary>
        public static readonly AppleNeuralEngineDevice Default = new();

        /// <summary>
        /// Detects and enumerates all available Apple Neural Engine devices.
        /// </summary>
        /// <param name="registry">The device registry.</param>
        public static void RegisterDevices(DeviceRegistry registry)
        {
            if (!IsAppleSilicon() || !ANECapabilities.DetectNeuralEngine())
                return;

            registry.Register(
                Default,
                device => IsAppleSilicon() && ANECapabilities.DetectNeuralEngine());
        }

        /// <summary>
        /// Checks if running on Apple Silicon.
        /// </summary>
        /// <returns>True if running on Apple Silicon; otherwise, false.</returns>
        private static bool IsAppleSilicon() =>
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) &&
            RuntimeInformation.ProcessArchitecture == Architecture.Arm64;

        #endregion

        #region Instance

        /// <summary>
        /// Initializes a new Apple Neural Engine device instance.
        /// </summary>
        private AppleNeuralEngineDevice()
        {
            if (!IsAppleSilicon())
                throw new NotSupportedException("Apple Neural Engine only supported on Apple Silicon");

            var capabilities = ANECapabilities.Query();
            if (!capabilities.IsAvailable)
                throw new NotSupportedException("Apple Neural Engine not available on this device");
            
            Name = $"Apple Neural Engine ({capabilities.Generation})";
            MemorySize = 16L * 1024 * 1024 * 1024; // 16 GB unified memory estimate
            MaxGridSize = new Index3D(65535, 65535, 65535);
            MaxGroupSize = new Index3D(1024, 1024, 1024);
            MaxNumThreadsPerGroup = 1024;
            MaxSharedMemoryPerGroup = 64 * 1024; // 64KB shared memory estimate
            MaxConstantMemory = 64 * 1024; // 64KB constant memory
            WarpSize = 32; // Neural Engine execution units
            NumMultiprocessors = 16; // Estimate for Apple Silicon
            MaxNumThreadsPerMultiprocessor = 2048;
            // NumThreads is calculated automatically by MaxNumThreads property
            Capabilities = new ANECapabilityContext(capabilities);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the unified device identifier for this Apple Neural Engine device.
        /// </summary>
        public override DeviceId DeviceId => DeviceId.FromANE();

        #endregion

        #region Methods

        /// <inheritdoc/>
        public override Accelerator CreateAccelerator(Context context) =>
            new AppleNeuralEngineAccelerator(context, this);

        #endregion

        #region Object

        /// <inheritdoc/>
        public override bool Equals(object? obj) =>
            obj is AppleNeuralEngineDevice && base.Equals(obj);

        /// <inheritdoc/>
        public override int GetHashCode() =>
            HashCode.Combine(base.GetHashCode(), nameof(AppleNeuralEngineDevice));

        #endregion
    }

    /// <summary>
    /// Apple Neural Engine-specific capability context.
    /// </summary>
    /// <remarks>
    /// Initializes a new ANE capability context.
    /// </remarks>
    /// <param name="capabilities">The ANE capabilities.</param>
    internal sealed class ANECapabilityContext(ANECapabilities capabilities) : CapabilityContext
    {

        /// <summary>
        /// Gets the ANE capabilities.
        /// </summary>
        public ANECapabilities ANECapabilities { get; } = capabilities;

        // ANE has broad vector support across different data types
        // These capabilities are queried from the actual hardware
    }
}