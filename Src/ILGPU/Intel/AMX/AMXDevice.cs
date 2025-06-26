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
// Change License: Apache License, Version 2.0using ILGPU.Runtime;
using System;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Represents an Intel Advanced Matrix Extensions (AMX) device.
    /// </summary>
    [DeviceType(AcceleratorType.IntelAMX)]
    public sealed class AMXDevice : Device
    {
        #region Static

        /// <summary>
        /// The default AMX device for systems that support AMX.
        /// </summary>
        public static readonly AMXDevice Default = new();

        /// <summary>
        /// Detects and enumerates all available AMX devices.
        /// </summary>
        /// <param name="registry">The device registry.</param>
        public static void RegisterDevices(DeviceRegistry registry)
        {
            if (!AMXCapabilities.IsAMXSupported())
                return;

            registry.Register(
                Default,
                device => AMXCapabilities.IsAMXSupported());
        }

        #endregion

        #region Instance

        /// <summary>
        /// Initializes a new AMX device instance.
        /// </summary>
        private AMXDevice()
        {
            if (!AMXCapabilities.IsAMXSupported())
                throw new NotSupportedException("Intel AMX not supported on this processor");

            var capabilities = AMXCapabilities.Query();
            
            Name = "Intel AMX Accelerator";
            MemorySize = GC.GetTotalMemory(false); // Use system memory
            MaxGridSize = new Index3D(65535, 65535, 65535);
            MaxGroupSize = new Index3D(1024, 1024, 1024);
            MaxNumThreadsPerGroup = 1024;
            MaxSharedMemoryPerGroup = capabilities.MaxTileBytes;
            MaxConstantMemory = capabilities.MaxConfigBytes;
            WarpSize = 16; // AMX operates on 16x64 byte tiles
            NumMultiprocessors = Environment.ProcessorCount;
            MaxNumThreadsPerMultiprocessor = 1024;
            // NumThreads is calculated automatically by MaxNumThreads property
            Capabilities = new AMXCapabilityContext(capabilities);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the unified device identifier for this AMX device.
        /// </summary>
        public override DeviceId DeviceId => DeviceId.FromAMX();

        #endregion

        #region Methods

        /// <inheritdoc/>
        public override Accelerator CreateAccelerator(Context context) =>
            new AMXAccelerator(context, this);

        #endregion

        #region Object

        /// <inheritdoc/>
        public override bool Equals(object? obj) =>
            obj is AMXDevice && base.Equals(obj);

        /// <inheritdoc/>
        public override int GetHashCode() =>
            HashCode.Combine(base.GetHashCode(), nameof(AMXDevice));

        #endregion
    }

    /// <summary>
    /// AMX-specific capability context.
    /// </summary>
    internal sealed class AMXCapabilityContext : CapabilityContext
    {
        /// <summary>
        /// Initializes a new AMX capability context.
        /// </summary>
        /// <param name="capabilities">The AMX capabilities.</param>
        public AMXCapabilityContext(AMXCapabilities capabilities)
        {
            AMXCapabilities = capabilities;
        }

        /// <summary>
        /// Gets the AMX capabilities.
        /// </summary>
        public AMXCapabilities AMXCapabilities { get; }

        // AMX has tile-based matrix operation capabilities
        // These capabilities are determined by the actual hardware
    }
}