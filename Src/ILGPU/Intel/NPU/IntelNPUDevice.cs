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

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Represents an Intel Neural Processing Unit (NPU) device.
    /// </summary>
    [DeviceType(AcceleratorType.IntelNPU)]
    public sealed class IntelNPUDevice : Device
    {
        #region Static

        /// <summary>
        /// The default Intel NPU device for systems that support NPU.
        /// </summary>
        public static readonly IntelNPUDevice Default = new();

        /// <summary>
        /// Detects and enumerates all available Intel NPU devices.
        /// </summary>
        /// <param name="registry">The device registry.</param>
        public static void RegisterDevices(DeviceRegistry registry)
        {
            if (!NPUCapabilities.DetectNPU())
                return;

            registry.Register(Default);
        }

        #endregion

        #region Instance

        /// <summary>
        /// Initializes a new Intel NPU device instance.
        /// </summary>
        private IntelNPUDevice()
        {
            if (!NPUCapabilities.DetectNPU())
                throw new NotSupportedException("Intel NPU not supported on this processor");

            var capabilities = NPUCapabilities.Query();
            
            Name = $"Intel NPU {capabilities.Generation} ({capabilities.MaxTOPS:F1} TOPS)";
            MemorySize = GC.GetTotalMemory(false); // Use system memory for now
            MaxGridSize = new Index3D(65535, 65535, 65535);
            MaxGroupSize = new Index3D(1024, 1024, 1024);
            MaxNumThreadsPerGroup = 1024;
            MaxSharedMemoryPerGroup = 64 * 1024; // 64KB shared memory
            MaxConstantMemory = 32 * 1024; // 32KB constant memory
            WarpSize = 32; // NPU execution unit size
            NumMultiprocessors = capabilities.ComputeUnits;
            MaxNumThreadsPerMultiprocessor = 1024;
            // NumThreads is calculated automatically from NumMultiprocessors * MaxNumThreadsPerMultiprocessor
            Capabilities = new NPUCapabilityContext(capabilities);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the unified device identifier for this Intel NPU device.
        /// </summary>
        public override DeviceId DeviceId => DeviceId.FromNPU();

        #endregion

        #region Methods

        /// <inheritdoc/>
        public override Accelerator CreateAccelerator(Context context) =>
            new IntelNPUAccelerator(context, this);

        #endregion

        #region Object

        /// <inheritdoc/>
        public override bool Equals(object? obj) =>
            obj is IntelNPUDevice && base.Equals(obj);

        /// <inheritdoc/>
        public override int GetHashCode() =>
            HashCode.Combine(base.GetHashCode(), nameof(IntelNPUDevice));

        #endregion
    }

    /// <summary>
    /// Intel NPU-specific capability context.
    /// </summary>
    internal sealed class NPUCapabilityContext : CapabilityContext
    {
        /// <summary>
        /// Initializes a new NPU capability context.
        /// </summary>
        /// <param name="capabilities">The NPU capabilities.</param>
        public NPUCapabilityContext(NPUCapabilities capabilities)
        {
            NPUCapabilities = capabilities;
        }

        /// <summary>
        /// Gets the NPU capabilities.
        /// </summary>
        public NPUCapabilities NPUCapabilities { get; }

    }
}