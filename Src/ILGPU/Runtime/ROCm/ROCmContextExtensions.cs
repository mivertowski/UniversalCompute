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

using ILGPU.Backends;
using ILGPU.Resources;
using System;
using System.Diagnostics.CodeAnalysis;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// ROCm-specific context extensions.
    /// </summary>
    public static class ROCmContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables all compatible ROCm devices.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder ROCm(this Context.Builder builder) =>
            builder.ROCm(device => true);

        /// <summary>
        /// Enables all ROCm devices that fulfill the given predicate.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <param name="predicate">The predicate to include a given device.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder ROCm(
            this Context.Builder builder,
            Predicate<ROCmDevice> predicate) =>
            builder.ROCmInternal(predicate);

        /// <summary>
        /// Internal ROCm device registration.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <param name="predicate">The predicate to include a given device.</param>
        /// <returns>The updated builder instance.</returns>
        private static Context.Builder ROCmInternal(
            this Context.Builder builder,
            Predicate<ROCmDevice> predicate)
        {
            if (!ROCmDevice.IsSupported())
                return builder;

            foreach (var device in ROCmDevice.GetDevices())
            {
                if (predicate(device))
                {
                    builder.Accelerators.Add(new ROCmAcceleratorBuilder(device));
                }
            }

            return builder;
        }

        #endregion

        #region Context

        /// <summary>
        /// Creates a new ROCm accelerator using the default device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created accelerator.</returns>
        public static ROCmAccelerator CreateROCmAccelerator(this Context context) =>
            CreateROCmAccelerator(context, 0);

        /// <summary>
        /// Creates a new ROCm accelerator using the specified device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="deviceId">The device ID.</param>
        /// <returns>The created accelerator.</returns>
        public static ROCmAccelerator CreateROCmAccelerator(this Context context, int deviceId)
        {
            if (!ROCmDevice.IsSupported())
                throw new NotSupportedException("ROCm is not supported on this system");

            var devices = ROCmDevice.GetDevices();
            if (deviceId < 0 || deviceId >= devices.Length)
                throw new ArgumentOutOfRangeException(nameof(deviceId), $"Device ID {deviceId} is out of range [0, {devices.Length - 1}]");

            return new ROCmAccelerator(context, devices[deviceId]);
        }

        /// <summary>
        /// Creates a new ROCm accelerator using the specified device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The ROCm device.</param>
        /// <returns>The created accelerator.</returns>
        public static ROCmAccelerator CreateROCmAccelerator(this Context context, ROCmDevice device)
        {
            if (device == null)
                throw new ArgumentNullException(nameof(device));

            return new ROCmAccelerator(context, device);
        }

        /// <summary>
        /// Gets all available ROCm devices.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>All available ROCm devices.</returns>
        public static ROCmDevice[] GetROCmDevices(this Context context) =>
            ROCmDevice.GetDevices();

        /// <summary>
        /// Gets the default ROCm device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The default ROCm device or null if none available.</returns>
        public static ROCmDevice? GetDefaultROCmDevice(this Context context) =>
            ROCmDevice.GetDefaultDevice();

        /// <summary>
        /// Checks if ROCm is supported on this system.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>True if ROCm is supported; otherwise, false.</returns>
        public static bool IsROCmSupported(this Context context) =>
            ROCmDevice.IsSupported();

        #endregion
    }

    /// <summary>
    /// ROCm accelerator builder implementation.
    /// </summary>
    internal sealed class ROCmAcceleratorBuilder : IAcceleratorBuilder
    {
        /// <summary>
        /// The associated ROCm device.
        /// </summary>
        private readonly ROCmDevice device;

        /// <summary>
        /// Initializes a new ROCm accelerator builder.
        /// </summary>
        /// <param name="device">The ROCm device.</param>
        public ROCmAcceleratorBuilder(ROCmDevice device)
        {
            this.device = device ?? throw new ArgumentNullException(nameof(device));
        }

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public AcceleratorType AcceleratorType => AcceleratorType.ROCm;

        /// <summary>
        /// Creates a new accelerator instance.
        /// </summary>
        /// <param name="context">The context instance.</param>
        /// <returns>The created accelerator instance.</returns>
        public Accelerator CreateAccelerator(Context context) =>
            new ROCmAccelerator(context, device);
    }

    /// <summary>
    /// Backend extensions for ROCm.
    /// </summary>
    public static class ROCmBackendExtensions
    {
        /// <summary>
        /// Creates a new ROCm backend.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="instruction">The instruction set.</param>
        /// <param name="capabilities">The device capabilities.</param>
        /// <returns>The created backend.</returns>
        public static Backend CreateROCmBackend(
            this Context context,
            ROCmInstructionSet instruction,
            ROCmCapabilities capabilities)
        {
            return new ROCmBackend(context, instruction, capabilities);
        }
    }

    /// <summary>
    /// ROCm instruction set enumeration.
    /// </summary>
    public enum ROCmInstructionSet
    {
        /// <summary>
        /// GCN 3.0 (Fiji/Polaris).
        /// </summary>
        GCN3,

        /// <summary>
        /// GCN 4.0 (Vega).
        /// </summary>
        GCN4,

        /// <summary>
        /// GCN 5.0 (Vega II).
        /// </summary>
        GCN5,

        /// <summary>
        /// RDNA 1.0 (Navi 10).
        /// </summary>
        RDNA1,

        /// <summary>
        /// RDNA 2.0 (Navi 2x).
        /// </summary>
        RDNA2,

        /// <summary>
        /// RDNA 3.0 (Navi 3x).
        /// </summary>
        RDNA3,

        /// <summary>
        /// RDNA 4.0 (Navi 4x).
        /// </summary>
        RDNA4
    }

    /// <summary>
    /// ROCm device capabilities.
    /// </summary>
    public readonly struct ROCmCapabilities
    {
        /// <summary>
        /// The compute capability.
        /// </summary>
        public Version ComputeCapability { get; }

        /// <summary>
        /// The instruction set.
        /// </summary>
        public ROCmInstructionSet InstructionSet { get; }

        /// <summary>
        /// Whether the device supports unified memory.
        /// </summary>
        public bool SupportsUnifiedMemory { get; }

        /// <summary>
        /// Whether the device supports managed memory.
        /// </summary>
        public bool SupportsManagedMemory { get; }

        /// <summary>
        /// Whether the device supports cooperative groups.
        /// </summary>
        public bool SupportsCooperativeGroups { get; }

        /// <summary>
        /// Initializes ROCm capabilities.
        /// </summary>
        /// <param name="computeCapability">The compute capability.</param>
        /// <param name="instructionSet">The instruction set.</param>
        /// <param name="supportsUnifiedMemory">Whether unified memory is supported.</param>
        /// <param name="supportsManagedMemory">Whether managed memory is supported.</param>
        /// <param name="supportsCooperativeGroups">Whether cooperative groups are supported.</param>
        public ROCmCapabilities(
            Version computeCapability,
            ROCmInstructionSet instructionSet,
            bool supportsUnifiedMemory,
            bool supportsManagedMemory,
            bool supportsCooperativeGroups)
        {
            ComputeCapability = computeCapability;
            InstructionSet = instructionSet;
            SupportsUnifiedMemory = supportsUnifiedMemory;
            SupportsManagedMemory = supportsManagedMemory;
            SupportsCooperativeGroups = supportsCooperativeGroups;
        }
    }
}