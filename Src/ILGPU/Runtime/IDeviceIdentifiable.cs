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

namespace ILGPU.Runtime
{
    /// <summary>
    /// Represents a unified device identification interface for all ILGPU devices.
    /// This interface enables consistent device identification across different
    /// accelerator types (CUDA, OpenCL, CPU, Velocity).
    /// </summary>
    /// <remarks>
    /// This interface addresses the critical issue where different device types
    /// use different identification mechanisms, making generic programming difficult.
    /// </remarks>
    public interface IDeviceIdentifiable
    {
        /// <summary>
        /// Gets the unique device identifier for this device.
        /// </summary>
        /// <remarks>
        /// This property provides a consistent way to identify devices across
        /// different accelerator types:
        /// - For CUDA devices: Returns the native CUDA device ID
        /// - For OpenCL devices: Returns a computed hash of platform and device IDs
        /// - For CPU/Velocity devices: Returns a synthetic ID based on configuration
        /// </remarks>
        DeviceId DeviceId { get; }

        /// <summary>
        /// Gets the accelerator type for this device.
        /// </summary>
        AcceleratorType AcceleratorType { get; }
    }

    /// <summary>
    /// Represents a unified device identifier that works across all accelerator types.
    /// </summary>
    /// <remarks>
    /// This structure provides a consistent way to identify and compare devices
    /// regardless of their underlying accelerator type, enabling generic programming
    /// patterns and dependency injection scenarios.
    /// </remarks>
    /// <remarks>
    /// Initializes a new device ID.
    /// </remarks>
    /// <param name="value">The device identifier value.</param>
    /// <param name="acceleratorType">The accelerator type.</param>
    public readonly struct DeviceId(long value, AcceleratorType acceleratorType) : IEquatable<DeviceId>, IComparable<DeviceId>
    {
        /// <summary>
        /// The underlying device identifier value.
        /// </summary>
        public long Value { get; } = value;

        /// <summary>
        /// The accelerator type this device ID belongs to.
        /// </summary>
        public AcceleratorType AcceleratorType { get; } = acceleratorType;

        /// <summary>
        /// Creates a device ID from a CUDA device index.
        /// </summary>
        /// <param name="cudaDeviceId">The CUDA device ID.</param>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromCuda(int cudaDeviceId) =>
            new(cudaDeviceId, AcceleratorType.Cuda);

        /// <summary>
        /// Creates a device ID from OpenCL platform and device pointers.
        /// </summary>
        /// <param name="platformId">The OpenCL platform ID.</param>
        /// <param name="deviceId">The OpenCL device ID.</param>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromOpenCL(IntPtr platformId, IntPtr deviceId)
        {
            // Combine platform and device IDs into a single value
            var combined = ((long)platformId.ToInt64() << 32) | (uint)deviceId.ToInt64();
            return new DeviceId(combined, AcceleratorType.OpenCL);
        }

        /// <summary>
        /// Creates a device ID for a CPU device.
        /// </summary>
        /// <param name="configuration">
        /// A hash representing the CPU device configuration.
        /// </param>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromCPU(int configuration) =>
            new(configuration, AcceleratorType.CPU);

        /// <summary>
        /// Creates a device ID for a Velocity device.
        /// </summary>
        /// <param name="configuration">
        /// A hash representing the Velocity device configuration.
        /// </param>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromVelocity(int configuration) =>
            new(configuration, AcceleratorType.Velocity);

        /// <summary>
        /// Creates a device ID for an Intel AMX device.
        /// </summary>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromAMX() =>
            new(0, AcceleratorType.IntelAMX);

        /// <summary>
        /// Creates a device ID for an Apple Neural Engine device.
        /// </summary>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromANE() =>
            new(0, AcceleratorType.AppleNeuralEngine);

        /// <summary>
        /// Creates a device ID for an Intel NPU device.
        /// </summary>
        /// <returns>A unified device ID.</returns>
        public static DeviceId FromNPU() =>
            new(0, AcceleratorType.IntelNPU);

        /// <summary>
        /// Gets a value indicating whether this device ID represents a CUDA device.
        /// </summary>
        public bool IsCuda => AcceleratorType == AcceleratorType.Cuda;

        /// <summary>
        /// Gets a value indicating whether this device ID represents an OpenCL device.
        /// </summary>
        public bool IsOpenCL => AcceleratorType == AcceleratorType.OpenCL;

        /// <summary>
        /// Gets a value indicating whether this device ID represents a CPU device.
        /// </summary>
        public bool IsCPU => AcceleratorType == AcceleratorType.CPU;

        /// <summary>
        /// Gets a value indicating whether this device ID represents a Velocity device.
        /// </summary>
        public bool IsVelocity => AcceleratorType == AcceleratorType.Velocity;

        /// <summary>
        /// Gets a value indicating whether this device ID represents an Intel AMX device.
        /// </summary>
        public bool IsIntelAMX => AcceleratorType == AcceleratorType.IntelAMX;

        /// <summary>
        /// Gets a value indicating whether this device ID represents an Apple Neural Engine device.
        /// </summary>
        public bool IsAppleNeuralEngine => AcceleratorType == AcceleratorType.AppleNeuralEngine;

        /// <summary>
        /// Gets a value indicating whether this device ID represents an Intel NPU device.
        /// </summary>
        public bool IsIntelNPU => AcceleratorType == AcceleratorType.IntelNPU;

        /// <summary>
        /// Gets the CUDA device ID if this is a CUDA device.
        /// </summary>
        /// <returns>The CUDA device ID.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if this is not a CUDA device.
        /// </exception>
        public int ToCudaDeviceId()
        {
            if (!IsCuda)
                throw new InvalidOperationException("DeviceId is not a CUDA device");
            return (int)Value;
        }

        /// <summary>
        /// Gets the OpenCL platform and device IDs if this is an OpenCL device.
        /// </summary>
        /// <returns>The platform and device ID pair.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if this is not an OpenCL device.
        /// </exception>
        public (IntPtr PlatformId, IntPtr DeviceId) ToOpenCLIds()
        {
            if (!IsOpenCL)
                throw new InvalidOperationException("DeviceId is not an OpenCL device");
            
            var platformId = new IntPtr(Value >> 32);
            var deviceId = new IntPtr((int)(Value & 0xFFFFFFFF));
            return (platformId, deviceId);
        }

        #region IEquatable

        /// <inheritdoc/>
        public bool Equals(DeviceId other) =>
            Value == other.Value && AcceleratorType == other.AcceleratorType;

        /// <inheritdoc/>
        public override bool Equals(object? obj) =>
            obj is DeviceId other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode() =>
            HashCode.Combine(Value, AcceleratorType);

        /// <summary>
        /// Compares two device IDs for equality.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if the device IDs are equal.</returns>
        public static bool operator ==(DeviceId left, DeviceId right) =>
            left.Equals(right);

        /// <summary>
        /// Compares two device IDs for inequality.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if the device IDs are not equal.</returns>
        public static bool operator !=(DeviceId left, DeviceId right) =>
            !left.Equals(right);

        #endregion

        #region IComparable

        /// <inheritdoc/>
        public int CompareTo(DeviceId other)
        {
            var typeComparison = AcceleratorType.CompareTo(other.AcceleratorType);
            return typeComparison != 0 ? typeComparison : Value.CompareTo(other.Value);
        }

        /// <summary>
        /// Compares two device IDs.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if left is less than right.</returns>
        public static bool operator <(DeviceId left, DeviceId right) =>
            left.CompareTo(right) < 0;

        /// <summary>
        /// Compares two device IDs.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if left is greater than right.</returns>
        public static bool operator >(DeviceId left, DeviceId right) =>
            left.CompareTo(right) > 0;

        /// <summary>
        /// Compares two device IDs.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if left is less than or equal to right.</returns>
        public static bool operator <=(DeviceId left, DeviceId right) =>
            left.CompareTo(right) <= 0;

        /// <summary>
        /// Compares two device IDs.
        /// </summary>
        /// <param name="left">The left device ID.</param>
        /// <param name="right">The right device ID.</param>
        /// <returns>True if left is greater than or equal to right.</returns>
        public static bool operator >=(DeviceId left, DeviceId right) =>
            left.CompareTo(right) >= 0;

        #endregion

        /// <inheritdoc/>
        public override string ToString() =>
            $"{AcceleratorType}:{Value}";
    }
}
