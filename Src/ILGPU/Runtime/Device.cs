﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: Device.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Resources;
using ILGPU.Util;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Reflection;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Represents an abstract device object.
    /// </summary>
    public interface IDevice
    {
        #region Properties

        /// <summary>
        /// Returns the name of this device.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Returns the memory size in bytes.
        /// </summary>
        long MemorySize { get; }

        /// <summary>
        /// Returns the max grid size.
        /// </summary>
        Index3D MaxGridSize { get; }

        /// <summary>
        /// Returns the max group size.
        /// </summary>
        Index3D MaxGroupSize { get; }

        /// <summary>
        /// Returns the maximum number of threads in a group.
        /// </summary>
        int MaxNumThreadsPerGroup { get; }

        /// <summary>
        /// Returns the maximum number of shared memory per thread group in bytes.
        /// </summary>
        int MaxSharedMemoryPerGroup { get; }

        /// <summary>
        /// Returns the maximum number of constant memory in bytes.
        /// </summary>
        int MaxConstantMemory { get; }

        /// <summary>
        /// Return the warp size.
        /// </summary>
        int WarpSize { get; }

        /// <summary>
        /// Returns the number of available multiprocessors.
        /// </summary>
        int NumMultiprocessors { get; }

        /// <summary>
        /// Returns the maximum number of threads per multiprocessor.
        /// </summary>
        int MaxNumThreadsPerMultiprocessor { get; }

        /// <summary>
        /// Returns the maximum number of threads of this accelerator.
        /// </summary>
        int MaxNumThreads { get; }

        /// <summary>
        /// Returns the supported capabilities of this accelerator.
        /// </summary>
        CapabilityContext Capabilities { get; }

        /// <summary>
        /// Gets the current status of this device.
        /// </summary>
        DeviceStatus Status { get; }

        /// <summary>
        /// Gets enhanced memory information for this device.
        /// </summary>
        MemoryInfo Memory { get; }

        /// <summary>
        /// Gets a value indicating whether this device supports unified memory.
        /// </summary>
        bool SupportsUnifiedMemory { get; }

        /// <summary>
        /// Gets a value indicating whether this device supports memory pools.
        /// </summary>
        bool SupportsMemoryPools { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Prints device information to the given text writer.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        void PrintInformation(TextWriter writer);

        #endregion
    }

    /// <summary>
    /// Represents a single device object.
    /// </summary>
    /// <remarks>
    /// Note that all derived class have to be annotated with the
    /// <see cref="DeviceTypeAttribute"/> attribute.
    /// </remarks>
    public abstract class Device : IDevice, IAcceleratorBuilder, IDeviceIdentifiable
    {
        #region Instance

        /// <summary>
        /// Constructs a new device.
        /// </summary>
        protected Device()
        {
            AcceleratorType = DeviceTypeAttribute.GetAcceleratorType(GetType());

            // NB: Initialized later by derived classes.
            Capabilities = Utilities.InitNotNullable<CapabilityContext>();
        }

        /// <summary>
        /// Initializes memory information for this device.
        /// This should be called by derived classes after setting up device properties.
        /// </summary>
        protected virtual void InitializeMemoryInfo()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                Memory = new MemoryInfo(
                    totalMemory: MemorySize,
                    availableMemory: MemorySize, // Assume all memory is available initially
                    usedMemory: 0,
                    maxAllocationSize: Math.Min(MemorySize, MaxConstantMemory > 0 ? Math.Max(MemorySize / 4, MaxConstantMemory) : MemorySize / 4),
                    allocationGranularity: 256, // Common GPU memory alignment
                    supportsVirtualMemory: false, // Default to false, can be overridden
                    supportsMemoryMapping: false, // Default to false, can be overridden
                    supportsZeroCopy: AcceleratorType == AcceleratorType.CPU, // CPU supports zero-copy by default
                    cacheLineSize: AcceleratorType == AcceleratorType.CPU ? 64 : 128, // Different cache line sizes
                    memoryBandwidth: 0 // Unknown bandwidth, can be set by specific implementations
                );
            }
            catch
            {
                // If memory info creation fails, use unknown
                Memory = MemoryInfo.Unknown;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the type of the associated accelerator.
        /// </summary>
        public AcceleratorType AcceleratorType { get; }

        /// <summary>
        /// Gets the unique device identifier for this device.
        /// </summary>
        /// <remarks>
        /// This property provides a consistent way to identify devices across
        /// different accelerator types, enabling generic programming patterns
        /// and dependency injection scenarios.
        /// </remarks>
        public abstract DeviceId DeviceId { get; }

        /// <summary>
        /// Returns the name of this device.
        /// </summary>
        public string Name { get; protected set; } = "<Unknown>";

        /// <summary>
        /// Returns the memory size in bytes.
        /// </summary>
        public long MemorySize { get; protected set; }

        /// <summary>
        /// Returns the max grid size.
        /// </summary>
        public Index3D MaxGridSize { get; protected set; }

        /// <summary>
        /// Returns the max group size.
        /// </summary>
        public Index3D MaxGroupSize { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads in a group.
        /// </summary>
        public int MaxNumThreadsPerGroup { get; protected set; }

        /// <summary>
        /// Returns the maximum shared memory per thread group in bytes.
        /// </summary>
        public int MaxSharedMemoryPerGroup { get; protected set; }

        /// <summary>
        /// Returns the maximum number of constant memory in bytes.
        /// </summary>
        public int MaxConstantMemory { get; protected set; }

        /// <summary>
        /// Return the warp size.
        /// </summary>
        public int WarpSize { get; protected set; }

        /// <summary>
        /// Returns the number of available multiprocessors.
        /// </summary>
        public int NumMultiprocessors { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads per multiprocessor.
        /// </summary>
        public int MaxNumThreadsPerMultiprocessor { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads of this accelerator.
        /// </summary>
        public int MaxNumThreads => NumMultiprocessors * MaxNumThreadsPerMultiprocessor;

        /// <summary>
        /// Returns the supported capabilities of this device.
        /// </summary>
        public CapabilityContext Capabilities { get; protected set; }

        /// <summary>
        /// Gets the current status of this device.
        /// </summary>
        /// <remarks>
        /// This property provides real-time device state information enabling
        /// applications to make informed decisions about device usage and
        /// handle device state changes appropriately.
        /// </remarks>
        public virtual DeviceStatus Status { get; protected set; } = DeviceStatus.Available;

        /// <summary>
        /// Gets enhanced memory information for this device.
        /// </summary>
        /// <remarks>
        /// This property provides comprehensive memory statistics and capabilities
        /// information, enabling efficient memory management and allocation strategies.
        /// </remarks>
        public virtual MemoryInfo Memory { get; protected set; } = MemoryInfo.Unknown;

        /// <summary>
        /// Gets a value indicating whether this device supports unified memory.
        /// </summary>
        /// <remarks>
        /// Unified memory allows the GPU and CPU to share a single memory space,
        /// simplifying memory management and enabling automatic data migration
        /// between host and device memory.
        /// </remarks>
        public virtual bool SupportsUnifiedMemory { get; protected set; }

        /// <summary>
        /// Gets a value indicating whether this device supports memory pools.
        /// </summary>
        /// <remarks>
        /// Memory pools provide efficient buffer reuse and reduced allocation
        /// overhead through sophisticated caching and recycling strategies,
        /// significantly improving performance for frequent allocations.
        /// </remarks>
        public virtual bool SupportsMemoryPools { get; protected set; }

        /// <summary>
        /// Gets the compute capability version of this device.
        /// </summary>
        /// <remarks>
        /// Compute capability represents the feature set and computational capabilities
        /// of the device. Different accelerator types use different versioning schemes:
        /// - CUDA devices use major.minor versions (e.g., 8.6 for RTX 3080)
        /// - ROCm devices use major.minor versions based on GCN/RDNA architecture
        /// - Other accelerators may use their own versioning or return 1.0 as default
        /// </remarks>
        public virtual Version ComputeCapability { get; protected set; } = new(1, 0);

        /// <summary>
        /// Gets a value indicating whether this device supports cooperative kernel launches.
        /// </summary>
        /// <remarks>
        /// Cooperative kernels allow thread blocks to synchronize and cooperate during
        /// execution, enabling more sophisticated parallel algorithms. This feature is
        /// typically available on modern GPU architectures with compute capability 6.0+
        /// for CUDA devices and equivalent capabilities for other accelerator types.
        /// </remarks>
        public virtual bool SupportsCooperativeKernels { get; protected set; }

        #endregion

        #region Methods

        /// <summary>
        /// Creates a new accelerator instance.
        /// </summary>
        /// <param name="context">The context instance.</param>
        /// <returns>The created accelerator instance.</returns>
        public abstract Accelerator CreateAccelerator(Context context);

        /// <summary>
        /// Prints device information to the given text writer.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        public void PrintInformation(TextWriter writer)
        {
            if (writer is null)
                throw new ArgumentNullException(nameof(writer));

            PrintHeader(writer);
            PrintGeneralInfo(writer);
        }

        /// <summary>
        /// Prints general header information that should appear at the top.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        protected virtual void PrintHeader(TextWriter writer)
        {
            writer.Write("Device: ");
            writer.WriteLine(Name);
            writer.Write("  Accelerator Type:                        ");
            writer.WriteLine(AcceleratorType.ToString());
        }

        /// <summary>
        /// Print general GPU specific information to the given text writer.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        protected virtual void PrintGeneralInfo(TextWriter writer)
        {
            writer.Write("  Warp size:                               ");
            writer.WriteLine(WarpSize);

            writer.Write("  Number of multiprocessors:               ");
            writer.WriteLine(NumMultiprocessors);

            writer.Write("  Max number of threads/multiprocessor:    ");
            writer.WriteLine(MaxNumThreadsPerMultiprocessor);

            writer.Write("  Max number of threads/group:             ");
            writer.WriteLine(MaxNumThreadsPerGroup);

            writer.Write("  Max number of total threads:             ");
            writer.WriteLine(MaxNumThreads);

            writer.Write("  Max dimension of a group size:           ");
            writer.WriteLine(MaxGroupSize.ToString());

            writer.Write("  Max dimension of a grid size:            ");
            writer.WriteLine(MaxGridSize.ToString());

            writer.Write("  Total amount of global memory:           ");
            writer.WriteLine(
                "{0} bytes, {1} MB",
                MemorySize,
                MemorySize / (1024 * 1024));

            writer.Write("  Total amount of constant memory:         ");
            writer.WriteLine(
                "{0} bytes, {1} KB",
                MaxConstantMemory,
                MaxConstantMemory / 1024);

            writer.Write("  Total amount of shared memory per group: ");
            writer.WriteLine(
                "{0} bytes, {1} KB",
                MaxSharedMemoryPerGroup,
                MaxSharedMemoryPerGroup / 1024);
        }

        #endregion

        #region Object

        /// <summary>
        /// Returns true if the given object is equal to the current device.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>
        /// True, if the given object is equal to the current device.
        /// </returns>
        public override bool Equals(object? obj) =>
            obj is Device device &&
            device.AcceleratorType == AcceleratorType &&
            device.Name == Name;

        /// <summary>
        /// Returns the hash code of this device.
        /// </summary>
        /// <returns>The hash code of this device.</returns>
        public override int GetHashCode() => (int)AcceleratorType;

        /// <summary>
        /// Returns the string representation of this accelerator description.
        /// </summary>
        /// <returns>The string representation of this accelerator.</returns>
        public override string ToString() =>
            $"{Name} [Type: {AcceleratorType}, WarpSize: {WarpSize}, " +
            $"MaxNumThreadsPerGroup: {MaxNumThreadsPerGroup}, " +
            $"MemorySize: {MemorySize}]";

        #endregion
    }

    /// <summary>
    /// Annotates classes derived from <see cref="Device"/> with their accelerator type.
    /// </summary>
    /// <remarks>
    /// This attribute is used to automatically identify the accelerator type for device classes,
    /// enabling proper device registration and type identification in the ILGPU context.
    /// </remarks>
    /// <param name="acceleratorType">The accelerator type of the annotated device.</param>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = true)]
    public sealed class DeviceTypeAttribute(AcceleratorType acceleratorType) : Attribute
    {
        /// <summary>
        /// Gets the accelerator type of the given device class.
        /// </summary>
        /// <param name="type">The device class type.</param>
        /// <returns>The accelerator type.</returns>
        public static AcceleratorType GetAcceleratorType(Type type)
        {
            if (type is null)
                throw new ArgumentNullException(nameof(type));

            var attribute = type.GetCustomAttribute<DeviceTypeAttribute>();
            return attribute is null
                ? throw new InvalidOperationException(
                    RuntimeErrorMessages.InvalidDeviceTypeAttribute)
                : attribute.AcceleratorType;
        }

        /// <summary>
        /// Returns the associated accelerator type.
        /// </summary>
        public AcceleratorType AcceleratorType { get; } = acceleratorType;
    }

    /// <summary>
    /// Extension methods for devices.
    /// </summary>
    public static class DeviceExtensions
    {
        /// <summary>
        /// Prints device information to the standard <see cref="Console.Out"/> stream.
        /// </summary>
        /// <param name="device">The device to print.</param>
        public static void PrintInformation(this IDevice device) =>
            device.PrintInformation(Console.Out);
    }

    /// <summary>
    /// A registry for device instances to avoid duplicate registrations.
    /// </summary>
    public sealed class DeviceRegistry
    {
        #region Instance

        /// <summary>
        /// The set of all registered devices.
        /// </summary>
        private readonly HashSet<Device> registered =
            [];

        /// <summary>
        /// Stores all registered accelerator device objects.
        /// </summary>
        private readonly
            ImmutableArray<Device>.Builder
            devices =
            ImmutableArray.CreateBuilder<Device>(8);

        #endregion

        #region Properties

        /// <summary>
        /// Returns the number of registered devices.
        /// </summary>
        public int Count => devices.Count;

        #endregion

        #region Methods

        /// <summary>
        /// Registers the given device.
        /// </summary>
        /// <param name="device">The device to register.</param>
        public void Register(Device device)
        {
            if (device is null)
                throw new ArgumentNullException(nameof(device));
            if (!registered.Add(device))
                return;

            devices.Add(device);
        }

        /// <summary>
        /// Registers the given device if the predicate evaluates to true.
        /// </summary>
        /// <typeparam name="TDevice">The device class type.</typeparam>
        /// <param name="device">The device to register.</param>
        /// <param name="predicate">
        /// The device predicate to check whether to include the device or not.
        /// </param>
        public void Register<TDevice>(TDevice device, Predicate<TDevice> predicate)
            where TDevice : Device
        {
            if (device is null)
                throw new ArgumentNullException(nameof(device));
            if (predicate is null)
                throw new ArgumentNullException(nameof(predicate));

            if (predicate(device))
                Register(device);
        }

        /// <summary>
        /// Converts this registry into an immutable array.
        /// </summary>
        /// <returns>The created immutable array of devices.</returns>
        public ImmutableArray<Device> ToImmutable() => devices.ToImmutable();

        #endregion
    }
}
