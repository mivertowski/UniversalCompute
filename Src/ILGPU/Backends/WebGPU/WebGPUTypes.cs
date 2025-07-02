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
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.JavaScript;
using System.Threading.Tasks;

namespace ILGPU.Backends.WebGPU
{
    /// <summary>
    /// WebGPU buffer usage flags.
    /// </summary>
    [Flags]
    public enum WebGPUBufferUsage : uint
    {
        /// <summary>
        /// Buffer can be mapped for reading.
        /// </summary>
        MapRead = 1,

        /// <summary>
        /// Buffer can be mapped for writing.
        /// </summary>
        MapWrite = 2,

        /// <summary>
        /// Buffer can be used as copy source.
        /// </summary>
        CopySrc = 4,

        /// <summary>
        /// Buffer can be used as copy destination.
        /// </summary>
        CopyDst = 8,

        /// <summary>
        /// Buffer can be used as index buffer.
        /// </summary>
        Index = 16,

        /// <summary>
        /// Buffer can be used as vertex buffer.
        /// </summary>
        Vertex = 32,

        /// <summary>
        /// Buffer can be used as uniform buffer.
        /// </summary>
        Uniform = 64,

        /// <summary>
        /// Buffer can be used as storage buffer.
        /// </summary>
        Storage = 128,

        /// <summary>
        /// Buffer can be used as indirect buffer.
        /// </summary>
        Indirect = 256,

        /// <summary>
        /// Buffer can be used as query resolve buffer.
        /// </summary>
        QueryResolve = 512
    }

    /// <summary>
    /// WebGPU adapter types.
    /// </summary>
    public enum WebGPUAdapterType
    {
        /// <summary>
        /// Discrete GPU adapter.
        /// </summary>
        DiscreteGPU,

        /// <summary>
        /// Integrated GPU adapter.
        /// </summary>
        IntegratedGPU,

        /// <summary>
        /// CPU adapter (software rendering).
        /// </summary>
        CPU,

        /// <summary>
        /// Unknown adapter type.
        /// </summary>
        Unknown
    }

    /// <summary>
    /// WebGPU power preference.
    /// </summary>
    public enum WebGPUPowerPreference
    {
        /// <summary>
        /// No power preference.
        /// </summary>
        Undefined,

        /// <summary>
        /// Low power preference.
        /// </summary>
        LowPower,

        /// <summary>
        /// High performance preference.
        /// </summary>
        HighPerformance
    }

    /// <summary>
    /// WebGPU adapter information.
    /// </summary>
    public sealed class WebGPUAdapterInfo
    {
        /// <summary>
        /// Gets the adapter vendor.
        /// </summary>
        public string Vendor { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the adapter architecture.
        /// </summary>
        public string Architecture { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the adapter device name.
        /// </summary>
        public string Device { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the adapter description.
        /// </summary>
        public string Description { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the adapter type.
        /// </summary>
        public WebGPUAdapterType AdapterType { get; internal set; } = WebGPUAdapterType.Unknown;
    }

    /// <summary>
    /// WebGPU device capabilities.
    /// </summary>
    public sealed class WebGPUCapabilities
    {
        /// <summary>
        /// Gets the maximum buffer size.
        /// </summary>
        public ulong MaxBufferSize { get; internal set; }

        /// <summary>
        /// Gets the maximum texture dimension 1D.
        /// </summary>
        public uint MaxTextureDimension1D { get; internal set; }

        /// <summary>
        /// Gets the maximum texture dimension 2D.
        /// </summary>
        public uint MaxTextureDimension2D { get; internal set; }

        /// <summary>
        /// Gets the maximum texture dimension 3D.
        /// </summary>
        public uint MaxTextureDimension3D { get; internal set; }

        /// <summary>
        /// Gets the maximum texture array layers.
        /// </summary>
        public uint MaxTextureArrayLayers { get; internal set; }

        /// <summary>
        /// Gets the maximum bind groups.
        /// </summary>
        public uint MaxBindGroups { get; internal set; }

        /// <summary>
        /// Gets the maximum bindings per bind group.
        /// </summary>
        public uint MaxBindingsPerBindGroup { get; internal set; }

        /// <summary>
        /// Gets the maximum dynamic uniform buffers per pipeline layout.
        /// </summary>
        public uint MaxDynamicUniformBuffersPerPipelineLayout { get; internal set; }

        /// <summary>
        /// Gets the maximum dynamic storage buffers per pipeline layout.
        /// </summary>
        public uint MaxDynamicStorageBuffersPerPipelineLayout { get; internal set; }

        /// <summary>
        /// Gets the maximum sampled textures per shader stage.
        /// </summary>
        public uint MaxSampledTexturesPerShaderStage { get; internal set; }

        /// <summary>
        /// Gets the maximum samplers per shader stage.
        /// </summary>
        public uint MaxSamplersPerShaderStage { get; internal set; }

        /// <summary>
        /// Gets the maximum storage buffers per shader stage.
        /// </summary>
        public uint MaxStorageBuffersPerShaderStage { get; internal set; }

        /// <summary>
        /// Gets the maximum storage textures per shader stage.
        /// </summary>
        public uint MaxStorageTexturesPerShaderStage { get; internal set; }

        /// <summary>
        /// Gets the maximum uniform buffers per shader stage.
        /// </summary>
        public uint MaxUniformBuffersPerShaderStage { get; internal set; }

        /// <summary>
        /// Gets the maximum uniform buffer binding size.
        /// </summary>
        public ulong MaxUniformBufferBindingSize { get; internal set; }

        /// <summary>
        /// Gets the maximum storage buffer binding size.
        /// </summary>
        public ulong MaxStorageBufferBindingSize { get; internal set; }

        /// <summary>
        /// Gets the minimum uniform buffer offset alignment.
        /// </summary>
        public uint MinUniformBufferOffsetAlignment { get; internal set; }

        /// <summary>
        /// Gets the minimum storage buffer offset alignment.
        /// </summary>
        public uint MinStorageBufferOffsetAlignment { get; internal set; }

        /// <summary>
        /// Gets the maximum vertex buffers.
        /// </summary>
        public uint MaxVertexBuffers { get; internal set; }

        /// <summary>
        /// Gets the maximum vertex attributes.
        /// </summary>
        public uint MaxVertexAttributes { get; internal set; }

        /// <summary>
        /// Gets the maximum vertex buffer array stride.
        /// </summary>
        public uint MaxVertexBufferArrayStride { get; internal set; }

        /// <summary>
        /// Gets the maximum inter-stage shader components.
        /// </summary>
        public uint MaxInterStageShaderComponents { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroup storage size.
        /// </summary>
        public uint MaxComputeWorkgroupStorageSize { get; internal set; }

        /// <summary>
        /// Gets the maximum compute invocations per workgroup.
        /// </summary>
        public uint MaxComputeInvocationsPerWorkgroup { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroup size X.
        /// </summary>
        public uint MaxComputeWorkgroupSizeX { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroup size Y.
        /// </summary>
        public uint MaxComputeWorkgroupSizeY { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroup size Z.
        /// </summary>
        public uint MaxComputeWorkgroupSizeZ { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroups per dimension.
        /// </summary>
        public uint MaxComputeWorkgroupsPerDimension { get; internal set; }

        /// <summary>
        /// Gets the maximum workgroup size.
        /// </summary>
        public uint MaxWorkgroupSize => MaxComputeInvocationsPerWorkgroup;

        /// <summary>
        /// Gets the maximum shared memory size.
        /// </summary>
        public int MaxSharedMemorySize => (int)MaxComputeWorkgroupStorageSize;

        /// <summary>
        /// Gets the estimated memory bandwidth in GB/s.
        /// </summary>
        public double EstimatedMemoryBandwidth { get; internal set; }

        /// <summary>
        /// Queries WebGPU capabilities from the specified device.
        /// </summary>
        /// <param name="device">WebGPU device.</param>
        /// <returns>WebGPU capabilities.</returns>
        public static WebGPUCapabilities Query(WebGPUDevice device)
        {
            // Default WebGPU limits as per specification
            return new WebGPUCapabilities
            {
                MaxBufferSize = 268_435_456, // 256 MB
                MaxTextureDimension1D = 8192,
                MaxTextureDimension2D = 8192,
                MaxTextureDimension3D = 2048,
                MaxTextureArrayLayers = 256,
                MaxBindGroups = 4,
                MaxBindingsPerBindGroup = 1000,
                MaxDynamicUniformBuffersPerPipelineLayout = 8,
                MaxDynamicStorageBuffersPerPipelineLayout = 4,
                MaxSampledTexturesPerShaderStage = 16,
                MaxSamplersPerShaderStage = 16,
                MaxStorageBuffersPerShaderStage = 8,
                MaxStorageTexturesPerShaderStage = 4,
                MaxUniformBuffersPerShaderStage = 12,
                MaxUniformBufferBindingSize = 65536, // 64 KB
                MaxStorageBufferBindingSize = 134_217_728, // 128 MB
                MinUniformBufferOffsetAlignment = 256,
                MinStorageBufferOffsetAlignment = 256,
                MaxVertexBuffers = 8,
                MaxVertexAttributes = 16,
                MaxVertexBufferArrayStride = 2048,
                MaxInterStageShaderComponents = 60,
                MaxComputeWorkgroupStorageSize = 16384, // 16 KB
                MaxComputeInvocationsPerWorkgroup = 256,
                MaxComputeWorkgroupSizeX = 256,
                MaxComputeWorkgroupSizeY = 256,
                MaxComputeWorkgroupSizeZ = 64,
                MaxComputeWorkgroupsPerDimension = 65535,
                EstimatedMemoryBandwidth = 100.0 // Conservative estimate for web environments
            };
        }
    }

    /// <summary>
    /// WebGPU adapter wrapper.
    /// </summary>
    public sealed class WebGPUAdapter : IDisposable
    {
        private JSObject? _jsAdapter;
        private bool _disposed;

        internal WebGPUAdapter(JSObject jsAdapter)
        {
            _jsAdapter = jsAdapter;
            Info = QueryAdapterInfo();
        }

        /// <summary>
        /// Gets the adapter information.
        /// </summary>
        public WebGPUAdapterInfo Info { get; private set; }

        /// <summary>
        /// Requests a WebGPU adapter.
        /// </summary>
        /// <param name="powerPreference">Power preference.</param>
        /// <returns>WebGPU adapter or null if not available.</returns>
        public static async Task<WebGPUAdapter?> RequestAdapterAsync(
            WebGPUPowerPreference powerPreference = WebGPUPowerPreference.Undefined)
        {
#if BROWSER
            try
            {
                var gpu = JSHost.GlobalThis.GetPropertyAsJSObject("navigator")
                    ?.GetPropertyAsJSObject("gpu");
                
                if (gpu == null) return null;

                // Create adapter request options
                var options = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                if (powerPreference != WebGPUPowerPreference.Undefined)
                {
                    options.SetProperty("powerPreference", powerPreference.ToString().ToLowerInvariant());
                }

                // Request adapter
                var adapterPromise = gpu.Invoke("requestAdapter", options);
                var adapter = await JSHost.ImportAsync("adapter_helper", "./adapter_helper.js")
                    .ContinueWith(m => m.Result.Invoke("awaitAdapter", adapterPromise));

                return adapter != null ? new WebGPUAdapter(adapter) : null;
            }
            catch
            {
                return null;
            }
#else
            // Non-browser implementation would use WebGPU native libraries
            await Task.Delay(1);
            return null;
#endif
        }

        /// <summary>
        /// Enumerates all available WebGPU adapters.
        /// </summary>
        /// <returns>Array of WebGPU adapters.</returns>
        public static async Task<WebGPUAdapter[]> EnumerateAdaptersAsync()
        {
            var adapter = await RequestAdapterAsync();
            return adapter != null ? new[] { adapter } : Array.Empty<WebGPUAdapter>();
        }

        /// <summary>
        /// Requests a WebGPU device from this adapter.
        /// </summary>
        /// <returns>WebGPU device or null if request failed.</returns>
        public async Task<WebGPUDevice?> RequestDeviceAsync()
        {
            if (_disposed || _jsAdapter == null) return null;

#if BROWSER
            try
            {
                var devicePromise = _jsAdapter.Invoke("requestDevice");
                var device = await JSHost.ImportAsync("device_helper", "./device_helper.js")
                    .ContinueWith(m => m.Result.Invoke("awaitDevice", devicePromise));

                return device != null ? new WebGPUDevice(device, Info) : null;
            }
            catch
            {
                return null;
            }
#else
            await Task.Delay(1);
            return null;
#endif
        }

        private WebGPUAdapterInfo QueryAdapterInfo()
        {
#if BROWSER
            if (_jsAdapter != null)
            {
                // Extract adapter info from JavaScript
                var info = _jsAdapter.GetPropertyAsJSObject("info");
                return new WebGPUAdapterInfo
                {
                    Vendor = info?.GetPropertyAsString("vendor") ?? "Unknown",
                    Architecture = info?.GetPropertyAsString("architecture") ?? "Unknown",
                    Device = info?.GetPropertyAsString("device") ?? "Unknown",
                    Description = info?.GetPropertyAsString("description") ?? "WebGPU Device",
                    AdapterType = ParseAdapterType(info?.GetPropertyAsString("adapterType"))
                };
            }
#endif
            return new WebGPUAdapterInfo
            {
                Description = "WebGPU Device",
                AdapterType = WebGPUAdapterType.Unknown
            };
        }

        private static WebGPUAdapterType ParseAdapterType(string? adapterType)
        {
            return adapterType?.ToLowerInvariant() switch
            {
                "discrete-gpu" => WebGPUAdapterType.DiscreteGPU,
                "integrated-gpu" => WebGPUAdapterType.IntegratedGPU,
                "cpu" => WebGPUAdapterType.CPU,
                _ => WebGPUAdapterType.Unknown
            };
        }

        /// <summary>
        /// Disposes the WebGPU adapter.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _jsAdapter?.Dispose();
                _jsAdapter = null;
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// WebGPU device wrapper.
    /// </summary>
    public sealed class WebGPUDevice : IDisposable
    {
        private JSObject? _jsDevice;
        private bool _disposed;

        internal WebGPUDevice(JSObject jsDevice, WebGPUAdapterInfo adapterInfo)
        {
            _jsDevice = jsDevice;
            AdapterInfo = adapterInfo;
        }

        /// <summary>
        /// Gets the adapter information.
        /// </summary>
        public WebGPUAdapterInfo AdapterInfo { get; }

        /// <summary>
        /// Gets the device queue.
        /// </summary>
        public WebGPUQueue GetQueue()
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var jsQueue = _jsDevice.GetPropertyAsJSObject("queue");
                return new WebGPUQueue(jsQueue!);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Creates a command encoder.
        /// </summary>
        public WebGPUCommandEncoder CreateCommandEncoder()
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var jsEncoder = _jsDevice.Invoke("createCommandEncoder");
                return new WebGPUCommandEncoder(jsEncoder);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Creates a shader module from WGSL source.
        /// </summary>
        public WebGPUShaderModule CreateShaderModule(string wgslSource)
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var descriptor = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                descriptor.SetProperty("code", wgslSource);
                
                var jsModule = _jsDevice.Invoke("createShaderModule", descriptor);
                return new WebGPUShaderModule(jsModule);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Creates a compute pipeline.
        /// </summary>
        public WebGPUComputePipeline CreateComputePipeline(WebGPUShaderModule shaderModule, string entryPoint)
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var descriptor = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                var compute = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                compute.SetProperty("module", shaderModule.JSObject);
                compute.SetProperty("entryPoint", entryPoint);
                descriptor.SetProperty("compute", compute);
                
                var jsPipeline = _jsDevice.Invoke("createComputePipeline", descriptor);
                return new WebGPUComputePipeline(jsPipeline);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Creates a buffer.
        /// </summary>
        public WebGPUBuffer CreateBuffer(ulong size, WebGPUBufferUsage usage)
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var descriptor = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                descriptor.SetProperty("size", (double)size);
                descriptor.SetProperty("usage", (uint)usage);
                
                var jsBuffer = _jsDevice.Invoke("createBuffer", descriptor);
                return new WebGPUBuffer(jsBuffer, size, usage);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Creates a bind group.
        /// </summary>
        public WebGPUBindGroup CreateBindGroup(WebGPUBindGroupLayout layout, WebGPUBindGroupEntry[] entries)
        {
#if BROWSER
            if (_jsDevice != null)
            {
                var descriptor = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
                descriptor.SetProperty("layout", layout.JSObject);
                
                // Convert entries to JavaScript array
                var jsEntries = JSHost.GlobalThis.GetPropertyAsJSObject("Array").Invoke("from", 
                    entries.Select(e => e.ToJSObject()).ToArray());
                descriptor.SetProperty("entries", jsEntries);
                
                var jsBindGroup = _jsDevice.Invoke("createBindGroup", descriptor);
                return new WebGPUBindGroup(jsBindGroup);
            }
#endif
            throw new InvalidOperationException("Device not available");
        }

        /// <summary>
        /// Disposes the WebGPU device.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
#if BROWSER
                _jsDevice?.Invoke("destroy");
#endif
                _jsDevice?.Dispose();
                _jsDevice = null;
                _disposed = true;
            }
        }
    }

    // Placeholder WebGPU wrapper classes - full implementations would contain proper JavaScript interop
    public sealed class WebGPUQueue : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUQueue(JSObject jsQueue) => JSObject = jsQueue;

        public void Submit(WebGPUCommandBuffer commandBuffer)
        {
#if BROWSER
            JSObject?.Invoke("submit", new[] { commandBuffer.JSObject });
#endif
        }

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUCommandEncoder : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUCommandEncoder(JSObject jsEncoder) => JSObject = jsEncoder;

        public WebGPUComputePassEncoder BeginComputePass()
        {
#if BROWSER
            var jsPassEncoder = JSObject?.Invoke("beginComputePass");
            return new WebGPUComputePassEncoder(jsPassEncoder!);
#else
            throw new NotSupportedException();
#endif
        }

        public WebGPUCommandBuffer Finish()
        {
#if BROWSER
            var jsCommandBuffer = JSObject?.Invoke("finish");
            return new WebGPUCommandBuffer(jsCommandBuffer!);
#else
            throw new NotSupportedException();
#endif
        }

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUComputePassEncoder : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUComputePassEncoder(JSObject jsPassEncoder) => JSObject = jsPassEncoder;

        public void SetPipeline(WebGPUComputePipeline pipeline)
        {
#if BROWSER
            JSObject?.Invoke("setPipeline", pipeline.JSObject);
#endif
        }

        public void SetBindGroup(uint index, WebGPUBindGroup bindGroup)
        {
#if BROWSER
            JSObject?.Invoke("setBindGroup", index, bindGroup.JSObject);
#endif
        }

        public void DispatchWorkgroups(uint x, uint y = 1, uint z = 1)
        {
#if BROWSER
            JSObject?.Invoke("dispatchWorkgroups", x, y, z);
#endif
        }

        public void End()
        {
#if BROWSER
            JSObject?.Invoke("end");
#endif
        }

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUCommandBuffer : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUCommandBuffer(JSObject jsCommandBuffer) => JSObject = jsCommandBuffer;

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUShaderModule : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUShaderModule(JSObject jsShaderModule) => JSObject = jsShaderModule;

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUComputePipeline : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUComputePipeline(JSObject jsComputePipeline) => JSObject = jsComputePipeline;

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUBuffer : MemoryBuffer
    {
        internal JSObject? JSObject { get; private set; }
        private readonly ulong _size;
        private readonly WebGPUBufferUsage _usage;
        private bool _disposed;

        internal WebGPUBuffer(JSObject jsBuffer, ulong size, WebGPUBufferUsage usage)
            : base(null, (long)size, 1)
        {
            JSObject = jsBuffer;
            _size = size;
            _usage = usage;
        }

        internal WebGPUBuffer(Accelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _size = (ulong)(length * elementSize);
            _usage = WebGPUBufferUsage.Storage;
        }

        public unsafe void* GetNativePtr() => throw new NotSupportedException("WebGPU buffers don't expose raw pointers");

        public void CopyFromCPU(IntPtr source, long sourceOffset, long targetOffset, long length)
        {
            // WebGPU buffer operations are asynchronous and handled through the queue
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        public void CopyToCPU(IntPtr target, long sourceOffset, long targetOffset, long length)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        public void CopyFrom(MemoryBuffer source, long sourceOffset, long targetOffset, long length)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        public void CopyTo(MemoryBuffer target, long sourceOffset, long targetOffset, long length)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        protected internal override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        protected internal override void CopyFrom(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        protected internal override void CopyTo(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            throw new NotSupportedException("Use WebGPU-specific buffer operations");
        }

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
#if BROWSER
                JSObject?.Invoke("destroy");
#endif
                JSObject?.Dispose();
                JSObject = null;
                _disposed = true;
            }
        }
    }

    public sealed class WebGPUBindGroup : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUBindGroup(JSObject jsBindGroup) => JSObject = jsBindGroup;

        public void Dispose() => JSObject?.Dispose();
    }

    public sealed class WebGPUBindGroupLayout : IDisposable
    {
        internal JSObject? JSObject { get; private set; }

        internal WebGPUBindGroupLayout(JSObject jsBindGroupLayout) => JSObject = jsBindGroupLayout;

        public void Dispose() => JSObject?.Dispose();
    }

    public struct WebGPUBindGroupEntry
    {
        public uint Binding { get; set; }
        public WebGPUBuffer? Buffer { get; set; }
        public ulong Offset { get; set; }
        public ulong Size { get; set; }

        internal JSObject ToJSObject()
        {
#if BROWSER
            var obj = JSHost.GlobalThis.GetPropertyAsJSObject("Object").Invoke("create", JSHost.GlobalThis.GetPropertyAsJSObject("Object"));
            obj.SetProperty("binding", Binding);
            obj.SetProperty("resource", Buffer?.JSObject);
            return obj;
#else
            throw new NotSupportedException();
#endif
        }
    }

    /// <summary>
    /// Native WebGPU API bindings for non-browser environments.
    /// </summary>
    internal static class WebGPUNative
    {
        /// <summary>
        /// Checks if WebGPU is supported in the current environment.
        /// </summary>
        internal static bool IsWebGPUSupported()
        {
            // For non-browser environments, check for WebGPU native implementation
            try
            {
                // This would check for dawn, wgpu-native, or other WebGPU implementations
                return false; // Placeholder
            }
            catch
            {
                return false;
            }
        }
    }
}