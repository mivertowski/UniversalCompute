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
// Change License: Apache License, Version 2.0#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Metal kernel implementation.
    /// </summary>
    public sealed class MetalKernel : IKernel
    {
        private readonly MetalAccelerator _accelerator;
        private readonly MetalFunction _function;
        private readonly IntPtr _pipelineState;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalKernel class.
        /// </summary>
        /// <param name="accelerator">The Metal accelerator.</param>
        /// <param name="function">The Metal function.</param>
        public MetalKernel(MetalAccelerator accelerator, MetalFunction function)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _function = function ?? throw new ArgumentNullException(nameof(function));
            
            // Create compute pipeline state
            _pipelineState = MetalNative.MTLDeviceNewComputePipelineStateWithFunction(
                accelerator.Device.NativeDevice, 
                function.NativeFunction, 
                out var error);

            if (_pipelineState == IntPtr.Zero || error != IntPtr.Zero)
            {
                if (error != IntPtr.Zero)
                    MetalNative.CFRelease(error);
                throw new InvalidOperationException("Failed to create Metal compute pipeline state");
            }

            Name = "MetalKernel";
            Info = new KernelInfo(0, 0, 1024); // Default values, would be queried from Metal
        }

        /// <summary>
        /// Gets the kernel name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the accelerator this kernel was compiled for.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Gets information about the kernel's resource requirements.
        /// </summary>
        public KernelInfo Info { get; }

        /// <summary>
        /// Gets the native Metal compute pipeline state.
        /// </summary>
        public IntPtr PipelineState => _pipelineState;

        /// <summary>
        /// Executes the kernel with the specified parameters.
        /// </summary>
        /// <param name="groupSize">The group size for kernel execution.</param>
        /// <param name="gridSize">The grid size for kernel execution.</param>
        /// <param name="parameters">The kernel parameters.</param>
        public void Execute(int groupSize, int gridSize, params object[] parameters)
        {
            ThrowIfDisposed();
            
            using var commandQueue = _accelerator.Device.CreateCommandQueue();
            using var commandBuffer = commandQueue.CreateCommandBuffer();
            
            ExecuteInternal(commandBuffer, groupSize, gridSize, parameters);
            
            commandBuffer.Commit();
            commandBuffer.WaitUntilCompleted();
        }

        /// <summary>
        /// Executes the kernel asynchronously with the specified parameters.
        /// </summary>
        /// <param name="groupSize">The group size for kernel execution.</param>
        /// <param name="gridSize">The grid size for kernel execution.</param>
        /// <param name="parameters">The kernel parameters.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task representing the asynchronous execution.</returns>
        public async Task ExecuteAsync(int groupSize, int gridSize, object[] parameters, 
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            
            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                Execute(groupSize, gridSize, parameters);
            }, cancellationToken);
        }

        /// <summary>
        /// Executes the kernel using the specified command buffer.
        /// </summary>
        /// <param name="commandBuffer">The Metal command buffer.</param>
        /// <param name="groupSize">The threadgroup size.</param>
        /// <param name="gridSize">The grid size.</param>
        /// <param name="parameters">The kernel parameters.</param>
        public void ExecuteInternal(MetalCommandBuffer commandBuffer, int groupSize, int gridSize, object[] parameters)
        {
            // Create compute command encoder
            var encoder = MetalNative.MTLCommandBufferComputeCommandEncoder(commandBuffer.NativeBuffer);
            if (encoder == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Metal compute command encoder");

            try
            {
                // Set compute pipeline state
                MetalNative.MTLComputeCommandEncoderSetComputePipelineState(encoder, _pipelineState);

                // Set kernel parameters (buffers, textures, etc.)
                SetKernelParameters(encoder, parameters);

                // Calculate threadgroup configuration
                var (threadsPerThreadgroup, threadgroupsPerGrid) = CalculateDispatchConfiguration(groupSize, gridSize);

                // Dispatch compute threads
                MetalNative.MTLComputeCommandEncoderDispatchThreadgroups(
                    encoder, 
                    threadgroupsPerGrid, 
                    threadsPerThreadgroup);

                // End encoding
                MetalNative.MTLComputeCommandEncoderEndEncoding(encoder);
            }
            finally
            {
                if (encoder != IntPtr.Zero)
                    MetalNative.CFRelease(encoder);
            }
        }

        private void SetKernelParameters(IntPtr encoder, object[] parameters)
        {
            if (parameters == null) return;

            for (int i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                
                if (parameter is MetalUnifiedBuffer<float> floatBuffer)
                {
                    MetalNative.MTLComputeCommandEncoderSetBuffer(
                        encoder, 
                        floatBuffer.NativeBuffer, 
                        0, // offset
                        (nuint)i);
                }
                else if (parameter is MetalUnifiedBuffer<int> intBuffer)
                {
                    MetalNative.MTLComputeCommandEncoderSetBuffer(
                        encoder, 
                        intBuffer.NativeBuffer, 
                        0, // offset
                        (nuint)i);
                }
                // Add support for other buffer types as needed
            }
        }

        private (MTLSize threadsPerThreadgroup, MTLSize threadgroupsPerGrid) CalculateDispatchConfiguration(int groupSize, int gridSize)
        {
            // Apple Silicon GPUs prefer certain threadgroup sizes
            var capabilities = _accelerator.Device.Capabilities;
            var optimalSize = capabilities.GetOptimalThreadgroupSize(groupSize);

            var threadsPerThreadgroup = new MTLSize(
                (nuint)optimalSize.width,
                (nuint)optimalSize.height,
                (nuint)optimalSize.depth);

            // Calculate number of threadgroups needed
            var totalThreads = gridSize * groupSize;
            var threadsPerGroup = optimalSize.width * optimalSize.height * optimalSize.depth;
            var numThreadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;

            var threadgroupsPerGrid = new MTLSize(
                (nuint)numThreadgroups,
                1,
                1);

            return (threadsPerThreadgroup, threadgroupsPerGrid);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalKernel));
        }

        /// <summary>
        /// Disposes the Metal kernel.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_pipelineState != IntPtr.Zero)
                {
                    MetalNative.CFRelease(_pipelineState);
                }
                _function?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Metal kernel compiler.
    /// </summary>
    public sealed class MetalKernelCompiler
    {
        private readonly MetalDevice _device;

        /// <summary>
        /// Initializes a new instance of the MetalKernelCompiler class.
        /// </summary>
        /// <param name="device">The Metal device.</param>
        public MetalKernelCompiler(MetalDevice device)
        {
            _device = device ?? throw new ArgumentNullException(nameof(device));
        }

        /// <summary>
        /// Compiles a kernel from the specified source.
        /// </summary>
        /// <param name="source">The kernel source.</param>
        /// <param name="options">The compilation options.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>The compiled kernel.</returns>
        public async Task<IKernel> CompileAsync(KernelSource source, CompilationOptions options, 
            CancellationToken cancellationToken = default)
        {
            if (source.Language != KernelLanguage.Metal && source.Language != KernelLanguage.ILGPU)
                throw new NotSupportedException($"Kernel language {source.Language} not supported by Metal compiler");

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Convert ILGPU to Metal Shading Language if needed
                var metalSource = source.Language == KernelLanguage.ILGPU
                    ? TranslateILGPUToMetal(source.Source)
                    : source.Source;

                // Compile Metal source
                var compileOptions = CreateMetalCompileOptions(options);
                var library = _device.CompileLibraryAsync(metalSource, compileOptions, cancellationToken).Result;
                var function = library.CreateFunction(source.EntryPoint);

                // Create Metal accelerator (assuming we have access to it)
                var accelerator = GetMetalAccelerator();
                return new MetalKernel(accelerator, function);
            }, cancellationToken);
        }

        private string TranslateILGPUToMetal(string ilgpuSource)
        {
            // This would contain the ILGPU IR to Metal Shading Language translation
            // For now, assume the source is already in Metal Shading Language
            return ilgpuSource;
        }

        private MetalCompileOptions CreateMetalCompileOptions(CompilationOptions options)
        {
            return new MetalCompileOptions
            {
                OptimizationLevel = options.OptimizationLevel,
                DebugMode = options.DebugMode,
                FastMath = options.FastMath
            };
        }

        private MetalAccelerator GetMetalAccelerator()
        {
            // This would typically be provided through the compilation context
            // For now, create a basic Metal accelerator
            throw new NotImplementedException("MetalAccelerator creation from compiler context not implemented");
        }
    }

    /// <summary>
    /// Metal compilation options.
    /// </summary>
    public sealed class MetalCompileOptions
    {
        /// <summary>
        /// Gets or sets the optimization level.
        /// </summary>
        public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.O2;

        /// <summary>
        /// Gets or sets whether debug mode is enabled.
        /// </summary>
        public bool DebugMode { get; set; }

        /// <summary>
        /// Gets or sets whether fast math is enabled.
        /// </summary>
        public bool FastMath { get; set; }

        /// <summary>
        /// Converts to native Metal compile options.
        /// </summary>
        /// <returns>Native Metal compile options handle.</returns>
        public IntPtr ToNativeOptions()
        {
            // This would create a native MTLCompileOptions object
            // For now, return zero (null) to use default options
            return IntPtr.Zero;
        }
    }

    /// <summary>
    /// Metal stream implementation.
    /// </summary>
    public sealed class MetalStream : AcceleratorStream
    {
        private readonly MetalAccelerator _accelerator;
        private readonly MetalCommandQueue _commandQueue;

        /// <summary>
        /// Initializes a new instance of the MetalStream class.
        /// </summary>
        /// <param name="accelerator">The Metal accelerator.</param>
        /// <param name="commandQueue">The Metal command queue.</param>
        public MetalStream(MetalAccelerator accelerator, MetalCommandQueue commandQueue)
            : base(accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _commandQueue = commandQueue ?? throw new ArgumentNullException(nameof(commandQueue));
        }

        /// <summary>
        /// Gets the Metal command queue.
        /// </summary>
        public MetalCommandQueue CommandQueue => _commandQueue;

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        public override void Synchronize()
        {
            // Metal command buffers are automatically synchronized when they complete
            // No explicit synchronization needed for unified memory architecture
        }

        /// <summary>
        /// Synchronizes the stream asynchronously.
        /// </summary>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task representing the asynchronous synchronization.</returns>
        public override async Task SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            await Task.Run(Synchronize, cancellationToken);
        }

        /// <summary>
        /// Disposes the Metal stream.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                _commandQueue?.Dispose();
            }
        }
    }
}
#endif