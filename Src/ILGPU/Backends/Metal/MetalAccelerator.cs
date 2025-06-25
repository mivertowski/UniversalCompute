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
// Change License: Apache License, Version 2.0

#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Apple Metal accelerator for GPU compute on Apple Silicon.
    /// </summary>
    public sealed class MetalAccelerator : Accelerator
    {
        #region Instance

        /// <summary>
        /// The underlying Metal device.
        /// </summary>
        public MetalDevice Device { get; }

        /// <summary>
        /// Gets whether this device supports the Apple Neural Engine.
        /// </summary>
        public bool SupportsNeuralEngine { get; }

        /// <summary>
        /// Gets whether this device supports Apple AMX (Apple Matrix Extensions).
        /// </summary>
        public bool SupportsAMX { get; }

        /// <summary>
        /// Gets the Metal Performance Shaders interface for optimized operations.
        /// </summary>
        public MetalPerformanceShaders MPS { get; }

        /// <summary>
        /// Apple Silicon has native unified memory architecture.
        /// </summary>
        public override bool SupportsUnifiedMemory => true;

        /// <summary>
        /// Apple devices support advanced memory pooling.
        /// </summary>
        public override bool SupportsMemoryPools => true;

        internal MetalAccelerator(
            Context context,
            MetalDevice device,
            MetalPerformanceShaders mps) : base(context)
        {
            Device = device ?? throw new ArgumentNullException(nameof(device));
            MPS = mps ?? throw new ArgumentNullException(nameof(mps));
            
            // Detect Apple-specific capabilities
            SupportsNeuralEngine = MetalCapabilities.DetectNeuralEngine();
            SupportsAMX = MetalCapabilities.DetectAMX();
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates memory using Apple's unified memory architecture.
        /// </summary>
        public override IMemoryBuffer<T> Allocate<T>(long length)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length));

            return new MetalUnifiedBuffer<T>(this, length);
        }

        /// <summary>
        /// Allocates 2D memory with optimal layout for Metal compute.
        /// </summary>
        public override IMemoryBuffer2D<T> Allocate2D<T>(long width, long height)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException();

            return new MetalUnifiedBuffer2D<T>(this, width, height);
        }

        /// <summary>
        /// Allocates 3D memory with optimal layout for Metal compute.
        /// </summary>
        public override IMemoryBuffer3D<T> Allocate3D<T>(long width, long height, long depth)
        {
            if (width <= 0 || height <= 0 || depth <= 0)
                throw new ArgumentOutOfRangeException();

            return new MetalUnifiedBuffer3D<T>(this, width, height, depth);
        }

        #endregion

        #region Kernel Compilation

        /// <summary>
        /// Compiles a kernel to Metal Shading Language.
        /// </summary>
        public override async Task<IKernel> CompileKernelAsync(
            KernelSource source,
            CompilationOptions options,
            CancellationToken cancellationToken = default)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var compiler = new MetalKernelCompiler(Device);
            return await compiler.CompileAsync(source, options, cancellationToken);
        }

        /// <summary>
        /// Loads a precompiled Metal kernel from a metallib binary.
        /// </summary>
        public async Task<IKernel> LoadCompiledKernelAsync(
            byte[] metallibData,
            string functionName,
            CancellationToken cancellationToken = default)
        {
            if (metallibData == null || metallibData.Length == 0)
                throw new ArgumentException("Metallib data cannot be null or empty", nameof(metallibData));

            if (string.IsNullOrEmpty(functionName))
                throw new ArgumentException("Function name cannot be null or empty", nameof(functionName));

            var library = await Device.CreateLibraryAsync(metallibData, cancellationToken);
            var function = library.CreateFunction(functionName);
            
            return new MetalKernel(this, function);
        }

        #endregion

        #region Stream Management

        /// <summary>
        /// Creates a new Metal command queue for asynchronous execution.
        /// </summary>
        public override AcceleratorStream CreateStream()
        {
            return new MetalStream(this, Device.CreateCommandQueue());
        }

        #endregion

        #region Device Information

        public override string Name => Device.Name;
        public override AcceleratorType AcceleratorType => AcceleratorType.Metal;
        public override MemoryInfo MemoryInfo => Device.MemoryInfo;

        #endregion

        #region Apple-Specific Operations

        /// <summary>
        /// Executes an operation on the Apple Neural Engine if available.
        /// </summary>
        public async Task<ITensor<T>> ExecuteNeuralEngineAsync<T>(
            NeuralOperation operation,
            ITensor<T> input,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            if (!SupportsNeuralEngine)
                throw new NotSupportedException("Neural Engine not available on this device");

            if (operation == null)
                throw new ArgumentNullException(nameof(operation));

            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Route to Neural Engine through CoreML or BNNS
            var neuralEngine = new AppleNeuralEngine(Device);
            return await neuralEngine.ExecuteAsync(operation, input, cancellationToken);
        }

        /// <summary>
        /// Executes matrix operations using Apple AMX if available.
        /// </summary>
        public async Task<ITensor<T>> ExecuteAMXAsync<T>(
            MatrixOperation operation,
            ITensor<T> a,
            ITensor<T> b,
            CancellationToken cancellationToken = default)
            where T : unmanaged
        {
            if (!SupportsAMX)
                throw new NotSupportedException("Apple AMX not available on this device");

            if (operation == null)
                throw new ArgumentNullException(nameof(operation));

            // Route to Apple AMX through Accelerate framework
            var amx = new AppleAMX();
            return await amx.ExecuteAsync(operation, a, b, cancellationToken);
        }

        #endregion

        #region Disposal

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                Device?.Dispose();
                MPS?.Dispose();
            }
        }

        #endregion
    }
}
#endif