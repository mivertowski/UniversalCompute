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

#if INTEL_OPENVINO_AVAILABLE
using OpenVinoSharp;
#endif
using System.Diagnostics.CodeAnalysis;

namespace ILGPU.Benchmarks.Infrastructure;

#if INTEL_OPENVINO_AVAILABLE

/// <summary>
/// Real Intel NPU accelerator using OpenVINO Runtime.
/// </summary>
[SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "Specialized accelerator must handle all exceptions gracefully for fallback behavior")]
public sealed class OpenVINONPUAccelerator : ISpecializedAccelerator
{
    private readonly Core _core;
    private readonly string _npuDevice;
    private CompiledModel? _matrixMultiplyModel;
    private CompiledModel? _convolutionModel;
    private CompiledModel? _inferenceModel;
    private bool _disposed;

    public string Name => $"Intel NPU ({_npuDevice})";
    public HardwareCapabilities SupportedOperations => 
        HardwareCapabilities.IntelNPU;

    public bool IsAvailable { get; private set; }

    public OpenVINONPUAccelerator()
    {
        try
        {
            _core = new Core();
            var devices = _core.get_available_devices();
            _npuDevice = devices.FirstOrDefault(d => d.Contains("NPU") || d.Contains("VPU")) 
                        ?? throw new InvalidOperationException("No NPU device found");
            
            IsAvailable = true;
            Console.WriteLine($"✅ Intel NPU initialized: {_npuDevice}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Intel NPU initialization failed: {ex.Message}");
            IsAvailable = false;
            throw;
        }
    }

    public async Task<float[]> ExecuteMatrixMultiplyAsync(float[] a, float[] b, int size)
    {
        ThrowIfDisposed();
        
        try
        {
            // Create or get cached matrix multiply model
            if (_matrixMultiplyModel == null)
            {
                _matrixMultiplyModel = await CreateMatrixMultiplyModelAsync(size);
            }

            // Create inference request
            using var inferRequest = _matrixMultiplyModel.create_infer_request();
            
            // Set input tensors
            var inputTensorA = new Tensor(new Shape(new ulong[] { (ulong)size, (ulong)size }), ElementType.F32, a);
            var inputTensorB = new Tensor(new Shape(new ulong[] { (ulong)size, (ulong)size }), ElementType.F32, b);
            
            inferRequest.set_input_tensor(0, inputTensorA);
            inferRequest.set_input_tensor(1, inputTensorB);

            // Execute inference on NPU
            inferRequest.infer();

            // Get output tensor
            var outputTensor = inferRequest.get_output_tensor();
            var outputData = outputTensor.get_data<float>();

            Console.WriteLine($"✅ Intel NPU matrix multiply completed: {size}x{size}");
            return outputData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Intel NPU matrix multiply failed: {ex.Message}");
            throw;
        }
    }

    public async Task<float[]> ExecuteConvolutionAsync(float[] input, float[] kernel, int[] dimensions)
    {
        ThrowIfDisposed();
        
        try
        {
            var batchSize = dimensions[0];
            var channels = dimensions[1];
            var height = dimensions[2];
            var width = dimensions[3];
            var kernelSize = (int)Math.Sqrt(kernel.Length / channels);

            // Create or get cached convolution model
            if (_convolutionModel == null)
            {
                _convolutionModel = await CreateConvolutionModelAsync(channels, height, width, kernelSize);
            }

            using var inferRequest = _convolutionModel.create_infer_request();
            
            // Set input data
            var inputTensor = new Tensor(new Shape(new ulong[] { (ulong)batchSize, (ulong)channels, (ulong)height, (ulong)width }), ElementType.F32, input);
            var kernelTensor = new Tensor(new Shape(new ulong[] { (ulong)channels, (ulong)channels, (ulong)kernelSize, (ulong)kernelSize }), ElementType.F32, kernel);
            
            inferRequest.set_input_tensor(0, inputTensor);
            inferRequest.set_input_tensor(1, kernelTensor);

            // Execute inference
            inferRequest.infer();

            var outputTensor = inferRequest.get_output_tensor();
            var outputData = outputTensor.get_data<float>();

            Console.WriteLine($"✅ Intel NPU convolution completed: {channels}x{height}x{width}");
            return outputData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Intel NPU convolution failed: {ex.Message}");
            throw;
        }
    }

    public async Task<float[]> ExecuteInferenceAsync(float[] input, string modelPath)
    {
        ThrowIfDisposed();
        
        try
        {
            // Load and compile model for NPU
            var model = _core.read_model(modelPath);
            var compiledModel = _core.compile_model(model, _npuDevice);
            
            using var inferRequest = compiledModel.create_infer_request();
            
            // Set input tensor (assuming single input)
            var inputShape = model.get_parameters()[0].get_shape();
            var inputTensor = new Tensor(inputShape, ElementType.F32, input);
            inferRequest.set_input_tensor(inputTensor);

            // Execute inference
            inferRequest.infer();

            // Get output
            var outputTensor = inferRequest.get_output_tensor();
            var outputData = outputTensor.get_data<float>();

            Console.WriteLine($"✅ Intel NPU inference completed: {modelPath}");
            return outputData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Intel NPU inference failed: {ex.Message}");
            throw;
        }
    }

    private async Task<CompiledModel> CreateMatrixMultiplyModelAsync(int size)
    {
        // Create a simple matrix multiply model in OpenVINO IR format
        // For demo purposes, we'll create a basic linear layer
        try
        {
            // Create model using OpenVINO model builder
            var builder = new ModelBuilder();
            
            // Define inputs
            var inputA = builder.parameter(new Shape(new ulong[] { (ulong)size, (ulong)size }), ElementType.F32, "input_a");
            var inputB = builder.parameter(new Shape(new ulong[] { (ulong)size, (ulong)size }), ElementType.F32, "input_b");
            
            // Matrix multiplication operation
            var matmul = builder.matmul(inputA, inputB, false, false);
            
            // Create model
            var model = new Model(new[] { matmul }, new[] { inputA, inputB }, "matrix_multiply");
            
            // Compile for NPU with optimization
            var config = new Dictionary<string, object>
            {
                ["PERFORMANCE_HINT"] = "THROUGHPUT",
                ["INFERENCE_PRECISION_HINT"] = "f16" // Use FP16 for NPU optimization
            };
            
            return _core.compile_model(model, _npuDevice, config);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to create matrix multiply model: {ex.Message}");
            throw;
        }
    }

    private async Task<CompiledModel> CreateConvolutionModelAsync(int channels, int height, int width, int kernelSize)
    {
        try
        {
            var builder = new ModelBuilder();
            
            // Define inputs
            var input = builder.parameter(new Shape(new ulong[] { 1, (ulong)channels, (ulong)height, (ulong)width }), ElementType.F32, "input");
            var kernel = builder.parameter(new Shape(new ulong[] { (ulong)channels, (ulong)channels, (ulong)kernelSize, (ulong)kernelSize }), ElementType.F32, "kernel");
            
            // Convolution operation
            var conv = builder.convolution(input, kernel, new ulong[] { 1, 1 }, new ulong[] { 0, 0, 0, 0 }, new ulong[] { 1, 1 });
            
            var model = new Model(new[] { conv }, new[] { input, kernel }, "convolution");
            
            var config = new Dictionary<string, object>
            {
                ["PERFORMANCE_HINT"] = "THROUGHPUT",
                ["INFERENCE_PRECISION_HINT"] = "f16"
            };
            
            return _core.compile_model(model, _npuDevice, config);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to create convolution model: {ex.Message}");
            throw;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(OpenVINONPUAccelerator));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _matrixMultiplyModel?.Dispose();
            _convolutionModel?.Dispose();
            _inferenceModel?.Dispose();
            _core?.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Error disposing OpenVINO NPU accelerator: {ex.Message}");
        }
        finally
        {
            _disposed = true;
        }
    }
}

/// <summary>
/// Model builder helper for creating OpenVINO models programmatically.
/// </summary>
internal class ModelBuilder
{
    public Parameter parameter(Shape shape, ElementType elementType, string name)
    {
        return new Parameter(elementType, shape) { Name = name };
    }

    public Operation matmul(Parameter a, Parameter b, bool transposeA, bool transposeB)
    {
        // Create MatMul operation
        return new MatMul(a, b, transposeA, transposeB);
    }

    public Operation convolution(Parameter input, Parameter kernel, ulong[] strides, ulong[] padding, ulong[] dilations)
    {
        // Create Convolution operation
        return new Convolution(input, kernel, strides, padding, padding, dilations);
    }
}

#else

/// <summary>
/// Stub implementation when OpenVINO is not available.
/// </summary>
[SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "Specialized accelerator must handle all exceptions gracefully for fallback behavior")]
public sealed class OpenVINONPUAccelerator : ISpecializedAccelerator
{
    public string Name => "Intel NPU (OpenVINO not available)";
    public HardwareCapabilities SupportedOperations => HardwareCapabilities.None;
    public bool IsAvailable => false;

    public OpenVINONPUAccelerator()
    {
        Console.WriteLine("❌ Intel NPU not available: OpenVINO runtime not compiled");
    }

    public Task<float[]> ExecuteMatrixMultiplyAsync(float[] a, float[] b, int size)
    {
        throw new NotSupportedException("Intel NPU not available on this platform");
    }

    public Task<float[]> ExecuteConvolutionAsync(float[] input, float[] kernel, int[] dimensions)
    {
        throw new NotSupportedException("Intel NPU not available on this platform");
    }

    public Task<float[]> ExecuteInferenceAsync(float[] input, string modelPath)
    {
        throw new NotSupportedException("Intel NPU not available on this platform");
    }

    public void Dispose()
    {
        // Nothing to dispose
    }
}

#endif