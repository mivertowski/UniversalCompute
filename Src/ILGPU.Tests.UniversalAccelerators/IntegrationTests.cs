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

using ILGPU.Algorithms.Cryptography;
using ILGPU.Algorithms.ComputerVision;
using ILGPU.Algorithms.FFT;
using ILGPU.Algorithms.SparseMatrix;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using System.Numerics;
using System.Text;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators
{
    /// <summary>
    /// Integration tests for the universal accelerator framework.
    /// Tests cross-component functionality and end-to-end scenarios.
    /// </summary>
    public class IntegrationTests : TestBase
    {
        #region Multi-Component Integration Tests

        [Fact]
        public void TestCryptographicImageProcessingPipeline()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 64;
            const int height = 64;
            const int channels = 3;
            
            // Create test image
            var imageData = new byte[width * height * channels];
            var random = new Random(42);
            random.NextBytes(imageData);
            
            using var image = Image<byte>.Create(accelerator!, width, height, channels, imageData);
            using var processedImage = Image<byte>.Create(accelerator!, width, height, channels);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            
            // Process image with Gaussian blur
            ImageProcessing.GaussianBlur(image, processedImage, kernelSize: 3, sigma: 1.0f, accelerator!.DefaultStream);
            
            // Compute cryptographic hash of processed image
            HashFunctions.SHA256(processedImage.Data.AsArrayView(), hashBuffer.View, accelerator!.DefaultStream);
            
            var hash = hashBuffer.GetAsArray1D();
            
            // Verify pipeline completion
            Assert.True(hash.Any(b => b != 0), "Hash should not be all zeros");
            
            // Verify image processing altered the data
            var originalData = image.Data.GetAsArray1D();
            var processedData = processedImage.Data.GetAsArray1D();
            Assert.False(originalData.SequenceEqual(processedData), "Image processing should alter the data");
        }

        [Fact]
        public void TestFFTBasedSignalEncryption()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int signalSize = 1024;
            
            // Generate test signal
            var signal = new Complex[signalSize];
            for (int i = 0; i < signalSize; i++)
            {
                signal[i] = new Complex(Math.Sin(2 * Math.PI * i / 64.0), 0); // 16 cycles
            }
            
            using var signalBuffer = accelerator!.Allocate1D(signal);
            using var fftBuffer = accelerator!.Allocate1D<Complex>(signalSize);
            using var encryptedBuffer = accelerator!.Allocate1D<byte>(signalSize * 8); // Complex as bytes
            using var keyBuffer = accelerator!.Allocate1D<byte>(32);
            using var ivBuffer = accelerator!.Allocate1D<byte>(16);
            
            var fft = new FFT<float>(accelerator!);
            
            // Step 1: Transform signal to frequency domain
            fft.Forward1D(signalBuffer.View, fftBuffer.View);
            
            // Step 2: Convert complex FFT result to bytes for encryption
            var fftData = fftBuffer.GetAsArray1D();
            var fftBytes = new byte[signalSize * 8];
            Buffer.BlockCopy(fftData.SelectMany(c => BitConverter.GetBytes(c.Real).Concat(BitConverter.GetBytes(c.Imaginary))).ToArray(), 
                0, fftBytes, 0, fftBytes.Length);
            
            using var fftBytesBuffer = accelerator!.Allocate1D(fftBytes);
            
            // Step 3: Encrypt frequency domain data
            var random = new Random(42);
            var key = new byte[32];
            var iv = new byte[16];
            random.NextBytes(key);
            random.NextBytes(iv);
            
            keyBuffer.CopyFromCPU(key);
            ivBuffer.CopyFromCPU(iv);
            
            SymmetricCryptography.AESEncrypt(
                fftBytesBuffer.View, encryptedBuffer.View,
                keyBuffer.View, ivBuffer.View,
                AESMode.CBC, accelerator!.DefaultStream);
            
            // Verify the pipeline worked
            var encryptedData = encryptedBuffer.GetAsArray1D();
            Assert.False(fftBytes.SequenceEqual(encryptedData), "Encryption should change the data");
            Assert.True(encryptedData.Any(b => b != 0), "Encrypted data should not be all zeros");
        }

        [Fact]
        public void TestSparseMatrixGraphAnalyticsPipeline()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create adjacency matrix for a small graph
            const int numVertices = 6;
            var rowPtr = new int[] { 0, 2, 4, 6, 8, 10, 12 };
            var colIndices = new int[] { 1, 2, 0, 3, 0, 4, 1, 5, 2, 5, 3, 4 };
            var values = new float[12];
            for (int i = 0; i < values.Length; i++)
                values[i] = 1.0f; // Unweighted graph
            
            using var adjacencyMatrix = CSRMatrix<float>.Create(
                accelerator!, numVertices, numVertices, values.Length,
                rowPtr, colIndices, values);
            
            // Test 1: Sparse matrix-vector multiplication (simulating PageRank iteration)
            var pageRank = new float[numVertices];
            for (int i = 0; i < numVertices; i++)
                pageRank[i] = 1.0f / numVertices; // Initial uniform distribution
            
            using var pageRankBuffer = accelerator!.Allocate1D(pageRank);
            using var newPageRankBuffer = accelerator!.Allocate1D<float>(numVertices);
            
            // One PageRank iteration: new_pr = 0.85 * A^T * pr + 0.15 / n
            SparseMatrixOperations.SpMV(adjacencyMatrix, pageRankBuffer.View, newPageRankBuffer.View, 0.85f, 0.0f);
            
            var newPageRank = newPageRankBuffer.GetAsArray1D();
            
            // Add teleportation probability
            for (int i = 0; i < numVertices; i++)
                newPageRank[i] += 0.15f / numVertices;
            
            // Verify PageRank properties
            var sum = newPageRank.Sum();
            Assert.True(Math.Abs(sum - 1.0f) < 1e-5f, $"PageRank should sum to 1, got {sum}");
            Assert.True(newPageRank.All(pr => pr > 0), "All PageRank values should be positive");
            
            // Test 2: Convert to different sparse matrix format
            using var cscMatrix = SparseMatrixOperations.ConvertCSRToCSC(adjacencyMatrix, accelerator!.DefaultStream);
            
            // Verify conversion preserves matrix properties
            Assert.Equal(adjacencyMatrix.NumRows, cscMatrix.NumRows);
            Assert.Equal(adjacencyMatrix.NumCols, cscMatrix.NumCols);
            Assert.Equal(adjacencyMatrix.NumNonZeros, cscMatrix.NumNonZeros);
        }

        [Fact]
        public void TestComputerVisionFeatureHashingPipeline()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int imageSize = 32;
            
            // Create test image with simple pattern
            var imageData = new float[imageSize * imageSize];
            for (int y = 0; y < imageSize; y++)
            {
                for (int x = 0; x < imageSize; x++)
                {
                    // Create checkerboard pattern
                    imageData[y * imageSize + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
                }
            }
            
            using var image = Image<float>.Create(accelerator!, imageSize, imageSize, 1, imageData);
            using var edges = Image<float>.Create(accelerator!, imageSize, imageSize, 1);
            using var corners = Image<float>.Create(accelerator!, imageSize, imageSize, 1);
            
            // Step 1: Edge detection
            FeatureDetection.CannyEdgeDetection(image, edges,
                lowThreshold: 0.1f, highThreshold: 0.3f, sigma: 1.0f, accelerator!.DefaultStream);
            
            // Step 2: Corner detection
            FeatureDetection.HarrisCornerDetection(image, corners,
                threshold: 0.1f, k: 0.04f, accelerator!.DefaultStream);
            
            // Step 3: Extract feature descriptors (simplified)
            var edgeData = edges.Data.GetAsArray1D();
            var cornerData = corners.Data.GetAsArray1D();
            
            // Combine edge and corner features into descriptor
            var featureDescriptor = new byte[edgeData.Length + cornerData.Length];
            for (int i = 0; i < edgeData.Length; i++)
                featureDescriptor[i] = (byte)(edgeData[i] * 255);
            for (int i = 0; i < cornerData.Length; i++)
                featureDescriptor[edgeData.Length + i] = (byte)(cornerData[i] * 255);
            
            // Step 4: Hash the feature descriptor for fast matching
            using var descriptorBuffer = accelerator!.Allocate1D(featureDescriptor);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            
            HashFunctions.SHA256(descriptorBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var featureHash = hashBuffer.GetAsArray1D();
            
            // Verify pipeline completion
            Assert.True(featureHash.Any(b => b != 0), "Feature hash should not be all zeros");
            Assert.True(edgeData.Any(f => f > 0.1f), "Edge detection should find some edges");
            Assert.True(cornerData.Any(f => f > 0.1f), "Corner detection should find some corners");
        }

        #endregion

        #region Performance Integration Tests

        [Fact]
        public void TestMultiAlgorithmPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int dataSize = 1024;
            var testData = CreateTestData(dataSize);
            
            // Test sequential execution of multiple algorithms
            var totalTime = MeasureTime(() =>
            {
                // FFT
                var complexData = testData.Select(f => new Complex(f, 0)).ToArray();
                using var fftBuffer = accelerator!.Allocate1D(complexData);
                using var fftResult = accelerator!.Allocate1D<Complex>(dataSize);
                var fft = new FFT<float>(accelerator!);
                fft.Forward1D(fftBuffer.View, fftResult.View);
                
                // Hash computation
                var byteData = testData.SelectMany(f => BitConverter.GetBytes(f)).ToArray();
                using var hashInput = accelerator!.Allocate1D(byteData);
                using var hashOutput = accelerator!.Allocate1D<byte>(32);
                HashFunctions.SHA256(hashInput.View, hashOutput.View, accelerator!.DefaultStream);
                
                // Sparse matrix operation
                var rowPtr = new int[] { 0, 2, 4 };
                var colIndices = new int[] { 0, 1, 0, 1 };
                var values = new float[] { 1.0f, 0.5f, 0.5f, 1.0f };
                using var matrix = CSRMatrix<float>.Create(accelerator!, 2, 2, 4, rowPtr, colIndices, values);
                using var vector = accelerator!.Allocate1D(new float[] { 1.0f, 2.0f });
                using var result = accelerator!.Allocate1D<float>(2);
                SparseMatrixOperations.SpMV(matrix, vector.View, result.View, 1.0f, 0.0f);
                
                accelerator!.Synchronize();
            });
            
            Assert.True(totalTime < 5000, $"Multi-algorithm execution took {totalTime}ms, expected < 5000ms");
        }

        [Fact]
        public void TestMemoryIntensiveOperations()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int largeSize = 10000;
            
            // Test memory allocation and operations with large datasets
            using var largeBuffer1 = accelerator!.Allocate1D<float>(largeSize);
            using var largeBuffer2 = accelerator!.Allocate1D<float>(largeSize);
            using var largeBuffer3 = accelerator!.Allocate1D<float>(largeSize);
            
            var testData = CreateTestData(largeSize);
            largeBuffer1.CopyFromCPU(testData);
            largeBuffer2.CopyFromCPU(testData);
            
            var operationTime = MeasureTime(() =>
            {
                // Simulate memory-intensive operation (element-wise operations)
                var kernel = accelerator!.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, c) => c[index] = a[index] + b[index] * 2.0f);
                
                kernel(largeSize, largeBuffer1.View, largeBuffer2.View, largeBuffer3.View);
                accelerator!.Synchronize();
            });
            
            Assert.True(operationTime < 2000, $"Large memory operation took {operationTime}ms, expected < 2000ms");
            
            // Verify operation correctness
            var result = largeBuffer3.GetAsArray1D();
            for (int i = 0; i < Math.Min(100, largeSize); i++) // Check first 100 elements
            {
                var expected = testData[i] + testData[i] * 2.0f;
                Assert.True(Math.Abs(result[i] - expected) < 1e-5f, 
                    $"Large operation incorrect at index {i}: expected {expected}, got {result[i]}");
            }
        }

        #endregion

        #region Cross-Platform Compatibility Tests

        [Fact]
        public void TestAcceleratorCompatibility()
        {
            using var context = Context.CreateDefault();
            
            // Test that at least one accelerator is available
            var cpuAccelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            Assert.NotNull(cpuAccelerator);
            
            using (cpuAccelerator)
            {
                // Test basic operations work on available accelerator
                const int size = 100;
                var testData = CreateTestData(size);
                
                using var buffer = cpuAccelerator!.Allocate1D(testData);
                var result = buffer.GetAsArray1D();
                
                AssertEqual(testData, result);
            }
        }

        [Fact]
        public void TestUniversalAlgorithmAccess()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Verify that all major algorithm categories are accessible
            
            // FFT
            Assert.NotNull(new FFT<float>(accelerator!));
            
            // Cryptography
            using var testBuffer = accelerator!.Allocate1D<byte>(32);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            HashFunctions.SHA256(testBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            // Computer Vision
            using var image = Image<float>.Create(accelerator!, 10, 10, 1);
            Assert.NotNull(image);
            
            // Sparse Matrix
            var rowPtr = new int[] { 0, 1 };
            var colIndices = new int[] { 0 };
            var values = new float[] { 1.0f };
            using var matrix = CSRMatrix<float>.Create(accelerator!, 1, 1, 1, rowPtr, colIndices, values);
            Assert.NotNull(matrix);
        }

        #endregion

        #region Error Recovery Tests

        [Fact]
        public void TestGracefulErrorHandling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test that errors in one operation don't affect subsequent operations
            
            // First, cause an error (invalid dimensions)
            try
            {
                using var invalidImage = Image<float>.Create(accelerator!, 0, 10, 1);
                Assert.True(false, "Should have thrown an exception");
            }
            catch (ArgumentException)
            {
                // Expected
            }
            
            // Then verify that accelerator still works normally
            using var validImage = Image<float>.Create(accelerator!, 10, 10, 1);
            Assert.NotNull(validImage);
            
            var testData = CreateTestData(100);
            using var buffer = accelerator!.Allocate1D(testData);
            var result = buffer.GetAsArray1D();
            
            AssertEqual(testData, result);
        }

        [Fact]
        public void TestResourceCleanup()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test that resources are properly cleaned up
            const int iterations = 10;
            
            for (int i = 0; i < iterations; i++)
            {
                using var image = Image<float>.Create(accelerator!, 100, 100, 3);
                using var matrix = CSRMatrix<float>.Create(
                    accelerator!, 10, 10, 10,
                    new int[11], new int[10], new float[10]);
                using var buffer = accelerator!.Allocate1D<float>(1000);
                
                // Use the resources
                var data = CreateTestData(1000);
                buffer.CopyFromCPU(data);
                var result = buffer.GetAsArray1D();
                
                Assert.Equal(1000, result.Length);
            }
            
            // If we get here without running out of memory, cleanup is working
            Assert.True(true, "Resource cleanup test completed successfully");
        }

        #endregion

        #region End-to-End Workflow Tests

        [Fact]
        public void TestScientificComputingWorkflow()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Simulate a scientific computing workflow:
            // 1. Load data
            // 2. Preprocess with FFT
            // 3. Solve sparse linear system
            // 4. Validate results with hash
            
            const int dataSize = 256;
            
            // Step 1: Generate synthetic scientific data
            var rawData = new Complex[dataSize];
            for (int i = 0; i < dataSize; i++)
            {
                // Simulate noisy signal with multiple frequency components
                var signal = Math.Sin(2 * Math.PI * i / 32.0) + 0.5 * Math.Sin(2 * Math.PI * i / 8.0);
                var noise = (new Random(i).NextDouble() - 0.5) * 0.1;
                rawData[i] = new Complex(signal + noise, 0);
            }
            
            // Step 2: FFT preprocessing for frequency analysis
            using var dataBuffer = accelerator!.Allocate1D(rawData);
            using var fftBuffer = accelerator!.Allocate1D<Complex>(dataSize);
            
            var fft = new FFT<float>(accelerator!);
            fft.Forward1D(dataBuffer.View, fftBuffer.View);
            
            // Step 3: Extract dominant frequencies and create sparse system
            var fftResult = fftBuffer.GetAsArray1D();
            var dominantFreqs = fftResult
                .Select((c, i) => new { Index = i, Magnitude = c.Magnitude })
                .OrderByDescending(x => x.Magnitude)
                .Take(10)
                .ToArray();
            
            // Create sparse matrix from dominant frequency relationships
            var matrixSize = 10;
            var rowPtr = new int[matrixSize + 1];
            var colIndices = new System.Collections.Generic.List<int>();
            var values = new System.Collections.Generic.List<float>();
            
            for (int i = 0; i < matrixSize; i++)
            {
                rowPtr[i] = colIndices.Count;
                
                // Add diagonal element
                colIndices.Add(i);
                values.Add(2.0f + (float)dominantFreqs[i].Magnitude * 0.1f);
                
                // Add off-diagonal elements based on frequency relationships
                if (i > 0)
                {
                    colIndices.Add(i - 1);
                    values.Add(-0.5f);
                }
                if (i < matrixSize - 1)
                {
                    colIndices.Add(i + 1);
                    values.Add(-0.5f);
                }
            }
            rowPtr[matrixSize] = colIndices.Count;
            
            using var sparseMatrix = CSRMatrix<float>.Create(
                accelerator!, matrixSize, matrixSize, values.Count,
                rowPtr.ToArray(), colIndices.ToArray(), values.ToArray());
            
            // Step 4: Solve linear system Ax = b
            var rhs = dominantFreqs.Select(f => (float)f.Magnitude).ToArray();
            var solution = new float[matrixSize];
            
            using var rhsBuffer = accelerator!.Allocate1D(rhs);
            using var solutionBuffer = accelerator!.Allocate1D(solution);
            
            int iterations = SparseMatrixSolvers.ConjugateGradient(
                sparseMatrix, rhsBuffer.View, solutionBuffer.View,
                tolerance: 1e-6f, maxIterations: 100);
            
            Assert.True(iterations > 0 && iterations <= 100, "CG solver should converge");
            
            // Step 5: Validate and hash results
            var finalSolution = solutionBuffer.GetAsArray1D();
            var solutionBytes = finalSolution.SelectMany(f => BitConverter.GetBytes(f)).ToArray();
            
            using var solutionBytesBuffer = accelerator!.Allocate1D(solutionBytes);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            
            HashFunctions.SHA256(solutionBytesBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var resultHash = hashBuffer.GetAsArray1D();
            
            // Verify workflow completion
            Assert.True(resultHash.Any(b => b != 0), "Result hash should be computed");
            Assert.True(finalSolution.All(x => !float.IsNaN(x) && !float.IsInfinity(x)), 
                "Solution should be numerically valid");
        }

        [Fact]
        public void TestMachineLearningWorkflow()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Simulate ML workflow:
            // 1. Image preprocessing
            // 2. Feature extraction
            // 3. Data encryption for privacy
            // 4. Model computation simulation
            
            const int imageSize = 28; // MNIST-like
            const int numImages = 10;
            
            // Step 1: Create batch of synthetic images
            var imageData = new float[numImages * imageSize * imageSize];
            var random = new Random(42);
            
            for (int img = 0; img < numImages; img++)
            {
                for (int i = 0; i < imageSize * imageSize; i++)
                {
                    imageData[img * imageSize * imageSize + i] = (float)random.NextDouble();
                }
            }
            
            // Step 2: Batch image preprocessing
            using var imagesBuffer = accelerator!.Allocate1D(imageData);
            using var processedImagesBuffer = accelerator!.Allocate1D<float>(imageData.Length);
            
            ImageProcessing.BatchGaussianBlur(
                imagesBuffer.View, processedImagesBuffer.View,
                numImages, imageSize, imageSize, 1,
                kernelSize: 3, sigma: 1.0f, accelerator!.DefaultStream);
            
            // Step 3: Extract features (simulate with FFT)
            var features = new Complex[numImages * 64]; // Reduced feature size
            var processedData = processedImagesBuffer.GetAsArray1D();
            
            for (int img = 0; img < numImages; img++)
            {
                // Take subset of image for FFT
                for (int i = 0; i < 64; i++)
                {
                    var pixelValue = processedData[img * imageSize * imageSize + i];
                    features[img * 64 + i] = new Complex(pixelValue, 0);
                }
            }
            
            using var featuresBuffer = accelerator!.Allocate1D(features);
            using var fftFeaturesBuffer = accelerator!.Allocate1D<Complex>(features.Length);
            
            var fft = new FFT<float>(accelerator!);
            
            // Batch FFT for feature extraction
            for (int img = 0; img < numImages; img++)
            {
                var imgOffset = img * 64;
                fft.Forward1D(
                    featuresBuffer.View.SubView(imgOffset, 64),
                    fftFeaturesBuffer.View.SubView(imgOffset, 64));
            }
            
            // Step 4: Encrypt features for privacy-preserving ML
            var fftData = fftFeaturesBuffer.GetAsArray1D();
            var featureBytes = fftData.SelectMany(c => 
                BitConverter.GetBytes(c.Real).Concat(BitConverter.GetBytes(c.Imaginary))).ToArray();
            
            // Pad to block size
            var paddedSize = ((featureBytes.Length + 15) / 16) * 16;
            var paddedFeatures = new byte[paddedSize];
            Array.Copy(featureBytes, paddedFeatures, featureBytes.Length);
            
            var key = new byte[32];
            var iv = new byte[16];
            random.NextBytes(key);
            random.NextBytes(iv);
            
            using var featuresForEncryption = accelerator!.Allocate1D(paddedFeatures);
            using var encryptedFeatures = accelerator!.Allocate1D<byte>(paddedSize);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var ivBuffer = accelerator!.Allocate1D(iv);
            
            SymmetricCryptography.AESEncrypt(
                featuresForEncryption.View, encryptedFeatures.View,
                keyBuffer.View, ivBuffer.View,
                AESMode.CBC, accelerator!.DefaultStream);
            
            // Step 5: Verify ML workflow completion
            var encryptedData = encryptedFeatures.GetAsArray1D();
            
            Assert.True(encryptedData.Any(b => b != 0), "Encrypted features should not be all zeros");
            Assert.False(paddedFeatures.SequenceEqual(encryptedData), 
                "Encryption should change the feature data");
            
            // Verify we can decrypt and get back features
            using var decryptedFeatures = accelerator!.Allocate1D<byte>(paddedSize);
            SymmetricCryptography.AESDecrypt(
                encryptedFeatures.View, decryptedFeatures.View,
                keyBuffer.View, ivBuffer.View,
                AESMode.CBC, accelerator!.DefaultStream);
            
            var decryptedData = decryptedFeatures.GetAsArray1D();
            Assert.True(paddedFeatures.SequenceEqual(decryptedData), 
                "Decryption should restore original features");
        }

        #endregion
    }
}