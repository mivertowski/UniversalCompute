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

using ILGPU.Algorithms.ComputerVision;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.ComputerVision
{
    /// <summary>
    /// Tests for computer vision algorithms.
    /// </summary>
    public class ComputerVisionTests : TestBase
    {
        #region Image Creation and Basic Operations

        [Fact]
        public void TestImageCreation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 640;
            const int height = 480;
            const int channels = 3;
            
            using var image = Image<byte>.Create(accelerator!, width, height, channels);
            
            Assert.Equal(width, image.Width);
            Assert.Equal(height, image.Height);
            Assert.Equal(channels, image.Channels);
            Assert.Equal(width * height * channels, image.Data.Length);
        }

        [Fact]
        public void TestImageDataAccess()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 4;
            const int height = 4;
            const int channels = 1;
            
            var testData = new byte[width * height * channels];
            for (int i = 0; i < testData.Length; i++)
                testData[i] = (byte)(i % 256);
            
            using var image = Image<byte>.Create(accelerator!, width, height, channels, testData);
            
            var retrievedData = image.Data.GetAsArray1D();
            
            for (int i = 0; i < testData.Length; i++)
            {
                Assert.Equal(testData[i], retrievedData[i]);
            }
        }

        [Fact]
        public void TestPixelAccessor()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 3;
            const int height = 3;
            const int channels = 3;
            
            using var image = Image<byte>.Create(accelerator!, width, height, channels);
            
            // Test setting specific pixel values
            var testPixel = new Pixel<byte> { R = 255, G = 128, B = 64 };
            
            using var pixelBuffer = accelerator!.Allocate1D<Pixel<byte>>(1);
            pixelBuffer.CopyFromCPU(new[] { testPixel });
            
            ImageProcessing.SetPixel(image, 1, 1, pixelBuffer.View.AsArrayView(), accelerator!.DefaultStream);
            
            var resultData = image.Data.GetAsArray1D();
            
            // Verify pixel was set correctly (RGB format)
            int pixelIndex = (1 * width + 1) * channels;
            Assert.Equal(255, resultData[pixelIndex + 0]); // R
            Assert.Equal(128, resultData[pixelIndex + 1]); // G
            Assert.Equal(64, resultData[pixelIndex + 2]);  // B
        }

        #endregion

        #region Color Space Conversions

        [Fact]
        public void TestRGBToGrayscaleConversion()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 2;
            const int height = 2;
            
            // Create RGB image: red, green, blue, white pixels
            var rgbData = new byte[]
            {
                255, 0, 0,    // Red pixel
                0, 255, 0,    // Green pixel
                0, 0, 255,    // Blue pixel
                255, 255, 255 // White pixel
            };
            
            using var rgbImage = Image<byte>.Create(accelerator!, width, height, 3, rgbData);
            using var grayImage = Image<byte>.Create(accelerator!, width, height, 1);
            
            ImageProcessing.ConvertToGrayscale(rgbImage, grayImage, accelerator!.DefaultStream);
            
            var grayData = grayImage.Data.GetAsArray1D();
            
            // Verify grayscale conversion using standard weights: 0.299*R + 0.587*G + 0.114*B
            Assert.Equal((byte)(0.299 * 255), grayData[0]); // Red
            Assert.Equal((byte)(0.587 * 255), grayData[1]); // Green
            Assert.Equal((byte)(0.114 * 255), grayData[2]); // Blue
            Assert.Equal(255, grayData[3]); // White
        }

        [Fact]
        public void TestRGBToHSVConversion()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 2;
            const int height = 1;
            
            // Pure red and pure green pixels
            var rgbData = new byte[]
            {
                255, 0, 0,   // Pure red: H=0, S=1, V=1
                0, 255, 0    // Pure green: H=120, S=1, V=1
            };
            
            using var rgbImage = Image<byte>.Create(accelerator!, width, height, 3, rgbData);
            using var hsvImage = Image<float>.Create(accelerator!, width, height, 3);
            
            ImageProcessing.ConvertRGBToHSV(rgbImage, hsvImage, accelerator!.DefaultStream);
            
            var hsvData = hsvImage.Data.GetAsArray1D();
            
            // Check pure red (H=0, S=1, V=1)
            Assert.True(Math.Abs(hsvData[0] - 0.0f) < 1e-5f, $"Red hue should be 0, got {hsvData[0]}");
            Assert.True(Math.Abs(hsvData[1] - 1.0f) < 1e-5f, $"Red saturation should be 1, got {hsvData[1]}");
            Assert.True(Math.Abs(hsvData[2] - 1.0f) < 1e-5f, $"Red value should be 1, got {hsvData[2]}");
            
            // Check pure green (H=120/360, S=1, V=1)
            Assert.True(Math.Abs(hsvData[3] - (120.0f / 360.0f)) < 1e-2f, $"Green hue should be ~0.33, got {hsvData[3]}");
            Assert.True(Math.Abs(hsvData[4] - 1.0f) < 1e-5f, $"Green saturation should be 1, got {hsvData[4]}");
            Assert.True(Math.Abs(hsvData[5] - 1.0f) < 1e-5f, $"Green value should be 1, got {hsvData[5]}");
        }

        [Fact]
        public void TestYUVConversion()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 2;
            const int height = 1;
            
            var rgbData = new byte[]
            {
                255, 255, 255, // White
                0, 0, 0         // Black
            };
            
            using var rgbImage = Image<byte>.Create(accelerator!, width, height, 3, rgbData);
            using var yuvImage = Image<float>.Create(accelerator!, width, height, 3);
            using var convertedRgbImage = Image<byte>.Create(accelerator!, width, height, 3);
            
            // RGB -> YUV -> RGB round trip
            ImageProcessing.ConvertRGBToYUV(rgbImage, yuvImage, accelerator!.DefaultStream);
            ImageProcessing.ConvertYUVToRGB(yuvImage, convertedRgbImage, accelerator!.DefaultStream);
            
            var originalData = rgbImage.Data.GetAsArray1D();
            var convertedData = convertedRgbImage.Data.GetAsArray1D();
            
            // Verify round-trip conversion preserves data (within tolerance)
            for (int i = 0; i < originalData.Length; i++)
            {
                Assert.True(Math.Abs(originalData[i] - convertedData[i]) <= 2,
                    $"Round-trip conversion failed at index {i}: {originalData[i]} vs {convertedData[i]}");
            }
        }

        #endregion

        #region Filtering Operations

        [Fact]
        public void TestGaussianBlur()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 5;
            const int height = 5;
            
            // Create image with single bright pixel in center
            var imageData = new float[width * height];
            imageData[12] = 1.0f; // Center pixel (2,2) in 5x5 image
            
            using var image = Image<float>.Create(accelerator!, width, height, 1, imageData);
            using var blurredImage = Image<float>.Create(accelerator!, width, height, 1);
            
            ImageProcessing.GaussianBlur(image, blurredImage, kernelSize: 3, sigma: 1.0f, accelerator!.DefaultStream);
            
            var blurredData = blurredImage.Data.GetAsArray1D();
            
            // Center should still be brightest, but neighbors should have non-zero values
            Assert.True(blurredData[12] > blurredData[11], "Center should be brighter than neighbor");
            Assert.True(blurredData[11] > 0, "Neighbors should have non-zero values after blur");
            Assert.True(blurredData[7] > 0, "Vertical neighbors should have non-zero values");
            
            // Check that blur spreads intensity
            var totalIntensity = blurredData.Sum();
            Assert.True(Math.Abs(totalIntensity - 1.0f) < 1e-5f, "Total intensity should be preserved");
        }

        [Fact]
        public void TestSobelEdgeDetection()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 5;
            const int height = 5;
            
            // Create image with vertical edge (left half black, right half white)
            var imageData = new float[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    imageData[y * width + x] = (x >= width / 2) ? 1.0f : 0.0f;
                }
            }
            
            using var image = Image<float>.Create(accelerator!, width, height, 1, imageData);
            using var gradientX = Image<float>.Create(accelerator!, width, height, 1);
            using var gradientY = Image<float>.Create(accelerator!, width, height, 1);
            
            ImageProcessing.SobelEdgeDetection(image, gradientX, gradientY, accelerator!.DefaultStream);
            
            var gradientXData = gradientX.Data.GetAsArray1D();
            var gradientYData = gradientY.Data.GetAsArray1D();
            
            // For vertical edge, X gradient should be strong, Y gradient should be weak
            int centerIndex = 2 * width + 2; // Center pixel
            Assert.True(Math.Abs(gradientXData[centerIndex]) > 0.1f, 
                $"X gradient should be strong for vertical edge, got {gradientXData[centerIndex]}");
            Assert.True(Math.Abs(gradientYData[centerIndex]) < 0.5f, 
                $"Y gradient should be weak for vertical edge, got {gradientYData[centerIndex]}");
        }

        [Fact]
        public void TestMedianFilter()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 5;
            const int height = 5;
            
            // Create image with salt-and-pepper noise
            var imageData = new float[width * height];
            for (int i = 0; i < imageData.Length; i++)
                imageData[i] = 0.5f; // Gray background
            
            // Add noise
            imageData[6] = 1.0f;  // Salt
            imageData[8] = 0.0f;  // Pepper
            imageData[12] = 1.0f; // Center salt
            
            using var noisyImage = Image<float>.Create(accelerator!, width, height, 1, imageData);
            using var filteredImage = Image<float>.Create(accelerator!, width, height, 1);
            
            ImageProcessing.MedianFilter(noisyImage, filteredImage, kernelSize: 3, accelerator!.DefaultStream);
            
            var filteredData = filteredImage.Data.GetAsArray1D();
            
            // Median filter should remove salt-and-pepper noise
            // Center pixel should be closer to background value
            Assert.True(Math.Abs(filteredData[12] - 0.5f) < Math.Abs(imageData[12] - 0.5f),
                "Median filter should reduce noise in center pixel");
        }

        [Fact]
        public void TestBilateralFilter()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 7;
            const int height = 7;
            
            // Create image with step edge and noise
            var imageData = new float[width * height];
            var random = new Random(42);
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float baseValue = (x >= width / 2) ? 1.0f : 0.0f;
                    float noise = (float)(random.NextDouble() - 0.5) * 0.1f;
                    imageData[y * width + x] = baseValue + noise;
                }
            }
            
            using var noisyImage = Image<float>.Create(accelerator!, width, height, 1, imageData);
            using var filteredImage = Image<float>.Create(accelerator!, width, height, 1);
            
            ImageProcessing.BilateralFilter(
                noisyImage, filteredImage, 
                kernelSize: 5, sigmaSpace: 1.0f, sigmaRange: 0.2f, 
                accelerator!.DefaultStream);
            
            var filteredData = filteredImage.Data.GetAsArray1D();
            
            // Bilateral filter should preserve edges while reducing noise
            // Check that the step edge is preserved
            int leftSide = 1 * width + 1;   // Left side of image
            int rightSide = 1 * width + 5;  // Right side of image
            
            Assert.True(filteredData[rightSide] > filteredData[leftSide],
                "Bilateral filter should preserve step edge");
        }

        #endregion

        #region Geometric Transformations

        [Fact]
        public void TestImageRotation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 4;
            
            // Create simple pattern
            var imageData = new float[]
            {
                1, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0
            };
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var rotatedImage = Image<float>.Create(accelerator!, size, size, 1);
            
            // Rotate 90 degrees clockwise
            ImageProcessing.RotateImage(image, rotatedImage, 90.0f, accelerator!.DefaultStream);
            
            var rotatedData = rotatedImage.Data.GetAsArray1D();
            
            // After 90-degree rotation, bright pixel should move from (0,0) to (0,3)
            Assert.True(rotatedData[3] > 0.5f, "Rotated pixel should be at expected position");
            Assert.True(rotatedData[0] < 0.5f, "Original position should be dark after rotation");
        }

        [Fact]
        public void TestImageScaling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int originalSize = 2;
            const int scaledSize = 4;
            
            var imageData = new float[]
            {
                1.0f, 0.0f,
                0.0f, 1.0f
            };
            
            using var originalImage = Image<float>.Create(accelerator!, originalSize, originalSize, 1, imageData);
            using var scaledImage = Image<float>.Create(accelerator!, scaledSize, scaledSize, 1);
            
            ImageProcessing.ScaleImage(originalImage, scaledImage, 
                InterpolationMethod.Bilinear, accelerator!.DefaultStream);
            
            var scaledData = scaledImage.Data.GetAsArray1D();
            
            // Scaled image should preserve general pattern
            Assert.True(scaledData[0] > 0.5f, "Top-left should remain bright");
            Assert.True(scaledData[15] > 0.5f, "Bottom-right should remain bright");
            Assert.True(scaledData[1] < 0.8f, "Interpolated regions should have intermediate values");
        }

        [Fact]
        public void TestAffineTransformation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 4;
            
            var imageData = new float[size * size];
            imageData[5] = 1.0f; // Pixel at (1,1)
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var transformedImage = Image<float>.Create(accelerator!, size, size, 1);
            
            // Simple translation matrix: move by (1,1)
            var transformMatrix = new float[]
            {
                1, 0, 1,  // [1 0 tx]
                0, 1, 1,  // [0 1 ty]
                0, 0, 1   // [0 0  1]
            };
            
            using var transformBuffer = accelerator!.Allocate1D(transformMatrix);
            
            ImageProcessing.AffineTransform(image, transformedImage, transformBuffer.View, accelerator!.DefaultStream);
            
            var transformedData = transformedImage.Data.GetAsArray1D();
            
            // Pixel should move from (1,1) to (2,2)
            Assert.True(transformedData[10] > 0.5f, "Transformed pixel should be at new position (2,2)");
            Assert.True(transformedData[5] < 0.1f, "Original position should be empty");
        }

        #endregion

        #region Feature Detection

        [Fact]
        public void TestHarrisCornerDetection()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 7;
            
            // Create image with corner pattern
            var imageData = new float[size * size];
            
            // Create L-shaped corner
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    if ((x <= 3 && y <= 3) || (x <= 3 && y >= 3))
                        imageData[y * size + x] = 1.0f;
                }
            }
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var cornerResponse = Image<float>.Create(accelerator!, size, size, 1);
            
            FeatureDetection.HarrisCornerDetection(image, cornerResponse, 
                threshold: 0.1f, k: 0.04f, accelerator!.DefaultStream);
            
            var responseData = cornerResponse.Data.GetAsArray1D();
            
            // Corner should be detected at the L-junction
            int cornerIndex = 3 * size + 3; // Position (3,3)
            Assert.True(responseData[cornerIndex] > 0.05f, 
                $"Harris corner should be detected at junction, response: {responseData[cornerIndex]}");
            
            // Smooth regions should have low response
            int smoothIndex = 1 * size + 1; // Position (1,1) - inside bright region
            Assert.True(responseData[smoothIndex] < responseData[cornerIndex],
                "Smooth regions should have lower corner response");
        }

        [Fact]
        public void TestCannyEdgeDetection()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 7;
            
            // Create image with clear edge
            var imageData = new float[size * size];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    imageData[y * size + x] = (x >= size / 2) ? 1.0f : 0.0f;
                }
            }
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var edges = Image<float>.Create(accelerator!, size, size, 1);
            
            FeatureDetection.CannyEdgeDetection(image, edges,
                lowThreshold: 0.1f, highThreshold: 0.3f, 
                sigma: 1.0f, accelerator!.DefaultStream);
            
            var edgeData = edges.Data.GetAsArray1D();
            
            // Edge should be detected at boundary
            int edgeIndex = 3 * size + 3; // Middle of image near edge
            Assert.True(edgeData[edgeIndex] > 0.5f, 
                $"Canny should detect edge, got {edgeData[edgeIndex]}");
            
            // Away from edge should be background
            int backgroundIndex = 3 * size + 1; // Left side, away from edge
            Assert.True(edgeData[backgroundIndex] < 0.1f,
                $"Background should not have edges, got {edgeData[backgroundIndex]}");
        }

        [Fact]
        public void TestBlobDetection()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 9;
            
            // Create image with circular blob
            var imageData = new float[size * size];
            int center = size / 2;
            
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float distance = (float)Math.Sqrt((x - center) * (x - center) + (y - center) * (y - center));
                    if (distance <= 2.0f)
                        imageData[y * size + x] = 1.0f - distance / 2.0f; // Gaussian-like blob
                }
            }
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var blobResponse = Image<float>.Create(accelerator!, size, size, 1);
            
            FeatureDetection.BlobDetection(image, blobResponse,
                minRadius: 1.0f, maxRadius: 3.0f, threshold: 0.1f, accelerator!.DefaultStream);
            
            var responseData = blobResponse.Data.GetAsArray1D();
            
            // Blob should be detected at center
            int centerIndex = center * size + center;
            Assert.True(responseData[centerIndex] > 0.1f,
                $"Blob should be detected at center, response: {responseData[centerIndex]}");
            
            // Corners should have low response
            Assert.True(responseData[0] < responseData[centerIndex],
                "Corners should have lower blob response than center");
        }

        #endregion

        #region Morphological Operations

        [Fact]
        public void TestMorphologicalErosion()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 5;
            
            // Create binary image with small white square
            var imageData = new float[size * size];
            for (int y = 1; y <= 3; y++)
            {
                for (int x = 1; x <= 3; x++)
                {
                    imageData[y * size + x] = 1.0f;
                }
            }
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var erodedImage = Image<float>.Create(accelerator!, size, size, 1);
            
            // 3x3 square structuring element
            var structuringElement = new float[]
            {
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
            };
            
            using var structBuffer = accelerator!.Allocate1D(structuringElement);
            
            ImageProcessing.MorphologicalErosion(image, erodedImage, structBuffer.View, 3, accelerator!.DefaultStream);
            
            var erodedData = erodedImage.Data.GetAsArray1D();
            
            // Erosion should shrink the white region
            int centerIndex = 2 * size + 2; // Center should still be white
            Assert.True(erodedData[centerIndex] > 0.5f, "Center should remain after erosion");
            
            int edgeIndex = 1 * size + 1; // Edge should be eroded
            Assert.True(erodedData[edgeIndex] < 0.5f, "Edges should be eroded away");
        }

        [Fact]
        public void TestMorphologicalDilation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 5;
            
            // Create binary image with single white pixel
            var imageData = new float[size * size];
            imageData[2 * size + 2] = 1.0f; // Center pixel
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var dilatedImage = Image<float>.Create(accelerator!, size, size, 1);
            
            // 3x3 square structuring element
            var structuringElement = new float[]
            {
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
            };
            
            using var structBuffer = accelerator!.Allocate1D(structuringElement);
            
            ImageProcessing.MorphologicalDilation(image, dilatedImage, structBuffer.View, 3, accelerator!.DefaultStream);
            
            var dilatedData = dilatedImage.Data.GetAsArray1D();
            
            // Dilation should expand the white region
            Assert.True(dilatedData[2 * size + 2] > 0.5f, "Center should remain white");
            Assert.True(dilatedData[1 * size + 2] > 0.5f, "Neighbors should become white");
            Assert.True(dilatedData[2 * size + 1] > 0.5f, "Neighbors should become white");
            
            // Corners should remain black
            Assert.True(dilatedData[0] < 0.5f, "Distant corners should remain black");
        }

        [Fact]
        public void TestMorphologicalOpening()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 7;
            
            // Create binary image with noise and main structure
            var imageData = new float[size * size];
            
            // Main structure (4x4 square)
            for (int y = 1; y <= 4; y++)
            {
                for (int x = 1; x <= 4; x++)
                {
                    imageData[y * size + x] = 1.0f;
                }
            }
            
            // Add noise (single pixel)
            imageData[0 * size + 6] = 1.0f;
            imageData[6 * size + 0] = 1.0f;
            
            using var image = Image<float>.Create(accelerator!, size, size, 1, imageData);
            using var openedImage = Image<float>.Create(accelerator!, size, size, 1);
            
            // 3x3 square structuring element
            var structuringElement = new float[]
            {
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
            };
            
            using var structBuffer = accelerator!.Allocate1D(structuringElement);
            
            ImageProcessing.MorphologicalOpening(image, openedImage, structBuffer.View, 3, accelerator!.DefaultStream);
            
            var openedData = openedImage.Data.GetAsArray1D();
            
            // Opening should remove noise but preserve main structure
            Assert.True(openedData[0 * size + 6] < 0.5f, "Noise pixels should be removed");
            Assert.True(openedData[6 * size + 0] < 0.5f, "Noise pixels should be removed");
            Assert.True(openedData[2 * size + 2] > 0.5f, "Main structure should be preserved");
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestLargeImageProcessingPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int width = 1920;
            const int height = 1080;
            const int channels = 3;
            
            var imageData = CreateTestData(width * height * channels);
            
            using var image = Image<float>.Create(accelerator!, width, height, channels, imageData);
            using var blurredImage = Image<float>.Create(accelerator!, width, height, channels);
            
            // Measure Gaussian blur performance on HD image
            var blurTime = MeasureTime(() =>
            {
                ImageProcessing.GaussianBlur(image, blurredImage, kernelSize: 5, sigma: 1.0f, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            Assert.True(blurTime < 5000, $"HD image blur took {blurTime}ms, expected < 5000ms");
            
            // Verify result is not all zeros
            var result = blurredImage.Data.GetAsArray1D();
            Assert.True(result.Any(x => Math.Abs(x) > 1e-6f), "Processed image should not be all zeros");
        }

        [Fact]
        public void TestBatchImageProcessing()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int batchSize = 10;
            const int width = 256;
            const int height = 256;
            const int channels = 3;
            
            var batchData = new float[batchSize * width * height * channels];
            var random = new Random(42);
            for (int i = 0; i < batchData.Length; i++)
                batchData[i] = (float)random.NextDouble();
            
            using var batchBuffer = accelerator!.Allocate1D(batchData);
            using var processedBatchBuffer = accelerator!.Allocate1D<float>(batchData.Length);
            
            // Measure batch processing performance
            var batchTime = MeasureTime(() =>
            {
                ImageProcessing.BatchGaussianBlur(
                    batchBuffer.View, processedBatchBuffer.View,
                    batchSize, width, height, channels,
                    kernelSize: 3, sigma: 1.0f, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            Assert.True(batchTime < 3000, $"Batch processing took {batchTime}ms, expected < 3000ms");
            
            var processedData = processedBatchBuffer.GetAsArray1D();
            Assert.True(processedData.Any(x => Math.Abs(x) > 1e-6f), "Batch processing should produce non-zero results");
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestInvalidImageDimensions()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test invalid dimensions
            Assert.Throws<ArgumentException>(() =>
            {
                Image<byte>.Create(accelerator!, 0, 100, 3);
            });
            
            Assert.Throws<ArgumentException>(() =>
            {
                Image<byte>.Create(accelerator!, 100, 0, 3);
            });
            
            Assert.Throws<ArgumentException>(() =>
            {
                Image<byte>.Create(accelerator!, 100, 100, 0);
            });
        }

        [Fact]
        public void TestIncompatibleImageOperations()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            using var image1 = Image<float>.Create(accelerator!, 100, 100, 3);
            using var image2 = Image<float>.Create(accelerator!, 50, 50, 3);
            
            // Test operations with incompatible dimensions
            Assert.Throws<ArgumentException>(() =>
            {
                ImageProcessing.GaussianBlur(image1, image2, kernelSize: 3, sigma: 1.0f, accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestInvalidKernelSize()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            using var image = Image<float>.Create(accelerator!, 100, 100, 1);
            
            // Test invalid kernel sizes
            Assert.Throws<ArgumentException>(() =>
            {
                ImageProcessing.GaussianBlur(image, image, kernelSize: 0, sigma: 1.0f, accelerator!.DefaultStream);
            });
            
            Assert.Throws<ArgumentException>(() =>
            {
                ImageProcessing.GaussianBlur(image, image, kernelSize: 2, sigma: 1.0f, accelerator!.DefaultStream); // Even size
            });
        }

        #endregion
    }
}