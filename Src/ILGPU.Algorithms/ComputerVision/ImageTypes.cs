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

namespace ILGPU.Algorithms.ComputerVision
{
    /// <summary>
    /// Pixel formats for computer vision operations.
    /// </summary>
    public enum PixelFormat
    {
        /// <summary>8-bit grayscale</summary>
        Gray8,
        /// <summary>16-bit grayscale</summary>
        Gray16,
        /// <summary>32-bit floating-point grayscale</summary>
        GrayF32,
        /// <summary>24-bit RGB</summary>
        RGB24,
        /// <summary>32-bit RGBA</summary>
        RGBA32,
        /// <summary>32-bit BGRA</summary>
        BGRA32,
        /// <summary>96-bit RGB floating-point</summary>
        RGBF32,
        /// <summary>128-bit RGBA floating-point</summary>
        RGBAF32
    }

    /// <summary>
    /// Border handling modes for image processing operations.
    /// </summary>
    public enum BorderMode
    {
        /// <summary>Constant border (zero padding)</summary>
        Constant,
        /// <summary>Replicate edge pixels</summary>
        Replicate,
        /// <summary>Reflect pixels across border</summary>
        Reflect,
        /// <summary>Wrap around (periodic)</summary>
        Wrap,
        /// <summary>Mirror border</summary>
        Mirror
    }

    /// <summary>
    /// Interpolation methods for image resampling.
    /// </summary>
    public enum InterpolationMode
    {
        /// <summary>Nearest neighbor interpolation</summary>
        Nearest,
        /// <summary>Bilinear interpolation</summary>
        Linear,
        /// <summary>Bicubic interpolation</summary>
        Cubic,
        /// <summary>Lanczos interpolation</summary>
        Lanczos
    }

    /// <summary>
    /// GPU-based image representation for computer vision operations.
    /// </summary>
    /// <typeparam name="T">Pixel data type.</typeparam>
    public sealed class Image<T> : IDisposable where T : unmanaged
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new image.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        /// <param name="channels">Number of channels (1=grayscale, 3=RGB, 4=RGBA).</param>
        /// <param name="data">Optional initial pixel data.</param>
        public Image(Accelerator accelerator, int width, int height, int channels, T[]? data = null)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            if (width <= 0) throw new ArgumentException("Width must be positive", nameof(width));
            if (height <= 0) throw new ArgumentException("Height must be positive", nameof(height));
            if (channels <= 0 || channels > 4) throw new ArgumentException("Channels must be 1-4", nameof(channels));

            Width = width;
            Height = height;
            Channels = channels;
            
            var totalPixels = width * height * channels;
            if (data != null)
            {
                if (data.Length != totalPixels)
                    throw new ArgumentException($"Data array must have {totalPixels} elements");
                Data = Accelerator.Allocate1D(data);
            }
            else
            {
                Data = Accelerator.Allocate1D<T>(totalPixels);
            }
        }

        /// <summary>
        /// Gets the image width in pixels.
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// Gets the image height in pixels.
        /// </summary>
        public int Height { get; }

        /// <summary>
        /// Gets the number of channels.
        /// </summary>
        public int Channels { get; }

        /// <summary>
        /// Gets the total number of pixels.
        /// </summary>
        public int PixelCount => Width * Height;

        /// <summary>
        /// Gets the stride (width * channels).
        /// </summary>
        public int Stride => Width * Channels;

        /// <summary>
        /// Gets the pixel data buffer.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Data { get; }

        /// <summary>
        /// Gets the accelerator associated with this image.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Creates a view of the image data as a 2D array (for single-channel images).
        /// </summary>
        /// <returns>2D array view.</returns>
        public ArrayView2D<T, Stride2D.DenseX> As2D()
        {
            if (Channels != 1)
                throw new InvalidOperationException("2D view only supported for single-channel images");
            
            return Data.View.As2DDenseXView(new Index2D(Width, Height));
        }

        /// <summary>
        /// Creates a view of the image data as a 3D array (for multi-channel images).
        /// </summary>
        /// <returns>3D array view.</returns>
        public ArrayView3D<T, Stride3D.DenseXY> As3D()
        {
            return Data.View.As3DDenseXYView(new Index3D(Width, Height, Channels));
        }

        /// <summary>
        /// Gets a pixel value at the specified coordinates.
        /// </summary>
        /// <param name="x">X coordinate.</param>
        /// <param name="y">Y coordinate.</param>
        /// <param name="channel">Channel index (0-based).</param>
        /// <returns>Pixel value.</returns>
        public T GetPixel(int x, int y, int channel = 0)
        {
            if (x < 0 || x >= Width) throw new ArgumentOutOfRangeException(nameof(x));
            if (y < 0 || y >= Height) throw new ArgumentOutOfRangeException(nameof(y));
            if (channel < 0 || channel >= Channels) throw new ArgumentOutOfRangeException(nameof(channel));

            var index = y * Stride + x * Channels + channel;
            var hostData = new T[1];
            Data.View.SubView(index, 1).CopyToCPU(hostData);
            return hostData[0];
        }

        /// <summary>
        /// Sets a pixel value at the specified coordinates.
        /// </summary>
        /// <param name="x">X coordinate.</param>
        /// <param name="y">Y coordinate.</param>
        /// <param name="value">Pixel value.</param>
        /// <param name="channel">Channel index (0-based).</param>
        public void SetPixel(int x, int y, T value, int channel = 0)
        {
            if (x < 0 || x >= Width) throw new ArgumentOutOfRangeException(nameof(x));
            if (y < 0 || y >= Height) throw new ArgumentOutOfRangeException(nameof(y));
            if (channel < 0 || channel >= Channels) throw new ArgumentOutOfRangeException(nameof(channel));

            var index = y * Stride + x * Channels + channel;
            var hostData = new T[] { value };
            Data.View.SubView(index, 1).CopyFromCPU(hostData);
        }

        /// <summary>
        /// Creates a copy of this image.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Copied image.</returns>
        public Image<T> Clone(AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? Accelerator.DefaultStream;
            var cloned = new Image<T>(Accelerator, Width, Height, Channels);
            Data.View.CopyTo(cloned.Data.View);
            actualStream.Synchronize();
            return cloned;
        }

        /// <summary>
        /// Converts image to host memory.
        /// </summary>
        /// <returns>Pixel data as host array.</returns>
        public T[] ToHostArray()
        {
            var hostData = new T[Width * Height * Channels];
            Data.CopyToCPU(hostData);
            return hostData;
        }

        /// <summary>
        /// Creates an image from host data.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="width">Image width.</param>
        /// <param name="height">Image height.</param>
        /// <param name="channels">Number of channels.</param>
        /// <param name="data">Host pixel data.</param>
        /// <returns>New image.</returns>
#pragma warning disable CA1000 // Do not declare static members on generic types
        public static Image<T> FromHostArray(Accelerator accelerator, int width, int height, int channels, T[] data)
#pragma warning restore CA1000 // Do not declare static members on generic types
        {
            return new Image<T>(accelerator, width, height, channels, data);
        }

        /// <summary>
        /// Disposes the image and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                Data?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// RGB pixel structure.
    /// </summary>
    public struct RGB24
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public byte R, G, B;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="RGB24"/> struct.
        /// </summary>
        /// <param name="r">The r.</param>
        /// <param name="g">The g.</param>
        /// <param name="b">The b.</param>
        public RGB24(byte r, byte g, byte b)
        {
            R = r; G = g; B = b;
        }
    }

    /// <summary>
    /// RGBA pixel structure.
    /// </summary>
    public struct RGBA32
    {
        /// <summary>
        /// 
        /// </summary>

#pragma warning disable CA1051 // Do not declare visible instance fields
        public byte R, G, B, A;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="RGBA32"/> struct.
        /// </summary>
        /// <param name="r">The r.</param>
        /// <param name="g">The g.</param>
        /// <param name="b">The b.</param>
        /// <param name="a">a.</param>
        public RGBA32(byte r, byte g, byte b, byte a = 255)
        {
            R = r; G = g; B = b; A = a;
        }
    }

    /// <summary>
    /// Floating-point RGB pixel structure.
    /// </summary>
    public struct RGBF32
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public float R, G, B;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="RGBF32"/> struct.
        /// </summary>
        /// <param name="r">The r.</param>
        /// <param name="g">The g.</param>
        /// <param name="b">The b.</param>
        public RGBF32(float r, float g, float b)
        {
            R = r; G = g; B = b;
        }
    }

    /// <summary>
    /// Floating-point RGBA pixel structure.
    /// </summary>
    public struct RGBAF32
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public float R, G, B, A;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="RGBAF32"/> struct.
        /// </summary>
        /// <param name="r">The r.</param>
        /// <param name="g">The g.</param>
        /// <param name="b">The b.</param>
        /// <param name="a">a.</param>
        public RGBAF32(float r, float g, float b, float a = 1.0f)
        {
            R = r; G = g; B = b; A = a;
        }
    }

    /// <summary>
    /// 2D point structure for computer vision operations.
    /// </summary>
    public struct Point2D
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public float X, Y;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="Point2D"/> struct.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        public Point2D(float x, float y)
        {
            X = x; Y = y;
        }
    }

    /// <summary>
    /// Rectangle structure for region of interest operations.
    /// </summary>
    public struct Rectangle
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public int X, Y, Width, Height;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="Rectangle"/> struct.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rectangle(int x, int y, int width, int height)
        {
            X = x; Y = y; Width = width; Height = height;
        }

        /// <summary>
        /// Gets the right.
        /// </summary>
        /// <value>
        /// The right.
        /// </value>
        public int Right => X + Width;

        /// <summary>
        /// Gets the bottom.
        /// </summary>
        /// <value>
        /// The bottom.
        /// </value>
        public int Bottom => Y + Height;

        /// <summary>
        /// Gets the center.
        /// </summary>
        /// <value>
        /// The center.
        /// </value>
        public Point2D Center => new Point2D(X + Width / 2.0f, Y + Height / 2.0f);
    }

    /// <summary>
    /// Kernel structure for convolution operations.
    /// </summary>
    /// <typeparam name="T">Kernel coefficient type.</typeparam>
    public sealed class ConvolutionKernel<T> : IDisposable where T : unmanaged
    {
        /// <summary>
        /// Kernel width.
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// Kernel height.
        /// </summary>
        public int Height { get; }

        /// <summary>
        /// Kernel center X offset.
        /// </summary>
        public int CenterX { get; }

        /// <summary>
        /// Kernel center Y offset.
        /// </summary>
        public int CenterY { get; }

        /// <summary>
        /// Kernel coefficients.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Coefficients { get; }

        /// <summary>
        /// Accelerator instance.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Initializes a new convolution kernel.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="width">Kernel width.</param>
        /// <param name="height">Kernel height.</param>
        /// <param name="coefficients">Kernel coefficients.</param>
        /// <param name="centerX">Center X offset (default: width/2).</param>
        /// <param name="centerY">Center Y offset (default: height/2).</param>
        public ConvolutionKernel(Accelerator accelerator, int width, int height, T[] coefficients, 
            int? centerX = null, int? centerY = null)
        {
            if (coefficients.Length != width * height)
                throw new ArgumentException("Coefficients array size must match width * height");

            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Width = width;
            Height = height;
            CenterX = centerX ?? width / 2;
            CenterY = centerY ?? height / 2;
            Coefficients = accelerator.Allocate1D(coefficients);
        }

        /// <summary>
        /// Creates a Gaussian blur kernel.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="size">Kernel size (must be odd).</param>
        /// <param name="sigma">Gaussian standard deviation.</param>
        /// <returns>Gaussian kernel.</returns>
#pragma warning disable CA1000 // Do not declare static members on generic types
        public static ConvolutionKernel<float> CreateGaussian(Accelerator accelerator, int size, float sigma)
#pragma warning restore CA1000 // Do not declare static members on generic types
        {
            if (size % 2 == 0) throw new ArgumentException("Kernel size must be odd");

            var coefficients = new float[size * size];
            var center = size / 2;
            var sum = 0.0f;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    var dx = x - center;
                    var dy = y - center;
                    var value = (float)Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                    coefficients[y * size + x] = value;
                    sum += value;
                }
            }

            // Normalize
            for (int i = 0; i < coefficients.Length; i++)
                coefficients[i] /= sum;

            return new ConvolutionKernel<float>(accelerator, size, size, coefficients);
        }

        /// <summary>
        /// Creates a Sobel edge detection kernel.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="direction">Gradient direction (0=X, 1=Y).</param>
        /// <returns>Sobel kernel.</returns>
#pragma warning disable CA1000 // Do not declare static members on generic types
        public static ConvolutionKernel<float> CreateSobel(Accelerator accelerator, int direction)
#pragma warning restore CA1000 // Do not declare static members on generic types
        {
            float[] coefficients;
            
            if (direction == 0) // X direction
            {
                coefficients = new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
            }
            else // Y direction
            {
                coefficients = new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
            }

            return new ConvolutionKernel<float>(accelerator, 3, 3, coefficients);
        }

        /// <summary>
        /// Creates a box filter kernel.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="size">Kernel size.</param>
        /// <returns>Box filter kernel.</returns>
#pragma warning disable CA1000 // Do not declare static members on generic types
        public static ConvolutionKernel<float> CreateBox(Accelerator accelerator, int size)
#pragma warning restore CA1000 // Do not declare static members on generic types
        {
            var value = 1.0f / (size * size);
            var coefficients = new float[size * size];
            for (int i = 0; i < coefficients.Length; i++)
                coefficients[i] = value;

            return new ConvolutionKernel<float>(accelerator, size, size, coefficients);
        }

        /// <summary>
        /// Disposes the kernel and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            Coefficients?.Dispose();
        }
    }
}
