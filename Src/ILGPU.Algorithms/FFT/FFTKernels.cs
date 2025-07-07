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
using System.Numerics;

namespace ILGPU.Algorithms.FFT
{
    /// <summary>
    /// GPU kernels for Fast Fourier Transform operations.
    /// </summary>
    public static class FFTKernels
    {
        #region 1D FFT Kernels

        /// <summary>
        /// Cooley-Tukey radix-2 forward FFT kernel.
        /// </summary>
        public static void CooleyTukeyForward1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            var n = input.Length;
            var logN = plan.LogN;

            // Bit-reversal permutation
            BitReversalPermutation(input, output, (int)n);

            // Cooley-Tukey FFT
            for (int s = 1; s <= logN; s++)
            {
                var m = 1 << s;
                var wm = Complex.FromPolarCoordinates(1.0, -2.0 * Math.PI / m);

                for (int k = 0; k < n; k += m)
                {
                    var w = Complex.One;
                    for (int j = 0; j < m / 2; j++)
                    {
                        var t = w * output[k + j + m / 2];
                        var u = output[k + j];
                        output[k + j] = u + t;
                        output[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            }
        }

        /// <summary>
        /// Cooley-Tukey radix-2 inverse FFT kernel.
        /// </summary>
        public static void CooleyTukeyInverse1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            var n = input.Length;
            var logN = plan.LogN;

            // Bit-reversal permutation
            BitReversalPermutation(input, output, (int)n);

            // Cooley-Tukey inverse FFT
            for (int s = 1; s <= logN; s++)
            {
                var m = 1 << s;
                var wm = Complex.FromPolarCoordinates(1.0, 2.0 * Math.PI / m);

                for (int k = 0; k < n; k += m)
                {
                    var w = Complex.One;
                    for (int j = 0; j < m / 2; j++)
                    {
                        var t = w * output[k + j + m / 2];
                        var u = output[k + j];
                        output[k + j] = u + t;
                        output[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            }
        }

        /// <summary>
        /// Radix-4 forward FFT kernel.
        /// </summary>
        public static void Radix4Forward1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            var n = input.Length;
            var logN = plan.LogN;

            // Bit-reversal for radix-4
            BitReversalPermutation(input, output, (int)n);

            // Radix-4 FFT stages
            var stages = logN / 2;
            for (int stage = 0; stage < stages; stage++)
            {
                var stageSize = 1 << (2 * (stage + 1));
                var quarterStageSize = stageSize >> 2;
                var twoPiByStageSize = -2.0 * Math.PI / stageSize;

                for (int k = 0; k < n; k += stageSize)
                {
                    for (int j = 0; j < quarterStageSize; j++)
                    {
                        var idx0 = k + j;
                        var idx1 = idx0 + quarterStageSize;
                        var idx2 = idx1 + quarterStageSize;
                        var idx3 = idx2 + quarterStageSize;

                        var x0 = output[idx0];
                        var x1 = output[idx1];
                        var x2 = output[idx2];
                        var x3 = output[idx3];

                        // Radix-4 butterfly
                        var angle = twoPiByStageSize * j;
                        var w1 = Complex.FromPolarCoordinates(1.0, angle);
                        var w2 = Complex.FromPolarCoordinates(1.0, 2 * angle);
                        var w3 = Complex.FromPolarCoordinates(1.0, 3 * angle);

                        var t0 = x0 + x2;
                        var t1 = x0 - x2;
                        var t2 = x1 + x3;
                        var t3 = new Complex(-(x1 - x3).Imaginary, (x1 - x3).Real); // j*(x1-x3)

                        output[idx0] = t0 + t2;
                        output[idx1] = w1 * (t1 + t3);
                        output[idx2] = w2 * (t0 - t2);
                        output[idx3] = w3 * (t1 - t3);
                    }
                }
            }
        }

        /// <summary>
        /// Radix-4 inverse FFT kernel.
        /// </summary>
        public static void Radix4Inverse1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            // Similar to forward but with positive angle
            var n = input.Length;
            var logN = plan.LogN;

            BitReversalPermutation(input, output, (int)n);

            var stages = logN / 2;
            for (int stage = 0; stage < stages; stage++)
            {
                var stageSize = 1 << (2 * (stage + 1));
                var quarterStageSize = stageSize >> 2;
                var twoPiByStageSize = 2.0 * Math.PI / stageSize; // Positive for inverse

                for (int k = 0; k < n; k += stageSize)
                {
                    for (int j = 0; j < quarterStageSize; j++)
                    {
                        var idx0 = k + j;
                        var idx1 = idx0 + quarterStageSize;
                        var idx2 = idx1 + quarterStageSize;
                        var idx3 = idx2 + quarterStageSize;

                        var x0 = output[idx0];
                        var x1 = output[idx1];
                        var x2 = output[idx2];
                        var x3 = output[idx3];

                        var angle = twoPiByStageSize * j;
                        var w1 = Complex.FromPolarCoordinates(1.0, angle);
                        var w2 = Complex.FromPolarCoordinates(1.0, 2 * angle);
                        var w3 = Complex.FromPolarCoordinates(1.0, 3 * angle);

                        var t0 = x0 + x2;
                        var t1 = x0 - x2;
                        var t2 = x1 + x3;
                        var t3 = new Complex((x1 - x3).Imaginary, -(x1 - x3).Real); // -j*(x1-x3)

                        output[idx0] = t0 + t2;
                        output[idx1] = w1 * (t1 + t3);
                        output[idx2] = w2 * (t0 - t2);
                        output[idx3] = w3 * (t1 - t3);
                    }
                }
            }
        }

        /// <summary>
        /// Mixed-radix forward FFT kernel.
        /// </summary>
        public static void MixedRadixForward1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            // Implement mixed-radix FFT for non-power-of-2 sizes
            // This is a placeholder - real implementation would handle various factors
            CooleyTukeyForward1D(input, output, plan);
        }

        /// <summary>
        /// Mixed-radix inverse FFT kernel.
        /// </summary>
        public static void MixedRadixInverse1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan)
        {
            // Implement mixed-radix inverse FFT
            CooleyTukeyInverse1D(input, output, plan);
        }

        /// <summary>
        /// Batched forward 1D FFT kernel.
        /// </summary>
        public static void BatchedForward1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan,
            int batchSize)
        {
            var signalSize = input.Length / batchSize;

            // Process each signal in the batch
            for (int batch = 0; batch < batchSize; batch++)
            {
                var offset = batch * signalSize;
                var inputSlice = input.SubView(offset, signalSize);
                var outputSlice = output.SubView(offset, signalSize);
                
                CooleyTukeyForward1D(inputSlice, outputSlice, plan);
            }
        }

        /// <summary>
        /// Batched inverse 1D FFT kernel.
        /// </summary>
        public static void BatchedInverse1D(
            Index1D index,
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            FFTPlan plan,
            int batchSize)
        {
            var signalSize = input.Length / batchSize;

            for (int batch = 0; batch < batchSize; batch++)
            {
                var offset = batch * signalSize;
                var inputSlice = input.SubView(offset, signalSize);
                var outputSlice = output.SubView(offset, signalSize);
                
                CooleyTukeyInverse1D(inputSlice, outputSlice, plan);
            }
        }

        #endregion

        #region 2D FFT Kernels

        /// <summary>
        /// 2D FFT row-wise forward transform.
        /// </summary>
        public static void FFT2DRowForward(
            Index1D index,
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            FFTPlan plan)
        {
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            // Perform 1D FFT on each row
            for (int y = 0; y < height; y++)
            {
                var rowInput = input.SubView(new Index2D(0, y), new Index2D(width, 1));
                var rowOutput = output.SubView(new Index2D(0, y), new Index2D(width, 1));
                
                var input1D = rowInput.AsLinearView();
                var output1D = rowOutput.AsLinearView();
                
                CooleyTukeyForward1D(input1D, output1D, plan);
            }
        }

        /// <summary>
        /// 2D FFT column-wise forward transform.
        /// </summary>
        public static void FFT2DColumnForward(
            Index1D index,
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            FFTPlan plan)
        {
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            // Perform 1D FFT on each column
            for (int x = 0; x < width; x++)
            {
                // Extract column into temporary buffer
                var column = new Complex[height];
                for (int y = 0; y < height; y++)
                {
                    column[y] = input[new Index2D(x, y)];
                }

                // Perform FFT on column
                var columnView = new ArrayView<Complex>(column, 0, height);
                CooleyTukeyForward1D(columnView, columnView, plan);

                // Write back column
                for (int y = 0; y < height; y++)
                {
                    output[new Index2D(x, y)] = column[y];
                }
            }
        }

        /// <summary>
        /// 2D FFT row-wise inverse transform.
        /// </summary>
        public static void FFT2DRowInverse(
            Index1D index,
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            FFTPlan plan)
        {
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int y = 0; y < height; y++)
            {
                var rowInput = input.SubView(new Index2D(0, y), new Index2D(width, 1));
                var rowOutput = output.SubView(new Index2D(0, y), new Index2D(width, 1));
                
                var input1D = rowInput.AsLinearView();
                var output1D = rowOutput.AsLinearView();
                
                CooleyTukeyInverse1D(input1D, output1D, plan);
            }
        }

        /// <summary>
        /// 2D FFT column-wise inverse transform.
        /// </summary>
        public static void FFT2DColumnInverse(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            FFTPlan plan)
        {
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int x = 0; x < width; x++)
            {
                var column = new Complex[height];
                for (int y = 0; y < height; y++)
                {
                    column[y] = input[new Index2D(x, y)];
                }

                var columnView = new ArrayView<Complex>(column, 0, height);
                CooleyTukeyInverse1D(columnView, columnView, plan);

                for (int y = 0; y < height; y++)
                {
                    output[new Index2D(x, y)] = column[y];
                }
            }
        }

        #endregion

        #region 3D FFT Kernels

        /// <summary>
        /// 3D FFT along X dimension.
        /// </summary>
        public static void FFT3DXForward(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < height; y++)
                {
                    var line = new Complex[width];
                    for (int x = 0; x < width; x++)
                    {
                        line[x] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, width);
                    CooleyTukeyForward1D(lineView, lineView, plan);

                    for (int x = 0; x < width; x++)
                    {
                        output[new Index3D(x, y, z)] = line[x];
                    }
                }
            }
        }

        /// <summary>
        /// 3D FFT along Y dimension.
        /// </summary>
        public static void FFT3DYForward(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int z = 0; z < depth; z++)
            {
                for (int x = 0; x < width; x++)
                {
                    var line = new Complex[height];
                    for (int y = 0; y < height; y++)
                    {
                        line[y] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, height);
                    CooleyTukeyForward1D(lineView, lineView, plan);

                    for (int y = 0; y < height; y++)
                    {
                        output[new Index3D(x, y, z)] = line[y];
                    }
                }
            }
        }

        /// <summary>
        /// 3D FFT along Z dimension.
        /// </summary>
        public static void FFT3DZForward(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var line = new Complex[depth];
                    for (int z = 0; z < depth; z++)
                    {
                        line[z] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, depth);
                    CooleyTukeyForward1D(lineView, lineView, plan);

                    for (int z = 0; z < depth; z++)
                    {
                        output[new Index3D(x, y, z)] = line[z];
                    }
                }
            }
        }

        /// <summary>
        /// 3D FFT along X dimension (inverse).
        /// </summary>
        public static void FFT3DXInverse(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            // Similar to forward but using inverse 1D FFT
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < height; y++)
                {
                    var line = new Complex[width];
                    for (int x = 0; x < width; x++)
                    {
                        line[x] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, width);
                    CooleyTukeyInverse1D(lineView, lineView, plan);

                    for (int x = 0; x < width; x++)
                    {
                        output[new Index3D(x, y, z)] = line[x];
                    }
                }
            }
        }

        /// <summary>
        /// 3D FFT along Y dimension (inverse).
        /// </summary>
        public static void FFT3DYInverse(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int z = 0; z < depth; z++)
            {
                for (int x = 0; x < width; x++)
                {
                    var line = new Complex[height];
                    for (int y = 0; y < height; y++)
                    {
                        line[y] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, height);
                    CooleyTukeyInverse1D(lineView, lineView, plan);

                    for (int y = 0; y < height; y++)
                    {
                        output[new Index3D(x, y, z)] = line[y];
                    }
                }
            }
        }

        /// <summary>
        /// 3D FFT along Z dimension (inverse).
        /// </summary>
        public static void FFT3DZInverse(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            FFTPlan plan)
        {
            var depth = input.IntExtent.Z;
            var height = input.IntExtent.Y;
            var width = input.IntExtent.X;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var line = new Complex[depth];
                    for (int z = 0; z < depth; z++)
                    {
                        line[z] = input[new Index3D(x, y, z)];
                    }

                    var lineView = new ArrayView<Complex>(line, 0, depth);
                    CooleyTukeyInverse1D(lineView, lineView, plan);

                    for (int z = 0; z < depth; z++)
                    {
                        output[new Index3D(x, y, z)] = line[z];
                    }
                }
            }
        }

        #endregion

        #region Real-to-Complex FFT Kernels

        /// <summary>
        /// Real-to-complex forward FFT kernel.
        /// </summary>
        public static void RealToComplexForward<T>(
            ArrayView<T> input,
            ArrayView<Complex> output,
            FFTPlan plan)
            where T : unmanaged, INumber<T>
        {
            var n = input.Length;
            var halfN = n / 2 + 1;

            // Convert real input to complex
            var complexInput = new Complex[n];
            for (int i = 0; i < n; i++)
            {
                complexInput[i] = new Complex(Convert.ToDouble(input[i]), 0.0);
            }

            // Perform complex FFT
            var complexView = new ArrayView<Complex>(complexInput, 0, n);
            CooleyTukeyForward1D(complexView, complexView, plan);

            // Extract positive frequencies only (Hermitian symmetry)
            for (int i = 0; i < halfN; i++)
            {
                output[i] = complexInput[i];
            }
        }

        #endregion

        #region Utility Kernels

        /// <summary>
        /// Normalizes FFT output by a scalar factor.
        /// </summary>
        public static void Normalize(
            Index1D index,
            ArrayView<Complex> data,
            float scale)
        {
            if (index >= data.Length) return;
            data[index] *= scale;
        }

        /// <summary>
        /// Normalizes 2D FFT output.
        /// </summary>
        public static void Normalize2D(
            Index2D index,
            ArrayView2D<Complex, Stride2D.DenseX> data,
            float scale)
        {
            if (index.X >= data.IntExtent.X || index.Y >= data.IntExtent.Y) return;
            data[index] *= scale;
        }

        /// <summary>
        /// Performs bit-reversal permutation.
        /// </summary>
        private static void BitReversalPermutation(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            int n)
        {
            int logN = (int)Math.Log2(n);

            for (int i = 0; i < n; i++)
            {
                int rev = 0;
                int temp = i;
                for (int j = 0; j < logN; j++)
                {
                    rev = (rev << 1) | (temp & 1);
                    temp >>= 1;
                }
                output[rev] = input[i];
            }
        }

        #endregion
    }
}
