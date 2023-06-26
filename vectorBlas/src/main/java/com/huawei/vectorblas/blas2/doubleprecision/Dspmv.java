/*
 * Copyright (C) 2023. Huawei Technologies Co., Ltd.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.huawei.vectorblas.blas2.doubleprecision;

import static com.huawei.vectorblas.utils.ArrayUtil.loopBound;

import com.huawei.vectorblas.utils.BlasUtils;
import com.huawei.vectorblas.utils.Lsame;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Dspmv {
    public static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void dspmv(String uplo, int n, double alpha, double[] a, int aOffset, double[] x, int xOffset,
        int incx, double beta, double[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("DSPMV", 1, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        BlasUtils.checkParameter("DSPMV", 2, n >= 0);
        BlasUtils.checkParameter("DSPMV", 6, incx != 0);
        BlasUtils.checkParameter("DSPMV", 9, incy != 0);

        if (n == 0 || (BlasUtils.isZero(alpha) && Double.compare(beta, 1.0) == 0)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (1 + n) * n / 2 - 1, a.length);

        boolean uploFlag = Lsame.lsame(uplo, "U");
        int xStartIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        int yStartIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        if (Double.compare(beta, 1.0d) != 0) {
            DblasLevel2.dMulBeta(n, beta, y, yOffset, incy);
        }
        if (BlasUtils.isZero(alpha)) {
            return;
        }
        if (uploFlag) {
            if (incx == 1 && incy == 1) {
                vecDspmvU(n, alpha, a, aOffset, x, xOffset, y, yOffset);
            } else {
                norDspmvU(n, alpha, a, aOffset, x, xOffset, incx, y, yOffset, incy, xStartIndex, yStartIndex);
            }
        } else {
            if (incx == 1 && incy == 1) {
                vecDspmvL(n, alpha, a, aOffset, x, xOffset, y, yOffset);
            } else {
                norDspmvL(n, alpha, a, aOffset, x, xOffset, incx, y, yOffset, incy, xStartIndex, yStartIndex);
            }
        }
    }

    private static void vecDspmvU(int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, double[] y,
        int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            double alphaMulX0 = alpha * x[xOffset + col];
            double alphaMulX1 = alpha * x[xOffset + (col + 1)];
            double alphaMulX2 = alpha * x[xOffset + (col + 2)];
            double alphaMulX3 = alpha * x[xOffset + (col + 3)];
            DoubleVector alphaMulXV0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            DoubleVector alphaMulXV1 = DoubleVector.broadcast(DSPECIES, alphaMulX1);
            DoubleVector alphaMulXV2 = DoubleVector.broadcast(DSPECIES, alphaMulX2);
            DoubleVector alphaMulXV3 = DoubleVector.broadcast(DSPECIES, alphaMulX3);
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv1 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv2 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv3 = DoubleVector.zero(DSPECIES);
            int row = 0;
            for (; row < col - col % DSPECIES.length(); row += DSPECIES.length()) {
                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + col * (col + 1) / 2);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 1) * ((col + 1) + 1) / 2);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 2) * ((col + 2) + 1) / 2);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 3) * ((col + 3) + 1) / 2);
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, yOffset + row);
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, xOffset + row);
                yv = alphaMulXV0.fma(av0, yv);
                yv = alphaMulXV1.fma(av1, yv);
                yv = alphaMulXV2.fma(av2, yv);
                alphaMulXV3.fma(av3, yv).intoArray(y, yOffset + row);
                accumv0 = xv.fma(av0, accumv0);
                accumv1 = xv.fma(av1, accumv1);
                accumv2 = xv.fma(av2, accumv2);
                accumv3 = xv.fma(av3, accumv3);
            }
            double accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            double accum1 = alpha * accumv1.reduceLanes(VectorOperators.ADD);
            double accum2 = alpha * accumv2.reduceLanes(VectorOperators.ADD);
            double accum3 = alpha * accumv3.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                double a0 = a[aOffset + row + col * (col + 1) / 2];
                double a1 = a[aOffset + row + (col + 1) * ((col + 1) + 1) / 2];
                double a2 = a[aOffset + row + (col + 2) * ((col + 2) + 1) / 2];
                double a3 = a[aOffset + row + (col + 3) * ((col + 3) + 1) / 2];
                double x0 = x[row + xOffset];
                y[row + yOffset] += alpha * (a0 * x[col + xOffset] + a1 * x[(col + 1) + xOffset]
                    + a2 * x[(col + 2) + xOffset] + a3 * x[(col + 3) + xOffset]);
                accum0 += alpha * a0 * x0;
                accum1 += alpha * a1 * x0;
                accum2 += alpha * a2 * x0;
                accum3 += alpha * a3 * x0;
            }
            double a00 = a[aOffset + row + col * (col + 1) / 2];
            double a01 = a[aOffset + row + (col + 1) * ((col + 1) + 1) / 2];
            double a02 = a[aOffset + row + (col + 2) * ((col + 2) + 1) / 2];
            double a03 = a[aOffset + row + (col + 3) * ((col + 3) + 1) / 2];
            double a11 = a[aOffset + (row + 1) + (col + 1) * ((col + 1) + 1) / 2];
            double a12 = a[aOffset + (row + 1) + (col + 2) * ((col + 2) + 1) / 2];
            double a13 = a[aOffset + (row + 1) + (col + 3) * ((col + 3) + 1) / 2];
            double a22 = a[aOffset + (row + 2) + (col + 2) * ((col + 2) + 1) / 2];
            double a23 = a[aOffset + (row + 2) + (col + 3) * ((col + 3) + 1) / 2];
            double a33 = a[aOffset + (row + 3) + (col + 3) * ((col + 3) + 1) / 2];
            y[yOffset + col] += alphaMulX0 * a00 + alphaMulX1 * a01 + alphaMulX2 * a02 + alphaMulX3 * a03 + accum0;
            y[yOffset + (col + 1)] += alphaMulX0 * a01 + alphaMulX1 * a11 + alphaMulX2 * a12 + alphaMulX3 * a13
                + accum1;
            y[yOffset + (col + 2)] += alphaMulX0 * a02 + alphaMulX1 * a12 + alphaMulX2 * a22 + alphaMulX3 * a23
                + accum2;
            y[yOffset + (col + 3)] += alphaMulX0 * a03 + alphaMulX1 * a13 + alphaMulX2 * a23 + alphaMulX3 * a33
                + accum3;
        }
        for (; col < n; col += 1) {
            double alphaMulX0 = alpha * x[xOffset + col];
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            DoubleVector alphaMulXV0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            int row = 0;
            for (; row < col - col % DSPECIES.length(); row += DSPECIES.length()) {
                DoubleVector av = DoubleVector.fromArray(DSPECIES, a, aOffset + row + col * (col + 1) / 2);
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, yOffset + row);
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, xOffset + row);
                av.fma(alphaMulXV0, yv).intoArray(y, yOffset + row);
                accumv0 = av.fma(xv, accumv0);
            }
            double accum0 = accumv0.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                double a0 = a[aOffset + row + col * (col + 1) / 2];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += a[aOffset + row + col * (col + 1) / 2] * alphaMulX0 + alpha * accum0;
        }
    }

    private static void norDspmvU(int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, int incx,
        double[] y, int yOffset, int incy, int xStartIndex, int yStartIndex) {
        int aIndx = 1;
        for (int col = 0, xIndx = xStartIndex, yIndx = yStartIndex; col < n; col++, xIndx += incx, yIndx += incy) {
            double alphaMulX = alpha * x[xIndx + xOffset];
            double accum = 0.0d;

            for (int row = aIndx, xi = xStartIndex, yi = yStartIndex; row < aIndx + col; row++, xi += incx,
                yi += incy) {
                y[yi + yOffset] += alphaMulX * a[row - 1 + aOffset];
                accum += a[row - 1 + aOffset] * x[xi + xOffset];
            }

            y[yIndx + yOffset] = y[yIndx + yOffset] + alphaMulX * a[aIndx + col - 1 + aOffset] + alpha * accum;
            aIndx += col + 1;
        }
    }

    private static void vecDspmvL(int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, double[] y,
        int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            int row = col;
            double alphaMulX0 = alpha * x[xOffset + col];
            double alphaMulX1 = alpha * x[xOffset + (col + 1)];
            double alphaMulX2 = alpha * x[xOffset + (col + 2)];
            double alphaMulX3 = alpha * x[xOffset + (col + 3)];
            DoubleVector alphaMulXV0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            DoubleVector alphaMulXV1 = DoubleVector.broadcast(DSPECIES, alphaMulX1);
            DoubleVector alphaMulXV2 = DoubleVector.broadcast(DSPECIES, alphaMulX2);
            DoubleVector alphaMulXV3 = DoubleVector.broadcast(DSPECIES, alphaMulX3);
            double a00 = a[aOffset + row - col * (col + 1) / 2 + n * col];
            double a10 = a[aOffset + (row + 1) - col * (col + 1) / 2 + n * col];
            double a20 = a[aOffset + (row + 2) - col * (col + 1) / 2 + n * col];
            double a30 = a[aOffset + (row + 3) - col * (col + 1) / 2 + n * col];
            double a11 = a[aOffset + (row + 1) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            double a21 = a[aOffset + (row + 2) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            double a31 = a[aOffset + (row + 3) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            double a22 = a[aOffset + (row + 2) - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
            double a32 = a[aOffset + (row + 3) - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
            double a33 = a[aOffset + (row + 3) - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3)];
            double accum0 = alphaMulX0 * a00 + alphaMulX1 * a10 + alphaMulX2 * a20 + alphaMulX3 * a30;
            double accum1 = alphaMulX0 * a10 + alphaMulX1 * a11 + alphaMulX2 * a21 + alphaMulX3 * a31;
            double accum2 = alphaMulX0 * a20 + alphaMulX1 * a21 + alphaMulX2 * a22 + alphaMulX3 * a32;
            double accum3 = alphaMulX0 * a30 + alphaMulX1 * a31 + alphaMulX2 * a32 + alphaMulX3 * a33;
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv1 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv2 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv3 = DoubleVector.zero(DSPECIES);
            row += 4;
            for (; row <= (n - n % DSPECIES.length() - DSPECIES.length()); row += DSPECIES.length()) {
                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, aOffset + row - col * (col + 1) / 2 + n * col);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a,
                        aOffset + row - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1));
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a,
                        aOffset + row - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2));
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a,
                        aOffset + row - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3));
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, yOffset + row);
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, xOffset + row);
                yv = alphaMulXV0.fma(av0, yv);
                yv = alphaMulXV1.fma(av1, yv);
                yv = alphaMulXV2.fma(av2, yv);
                alphaMulXV3.fma(av3, yv).intoArray(y, yOffset + row);
                accumv0 = xv.fma(av0, accumv0);
                accumv1 = xv.fma(av1, accumv1);
                accumv2 = xv.fma(av2, accumv2);
                accumv3 = xv.fma(av3, accumv3);
            }
            accum0 += alpha * accumv0.reduceLanes(VectorOperators.ADD);
            accum1 += alpha * accumv1.reduceLanes(VectorOperators.ADD);
            accum2 += alpha * accumv2.reduceLanes(VectorOperators.ADD);
            accum3 += alpha * accumv3.reduceLanes(VectorOperators.ADD);
            for (; row < n; row += 1) {
                double a0 = a[aOffset + row - col * (col + 1) / 2 + n * col];
                double a1 = a[aOffset + row - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
                double a2 = a[aOffset + row - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
                double a3 = a[aOffset + row - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3)];
                y[yOffset + row] += alphaMulX0 * a0 + alphaMulX1 * a1 + alphaMulX2 * a2 + alphaMulX3 * a3;
                accum0 += alpha * x[xOffset + row] * a0;
                accum1 += alpha * x[xOffset + row] * a1;
                accum2 += alpha * x[xOffset + row] * a2;
                accum3 += alpha * x[xOffset + row] * a3;
            }
            y[yOffset + col] += accum0;
            y[yOffset + (col + 1)] += accum1;
            y[yOffset + (col + 2)] += accum2;
            y[yOffset + (col + 3)] += accum3;
        }
        for (; col < n; col += 1) {
            double alphaMulX0 = alpha * x[xOffset + col];
            y[yOffset + col] += a[aOffset + col - col * (col + 1) / 2 + n * col] * alphaMulX0;
            int row = col + 1;
            double accum0 = 0.0d;
            for (; row < n; row++) {
                double a0 = a[aOffset + row - col * (col + 1) / 2 + n * col];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += alpha * accum0;
        }
    }

    private static void norDspmvL(int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, int incx,
        double[] y, int yOffset, int incy, int xStartIndex, int yStartIndex) {
        int aIndx = 1;
        for (int col = 0, xIndx = xStartIndex, yIndx = yStartIndex; col < n; col++, xIndx += incx, yIndx += incy) {
            double alphaMulX = alpha * x[xIndx + xOffset];
            double accum = 0.0d;
            y[yIndx + yOffset] += alphaMulX * a[aIndx - 1 + aOffset];

            for (int row = aIndx + 1, xi = xIndx + incx, yi = yIndx + incy; row < aIndx + n - col; row++, xi += incx,
                yi += incy) {
                y[yi + yOffset] += alphaMulX * a[row - 1 + aOffset];
                accum += a[row - 1 + aOffset] * x[xi + xOffset];
            }
            y[yIndx + yOffset] += alpha * accum;
            aIndx += n - col;
        }
    }
}
