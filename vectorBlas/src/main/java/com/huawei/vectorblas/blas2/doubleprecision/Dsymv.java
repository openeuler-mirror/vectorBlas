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

public class Dsymv {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void dsymv(String uplo, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("DSYMV", 1, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        BlasUtils.checkParameter("DSYMV", 2, n >= 0);
        BlasUtils.checkParameter("DSYMV", 5, lda >= Math.max(1, n));
        BlasUtils.checkParameter("DSYMV", 7, incx != 0);
        BlasUtils.checkParameter("DSYMV", 10, incy != 0);

        if (n == 0 || (BlasUtils.isZero(alpha) && Double.compare(beta, 1.0) == 0)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (n - 1) + (n - 1) * lda, a.length);

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
                vecDsymvU(n, x, xOffset, alpha, y, yOffset, a, aOffset, lda);
            } else {
                norDsymvU(n, x, xOffset, incx, alpha, y, yOffset, incy, a, aOffset, lda, xStartIndex, yStartIndex);
            }
        } else if (incx == 1 && incy == 1) {
            vecDsymvL(n, x, xOffset, alpha, y, yOffset, a, aOffset, lda);
        } else {
            norDsymvL(n, x, xOffset, incx, alpha, y, yOffset, incy, a, aOffset, lda, xStartIndex, yStartIndex);
        }
    }

    private static void vecDsymvU(int n, double[] x, int xOffset, double alpha, double[] y, int yOffset, double[] a,
        int aOffset, int lda) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            double alphaMulX0 = alpha * x[col + xOffset];
            double alphaMulX1 = alpha * x[(col + 1) + xOffset];
            double alphaMulX2 = alpha * x[(col + 2) + xOffset];
            double alphaMulX3 = alpha * x[(col + 3) + xOffset];
            DoubleVector alphaXv0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            DoubleVector alphaXv1 = DoubleVector.broadcast(DSPECIES, alphaMulX1);
            DoubleVector alphaXv2 = DoubleVector.broadcast(DSPECIES, alphaMulX2);
            DoubleVector alphaXv3 = DoubleVector.broadcast(DSPECIES, alphaMulX3);
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv1 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv2 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv3 = DoubleVector.zero(DSPECIES);
            int row = 0;
            for (; row < col - col % DSPECIES.length(); row += DSPECIES.length()) {
                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, row + (col + 1) * lda + aOffset);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a, row + (col + 2) * lda + aOffset);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a, row + (col + 3) * lda + aOffset);
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, row + xOffset);
                yv = av0.fma(alphaXv0, yv);
                yv = av1.fma(alphaXv1, yv);
                yv = av2.fma(alphaXv2, yv);
                av3.fma(alphaXv3, yv).intoArray(y, row + yOffset);
                accumv0 = av0.fma(xv, accumv0);
                accumv1 = av1.fma(xv, accumv1);
                accumv2 = av2.fma(xv, accumv2);
                accumv3 = av3.fma(xv, accumv3);
            }
            double accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            double accum1 = alpha * accumv1.reduceLanes(VectorOperators.ADD);
            double accum2 = alpha * accumv2.reduceLanes(VectorOperators.ADD);
            double accum3 = alpha * accumv3.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                double a0 = a[row + col * lda + aOffset];
                double a1 = a[row + (col + 1) * lda + aOffset];
                double a2 = a[row + (col + 2) * lda + aOffset];
                double a3 = a[row + (col + 3) * lda + aOffset];
                double x0 = x[row + xOffset];
                y[row + yOffset] += alpha * (a0 * x[col + xOffset] + a1 * x[(col + 1) + xOffset]
                    + a2 * x[(col + 2) + xOffset] + a3 * x[(col + 3) + xOffset]);
                accum0 += alpha * a0 * x0;
                accum1 += alpha * a1 * x0;
                accum2 += alpha * a2 * x0;
                accum3 += alpha * a3 * x0;
            }
            double a00 = a[row + col * lda + aOffset];
            double a01 = a[row + (col + 1) * lda + aOffset];
            double a02 = a[row + (col + 2) * lda + aOffset];
            double a03 = a[row + (col + 3) * lda + aOffset];
            double a11 = a[(row + 1) + (col + 1) * lda + aOffset];
            double a12 = a[(row + 1) + (col + 2) * lda + aOffset];
            double a13 = a[(row + 1) + (col + 3) * lda + aOffset];
            double a22 = a[(row + 2) + (col + 2) * lda + aOffset];
            double a23 = a[(row + 2) + (col + 3) * lda + aOffset];
            double a33 = a[(row + 3) + (col + 3) * lda + aOffset];
            y[col + yOffset] += a00 * alphaMulX0 + a01 * alphaMulX1 + a02 * alphaMulX2 + a03 * alphaMulX3 + accum0;
            y[(col + 1) + yOffset] += a01 * alphaMulX0 + a11 * alphaMulX1 + a12 * alphaMulX2 + a13 * alphaMulX3
                    + accum1;
            y[(col + 2) + yOffset] += a02 * alphaMulX0 + a12 * alphaMulX1 + a22 * alphaMulX2 + a23 * alphaMulX3
                    + accum2;
            y[(col + 3) + yOffset] += a03 * alphaMulX0 + a13 * alphaMulX1 + a23 * alphaMulX2 + a33 * alphaMulX3
                    + accum3;
        }
        for (; col < n; col++) {
            double alphaMulX0 = alpha * x[col + xOffset];
            DoubleVector alphaXv0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            int row = 0;
            for (; row < col - col % DSPECIES.length(); row += DSPECIES.length()) {
                DoubleVector av = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, row + xOffset);
                av.fma(alphaXv0, yv).intoArray(y, row + yOffset);
                accumv0 = av.fma(xv, accumv0);
            }
            double accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                double a0 = a[row + col * lda + aOffset];
                y[row + yOffset] += a0 * alphaMulX0;
                accum0 += alpha * a0 * x[row + xOffset];
            }
            y[col + yOffset] += a[row + col * lda + aOffset] * alphaMulX0 + accum0;
        }
    }

    private static void norDsymvU(int n, double[] x, int xOffset, int incx, double alpha, double[] y, int yOffset,
        int incy, double[] a, int aOffset, int lda, int xStartIndex, int yStartIndex) {
        for (int col = 0, xj = xStartIndex, yj = yStartIndex; col < n; col++, xj += incx, yj += incy) {
            double alphaMulX = alpha * x[xj + xOffset];
            double accum = 0.0d;

            for (int row = 0, xIndx = xStartIndex, yIndx = yStartIndex; row < col; row++, xIndx += incx,
                yIndx += incy) {
                y[yIndx + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                accum += a[row + col * lda + aOffset] * x[xIndx + xOffset];
            }
            y[yj + yOffset] += alphaMulX * a[col + col * lda + aOffset] + alpha * accum;
        }
    }

    private static void vecDsymvL(int n, double[] x, int xOffset, double alpha, double[] y, int yOffset, double[] a,
        int aOffset, int lda) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            int row = col;
            double a00 = a[aOffset + row + col * lda];
            double a10 = a[aOffset + (row + 1) + col * lda];
            double a20 = a[aOffset + (row + 2) + col * lda];
            double a30 = a[aOffset + (row + 3) + col * lda];
            double a11 = a[aOffset + (row + 1) + (col + 1) * lda];
            double a21 = a[aOffset + (row + 2) + (col + 1) * lda];
            double a31 = a[aOffset + (row + 3) + (col + 1) * lda];
            double a22 = a[aOffset + (row + 2) + (col + 2) * lda];
            double a32 = a[aOffset + (row + 3) + (col + 2) * lda];
            double a33 = a[aOffset + (row + 3) + (col + 3) * lda];
            double alphaMulX0 = alpha * x[xOffset + col];
            double alphaMulX1 = alpha * x[xOffset + (col + 1)];
            double alphaMulX2 = alpha * x[xOffset + (col + 2)];
            double alphaMulX3 = alpha * x[xOffset + (col + 3)];
            double accum0 = alphaMulX0 * a00 + alphaMulX1 * a10 + alphaMulX2 * a20 + alphaMulX3 * a30;
            double accum1 = alphaMulX0 * a10 + alphaMulX1 * a11 + alphaMulX2 * a21 + alphaMulX3 * a31;
            double accum2 = alphaMulX0 * a20 + alphaMulX1 * a21 + alphaMulX2 * a22 + alphaMulX3 * a32;
            double accum3 = alphaMulX0 * a30 + alphaMulX1 * a31 + alphaMulX2 * a32 + alphaMulX3 * a33;
            DoubleVector alphaMulXV0 = DoubleVector.broadcast(DSPECIES, alphaMulX0);
            DoubleVector alphaMulXV1 = DoubleVector.broadcast(DSPECIES, alphaMulX1);
            DoubleVector alphaMulXV2 = DoubleVector.broadcast(DSPECIES, alphaMulX2);
            DoubleVector alphaMulXV3 = DoubleVector.broadcast(DSPECIES, alphaMulX3);
            DoubleVector accumv0 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv1 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv2 = DoubleVector.zero(DSPECIES);
            DoubleVector accumv3 = DoubleVector.zero(DSPECIES);
            row += 4;
            for (; row <= (n - n % DSPECIES.length() - DSPECIES.length()); row += DSPECIES.length()) {
                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + col * lda);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 1) * lda);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 2) * lda);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a, aOffset + row + (col + 3) * lda);
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
                double a0 = a[aOffset + row + col * lda];
                double a1 = a[aOffset + row + (col + 1) * lda];
                double a2 = a[aOffset + row + (col + 2) * lda];
                double a3 = a[aOffset + row + (col + 3) * lda];
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
            y[yOffset + col] += a[aOffset + col + col * lda] * alphaMulX0;
            int row = col + 1;
            double accum0 = 0.0d;
            for (; row < n; row++) {
                double a0 = a[aOffset + row + col * lda];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += alpha * accum0;
        }
    }

    private static void norDsymvL(int n, double[] x, int xOffset, int incx, double alpha, double[] y, int yOffset,
        int incy, double[] a, int aOffset, int lda, int xStartIndex, int yStartIndex) {
        for (int col = 0, xj = xStartIndex, yj = yStartIndex; col < n; col++, xj += incx, yj += incy) {
            double alphaMulX = alpha * x[xj + xOffset];
            y[yj + yOffset] += alphaMulX * a[col + col * lda + aOffset];
            double accum = 0.0d;

            for (int row = col + 1, xIndx = xj + incx, yIndx = yj + incy; row < n; row++, xIndx += incx,
                yIndx += incy) {
                y[yIndx + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                accum += a[row + col * lda + aOffset] * x[xIndx + xOffset];
            }
            y[yj + yOffset] += alpha * accum;
        }
    }
}
