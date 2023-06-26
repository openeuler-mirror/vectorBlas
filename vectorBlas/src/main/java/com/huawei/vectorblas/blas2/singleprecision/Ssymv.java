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

package com.huawei.vectorblas.blas2.singleprecision;

import static com.huawei.vectorblas.utils.ArrayUtil.loopBound;

import com.huawei.vectorblas.utils.BlasUtils;
import com.huawei.vectorblas.utils.Lsame;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Ssymv {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void ssymv(String uplo, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        int incx, float beta, float[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("SSYMV", 1, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        BlasUtils.checkParameter("SSYMV", 2, n >= 0);
        BlasUtils.checkParameter("SSYMV", 5, lda >= Math.max(1, n));
        BlasUtils.checkParameter("SSYMV", 7, incx != 0);
        BlasUtils.checkParameter("SSYMV", 10, incy != 0);

        if (n == 0 || (BlasUtils.isZero(alpha) && Float.compare(beta, 1.0f) == 0)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (n - 1) + (n - 1) * lda, a.length);

        boolean uploFlag = Lsame.lsame(uplo, "U");
        int xStartIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        int yStartIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        if (Float.compare(beta, 1.0f) != 0) {
            SblasLevel2.sMulBeta(n, beta, y, yOffset, incy);
        }
        if (BlasUtils.isZero(alpha)) {
            return;
        }
        if (uploFlag) {
            if (incx == 1 && incy == 1) {
                vecSsymvU(n, x, xOffset, alpha, y, yOffset, a, aOffset, lda);
            } else {
                norSsymvU(n, x, xOffset, incx, alpha, y, yOffset, incy, a, aOffset, lda, xStartIndex, yStartIndex);
            }
        } else if (incx == 1 && incy == 1) {
            vecSsymvL(n, x, xOffset, alpha, y, yOffset, a, aOffset, lda);
        } else {
            norSsymvL(n, x, xOffset, incx, alpha, y, yOffset, incy, a, aOffset, lda, xStartIndex, yStartIndex);
        }
    }

    private static void vecSsymvU(int n, float[] x, int xOffset, float alpha, float[] y, int yOffset, float[] a,
        int aOffset, int lda) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            float alphaMulX0 = alpha * x[col + xOffset];
            float alphaMulX1 = alpha * x[(col + 1) + xOffset];
            float alphaMulX2 = alpha * x[(col + 2) + xOffset];
            float alphaMulX3 = alpha * x[(col + 3) + xOffset];
            FloatVector alphaXv0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            FloatVector alphaXv1 = FloatVector.broadcast(SSPECIES, alphaMulX1);
            FloatVector alphaXv2 = FloatVector.broadcast(SSPECIES, alphaMulX2);
            FloatVector alphaXv3 = FloatVector.broadcast(SSPECIES, alphaMulX3);
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            FloatVector accumv1 = FloatVector.zero(SSPECIES);
            FloatVector accumv2 = FloatVector.zero(SSPECIES);
            FloatVector accumv3 = FloatVector.zero(SSPECIES);
            int row = 0;
            for (; row < col - col % SSPECIES.length(); row += SSPECIES.length()) {
                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, row + (col + 1) * lda + aOffset);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, row + (col + 2) * lda + aOffset);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, row + (col + 3) * lda + aOffset);
                FloatVector yv = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, row + xOffset);
                yv = av0.fma(alphaXv0, yv);
                yv = av1.fma(alphaXv1, yv);
                yv = av2.fma(alphaXv2, yv);
                av3.fma(alphaXv3, yv).intoArray(y, row + yOffset);
                accumv0 = av0.fma(xv, accumv0);
                accumv1 = av1.fma(xv, accumv1);
                accumv2 = av2.fma(xv, accumv2);
                accumv3 = av3.fma(xv, accumv3);
            }
            float accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            float accum1 = alpha * accumv1.reduceLanes(VectorOperators.ADD);
            float accum2 = alpha * accumv2.reduceLanes(VectorOperators.ADD);
            float accum3 = alpha * accumv3.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                float a0 = a[row + col * lda + aOffset];
                float a1 = a[row + (col + 1) * lda + aOffset];
                float a2 = a[row + (col + 2) * lda + aOffset];
                float a3 = a[row + (col + 3) * lda + aOffset];
                float x0 = x[row + xOffset];
                y[row + yOffset] += alpha * (a0 * x[col + xOffset] + a1 * x[(col + 1) + xOffset]
                    + a2 * x[(col + 2) + xOffset] + a3 * x[(col + 3) + xOffset]);
                accum0 += alpha * a0 * x0;
                accum1 += alpha * a1 * x0;
                accum2 += alpha * a2 * x0;
                accum3 += alpha * a3 * x0;
            }
            float a00 = a[row + col * lda + aOffset];
            float a01 = a[row + (col + 1) * lda + aOffset];
            float a02 = a[row + (col + 2) * lda + aOffset];
            float a03 = a[row + (col + 3) * lda + aOffset];
            float a11 = a[(row + 1) + (col + 1) * lda + aOffset];
            float a12 = a[(row + 1) + (col + 2) * lda + aOffset];
            float a13 = a[(row + 1) + (col + 3) * lda + aOffset];
            float a22 = a[(row + 2) + (col + 2) * lda + aOffset];
            float a23 = a[(row + 2) + (col + 3) * lda + aOffset];
            float a33 = a[(row + 3) + (col + 3) * lda + aOffset];
            y[col + yOffset] += a00 * alphaMulX0 + a01 * alphaMulX1 + a02 * alphaMulX2 + a03 * alphaMulX3 + accum0;
            y[(col + 1) + yOffset] += a01 * alphaMulX0 + a11 * alphaMulX1 + a12 * alphaMulX2 + a13 * alphaMulX3
                    + accum1;
            y[(col + 2) + yOffset] += a02 * alphaMulX0 + a12 * alphaMulX1 + a22 * alphaMulX2 + a23 * alphaMulX3
                    + accum2;
            y[(col + 3) + yOffset] += a03 * alphaMulX0 + a13 * alphaMulX1 + a23 * alphaMulX2 + a33 * alphaMulX3
                    + accum3;
        }
        for (; col < n; col++) {
            float alphaMulX0 = alpha * x[col + xOffset];
            FloatVector alphaXv0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            int row = 0;
            for (; row < col - col % SSPECIES.length(); row += SSPECIES.length()) {
                FloatVector av = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector yv = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, row + xOffset);
                av.fma(alphaXv0, yv).intoArray(y, row + yOffset);
                accumv0 = av.fma(xv, accumv0);
            }
            float accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                float a0 = a[row + col * lda + aOffset];
                y[row + yOffset] += a0 * alphaMulX0;
                accum0 += alpha * a0 * x[row + xOffset];
            }
            y[col + yOffset] += a[row + col * lda + aOffset] * alphaMulX0 + accum0;
        }
    }

    private static void norSsymvU(int n, float[] x, int xOffset, int incx, float alpha, float[] y, int yOffset,
        int incy, float[] a, int aOffset, int lda, int xStartIndex, int yStartIndex) {
        for (int col = 0, xj = xStartIndex, yj = yStartIndex; col < n; col++, xj += incx, yj += incy) {
            float alphaMulX = alpha * x[xj + xOffset];
            float accum = 0.0f;

            for (int row = 0, xIndx = xStartIndex, yIndx = yStartIndex; row < col; row++, xIndx += incx,
                yIndx += incy) {
                y[yIndx + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                accum += a[row + col * lda + aOffset] * x[xIndx + xOffset];
            }
            y[yj + yOffset] += alphaMulX * a[col + col * lda + aOffset] + alpha * accum;
        }
    }

    private static void vecSsymvL(int n, float[] x, int xOffset, float alpha, float[] y, int yOffset, float[] a,
        int aOffset, int lda) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            int row = col;
            float a00 = a[aOffset + row + col * lda];
            float a10 = a[aOffset + (row + 1) + col * lda];
            float a20 = a[aOffset + (row + 2) + col * lda];
            float a30 = a[aOffset + (row + 3) + col * lda];
            float a11 = a[aOffset + (row + 1) + (col + 1) * lda];
            float a21 = a[aOffset + (row + 2) + (col + 1) * lda];
            float a31 = a[aOffset + (row + 3) + (col + 1) * lda];
            float a22 = a[aOffset + (row + 2) + (col + 2) * lda];
            float a32 = a[aOffset + (row + 3) + (col + 2) * lda];
            float a33 = a[aOffset + (row + 3) + (col + 3) * lda];
            float alphaMulX0 = alpha * x[xOffset + col];
            float alphaMulX1 = alpha * x[xOffset + (col + 1)];
            float alphaMulX2 = alpha * x[xOffset + (col + 2)];
            float alphaMulX3 = alpha * x[xOffset + (col + 3)];
            float accum0 = alphaMulX0 * a00 + alphaMulX1 * a10 + alphaMulX2 * a20 + alphaMulX3 * a30;
            float accum1 = alphaMulX0 * a10 + alphaMulX1 * a11 + alphaMulX2 * a21 + alphaMulX3 * a31;
            float accum2 = alphaMulX0 * a20 + alphaMulX1 * a21 + alphaMulX2 * a22 + alphaMulX3 * a32;
            float accum3 = alphaMulX0 * a30 + alphaMulX1 * a31 + alphaMulX2 * a32 + alphaMulX3 * a33;
            FloatVector alphaMulXV0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            FloatVector alphaMulXV1 = FloatVector.broadcast(SSPECIES, alphaMulX1);
            FloatVector alphaMulXV2 = FloatVector.broadcast(SSPECIES, alphaMulX2);
            FloatVector alphaMulXV3 = FloatVector.broadcast(SSPECIES, alphaMulX3);
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            FloatVector accumv1 = FloatVector.zero(SSPECIES);
            FloatVector accumv2 = FloatVector.zero(SSPECIES);
            FloatVector accumv3 = FloatVector.zero(SSPECIES);
            row += 4;
            for (; row <= (n - n % SSPECIES.length() - SSPECIES.length()); row += SSPECIES.length()) {
                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, aOffset + row + col * lda);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 1) * lda);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 2) * lda);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 3) * lda);
                FloatVector yv = FloatVector.fromArray(SSPECIES, y, yOffset + row);
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, xOffset + row);
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
                float a0 = a[aOffset + row + col * lda];
                float a1 = a[aOffset + row + (col + 1) * lda];
                float a2 = a[aOffset + row + (col + 2) * lda];
                float a3 = a[aOffset + row + (col + 3) * lda];
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
            float alphaMulX0 = alpha * x[xOffset + col];
            y[yOffset + col] += a[aOffset + col + col * lda] * alphaMulX0;
            int row = col + 1;
            float accum0 = 0.0f;
            for (; row < n; row++) {
                float a0 = a[aOffset + row + col * lda];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += alpha * accum0;
        }
    }

    private static void norSsymvL(int n, float[] x, int xOffset, int incx, float alpha, float[] y, int yOffset,
        int incy, float[] a, int aOffset, int lda, int xStartIndex, int yStartIndex) {
        for (int col = 0, xj = xStartIndex, yj = yStartIndex; col < n; col++, xj += incx, yj += incy) {
            float alphaMulX = alpha * x[xj + xOffset];
            y[yj + yOffset] += alphaMulX * a[col + col * lda + aOffset];
            float accum = 0.0f;

            for (int row = col + 1, xIndx = xj + incx, yIndx = yj + incy; row < n; row++, xIndx += incx,
                yIndx += incy) {
                y[yIndx + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                accum += a[row + col * lda + aOffset] * x[xIndx + xOffset];
            }
            y[yj + yOffset] += alpha * accum;
        }
    }
}
