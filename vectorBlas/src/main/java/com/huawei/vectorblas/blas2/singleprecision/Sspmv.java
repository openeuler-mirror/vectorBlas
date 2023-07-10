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

public class Sspmv {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void sspmv(String uplo, int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, int incx,
        float beta, float[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("SSPMV", 1, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        BlasUtils.checkParameter("SSPMV", 2, n >= 0);
        BlasUtils.checkParameter("SSPMV", 6, incx != 0);
        BlasUtils.checkParameter("SSPMV", 9, incy != 0);

        if (n == 0 || (BlasUtils.isZero(alpha) && Float.compare(beta, 1.0f) == 0)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (1 + n) * n / 2 - 1, a.length);

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
                vecSspmvU(n, alpha, a, aOffset, x, xOffset, y, yOffset);
            } else {
                norSspmvU(n, alpha, a, aOffset, x, xOffset, incx, y, yOffset, incy, xStartIndex, yStartIndex);
            }
        } else {
            if (incx == 1 && incy == 1) {
                vecSspmvL(n, alpha, a, aOffset, x, xOffset, y, yOffset);
            } else {
                norSspmvL(n, alpha, a, aOffset, x, xOffset, incx, y, yOffset, incy, xStartIndex, yStartIndex);
            }
        }
    }

    private static void vecSspmvU(int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, float[] y,
        int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            float alphaMulX0 = alpha * x[xOffset + col];
            float alphaMulX1 = alpha * x[xOffset + (col + 1)];
            float alphaMulX2 = alpha * x[xOffset + (col + 2)];
            float alphaMulX3 = alpha * x[xOffset + (col + 3)];
            FloatVector alphaMulXV0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            FloatVector alphaMulXV1 = FloatVector.broadcast(SSPECIES, alphaMulX1);
            FloatVector alphaMulXV2 = FloatVector.broadcast(SSPECIES, alphaMulX2);
            FloatVector alphaMulXV3 = FloatVector.broadcast(SSPECIES, alphaMulX3);
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            FloatVector accumv1 = FloatVector.zero(SSPECIES);
            FloatVector accumv2 = FloatVector.zero(SSPECIES);
            FloatVector accumv3 = FloatVector.zero(SSPECIES);
            int row = 0;
            for (; row < col - col % SSPECIES.length(); row += SSPECIES.length()) {
                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, aOffset + row + col * (col + 1) / 2);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 1) * ((col + 1) + 1) / 2);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 2) * ((col + 2) + 1) / 2);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, aOffset + row + (col + 3) * ((col + 3) + 1) / 2);
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
            float accum0 = alpha * accumv0.reduceLanes(VectorOperators.ADD);
            float accum1 = alpha * accumv1.reduceLanes(VectorOperators.ADD);
            float accum2 = alpha * accumv2.reduceLanes(VectorOperators.ADD);
            float accum3 = alpha * accumv3.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                float a0 = a[aOffset + row + col * (col + 1) / 2];
                float a1 = a[aOffset + row + (col + 1) * ((col + 1) + 1) / 2];
                float a2 = a[aOffset + row + (col + 2) * ((col + 2) + 1) / 2];
                float a3 = a[aOffset + row + (col + 3) * ((col + 3) + 1) / 2];
                float x0 = x[row + xOffset];
                y[row + yOffset] += alpha * (a0 * x[col + xOffset] + a1 * x[(col + 1) + xOffset]
                    + a2 * x[(col + 2) + xOffset] + a3 * x[(col + 3) + xOffset]);
                accum0 += alpha * a0 * x0;
                accum1 += alpha * a1 * x0;
                accum2 += alpha * a2 * x0;
                accum3 += alpha * a3 * x0;
            }
            float a00 = a[aOffset + row + col * (col + 1) / 2];
            float a01 = a[aOffset + row + (col + 1) * ((col + 1) + 1) / 2];
            float a02 = a[aOffset + row + (col + 2) * ((col + 2) + 1) / 2];
            float a03 = a[aOffset + row + (col + 3) * ((col + 3) + 1) / 2];
            float a11 = a[aOffset + (row + 1) + (col + 1) * ((col + 1) + 1) / 2];
            float a12 = a[aOffset + (row + 1) + (col + 2) * ((col + 2) + 1) / 2];
            float a13 = a[aOffset + (row + 1) + (col + 3) * ((col + 3) + 1) / 2];
            float a22 = a[aOffset + (row + 2) + (col + 2) * ((col + 2) + 1) / 2];
            float a23 = a[aOffset + (row + 2) + (col + 3) * ((col + 3) + 1) / 2];
            float a33 = a[aOffset + (row + 3) + (col + 3) * ((col + 3) + 1) / 2];
            y[yOffset + col] += alphaMulX0 * a00 + alphaMulX1 * a01 + alphaMulX2 * a02 + alphaMulX3 * a03 + accum0;
            y[yOffset + (col + 1)] += alphaMulX0 * a01 + alphaMulX1 * a11 + alphaMulX2 * a12 + alphaMulX3 * a13
                + accum1;
            y[yOffset + (col + 2)] += alphaMulX0 * a02 + alphaMulX1 * a12 + alphaMulX2 * a22 + alphaMulX3 * a23
                + accum2;
            y[yOffset + (col + 3)] += alphaMulX0 * a03 + alphaMulX1 * a13 + alphaMulX2 * a23 + alphaMulX3 * a33
                + accum3;
        }
        for (; col < n; col += 1) {
            float alphaMulX0 = alpha * x[xOffset + col];
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            FloatVector alphaMulXV0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            int row = 0;
            for (; row < col - col % SSPECIES.length(); row += SSPECIES.length()) {
                FloatVector av = FloatVector.fromArray(SSPECIES, a, aOffset + row + col * (col + 1) / 2);
                FloatVector yv = FloatVector.fromArray(SSPECIES, y, yOffset + row);
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, xOffset + row);
                av.fma(alphaMulXV0, yv).intoArray(y, yOffset + row);
                accumv0 = av.fma(xv, accumv0);
            }
            float accum0 = accumv0.reduceLanes(VectorOperators.ADD);
            for (; row < col; row++) {
                float a0 = a[aOffset + row + col * (col + 1) / 2];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += a[aOffset + row + col * (col + 1) / 2] * alphaMulX0 + alpha * accum0;
        }
    }

    private static void norSspmvU(int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, int incx,
        float[] y, int yOffset, int incy, int xStartIndex, int yStartIndex) {
        int aIndx = 1;
        for (int col = 0, xIndx = xStartIndex, yIndx = yStartIndex; col < n; col++, xIndx += incx, yIndx += incy) {
            float alphaMulX = alpha * x[xIndx + xOffset];
            float accum = 0.0f;

            for (int row = aIndx, xi = xStartIndex, yi = yStartIndex; row < aIndx + col; row++, xi += incx,
                yi += incy) {
                y[yi + yOffset] += alphaMulX * a[row - 1 + aOffset];
                accum += a[row - 1 + aOffset] * x[xi + xOffset];
            }

            y[yIndx + yOffset] = y[yIndx + yOffset] + alphaMulX * a[aIndx + col - 1 + aOffset] + alpha * accum;
            aIndx += col + 1;
        }
    }

    private static void vecSspmvL(int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, float[] y,
        int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        for (; col < colLoopBound; col += 4) { // 4 is unroll size for column
            int row = col;
            float alphaMulX0 = alpha * x[xOffset + col];
            float alphaMulX1 = alpha * x[xOffset + (col + 1)];
            float alphaMulX2 = alpha * x[xOffset + (col + 2)];
            float alphaMulX3 = alpha * x[xOffset + (col + 3)];
            FloatVector alphaMulXV0 = FloatVector.broadcast(SSPECIES, alphaMulX0);
            FloatVector alphaMulXV1 = FloatVector.broadcast(SSPECIES, alphaMulX1);
            FloatVector alphaMulXV2 = FloatVector.broadcast(SSPECIES, alphaMulX2);
            FloatVector alphaMulXV3 = FloatVector.broadcast(SSPECIES, alphaMulX3);
            float a00 = a[aOffset + row - col * (col + 1) / 2 + n * col];
            float a10 = a[aOffset + (row + 1) - col * (col + 1) / 2 + n * col];
            float a20 = a[aOffset + (row + 2) - col * (col + 1) / 2 + n * col];
            float a30 = a[aOffset + (row + 3) - col * (col + 1) / 2 + n * col];
            float a11 = a[aOffset + (row + 1) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            float a21 = a[aOffset + (row + 2) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            float a31 = a[aOffset + (row + 3) - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
            float a22 = a[aOffset + (row + 2) - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
            float a32 = a[aOffset + (row + 3) - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
            float a33 = a[aOffset + (row + 3) - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3)];
            float accum0 = alphaMulX0 * a00 + alphaMulX1 * a10 + alphaMulX2 * a20 + alphaMulX3 * a30;
            float accum1 = alphaMulX0 * a10 + alphaMulX1 * a11 + alphaMulX2 * a21 + alphaMulX3 * a31;
            float accum2 = alphaMulX0 * a20 + alphaMulX1 * a21 + alphaMulX2 * a22 + alphaMulX3 * a32;
            float accum3 = alphaMulX0 * a30 + alphaMulX1 * a31 + alphaMulX2 * a32 + alphaMulX3 * a33;
            FloatVector accumv0 = FloatVector.zero(SSPECIES);
            FloatVector accumv1 = FloatVector.zero(SSPECIES);
            FloatVector accumv2 = FloatVector.zero(SSPECIES);
            FloatVector accumv3 = FloatVector.zero(SSPECIES);
            row += 4;
            for (; row <= (n - n % SSPECIES.length() - SSPECIES.length()); row += SSPECIES.length()) {
                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, aOffset + row - col * (col + 1) / 2 + n * col);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a,
                        aOffset + row - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1));
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a,
                        aOffset + row - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2));
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a,
                        aOffset + row - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3));
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
                float a0 = a[aOffset + row - col * (col + 1) / 2 + n * col];
                float a1 = a[aOffset + row - (col + 1) * ((col + 1) + 1) / 2 + n * (col + 1)];
                float a2 = a[aOffset + row - (col + 2) * ((col + 2) + 1) / 2 + n * (col + 2)];
                float a3 = a[aOffset + row - (col + 3) * ((col + 3) + 1) / 2 + n * (col + 3)];
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
            y[yOffset + col] += a[aOffset + col - col * (col + 1) / 2 + n * col] * alphaMulX0;
            int row = col + 1;
            float accum0 = 0.0f;
            for (; row < n; row++) {
                float a0 = a[aOffset + row - col * (col + 1) / 2 + n * col];
                y[yOffset + row] += a0 * alphaMulX0;
                accum0 += x[xOffset + row] * a0;
            }
            y[yOffset + col] += alpha * accum0;
        }
    }

    private static void norSspmvL(int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, int incx,
        float[] y, int yOffset, int incy, int xStartIndex, int yStartIndex) {
        int aIndx = 1;
        for (int col = 0, xIndx = xStartIndex, yIndx = yStartIndex; col < n; col++, xIndx += incx, yIndx += incy) {
            float alphaMulX = alpha * x[xIndx + xOffset];
            float accum = 0.0f;
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
