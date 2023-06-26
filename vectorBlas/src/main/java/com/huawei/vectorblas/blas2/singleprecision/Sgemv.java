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

public class Sgemv {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void sgemv(String trans, int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("SGEMV", 1, Lsame.lsame(trans, "N") || Lsame.lsame(trans, "T"));
        BlasUtils.checkParameter("SGEMV", 2, m >= 0);
        BlasUtils.checkParameter("SGEMV", 3, n >= 0);
        BlasUtils.checkParameter("SGEMV", 6, lda >= Math.max(1, m));
        BlasUtils.checkParameter("SGEMV", 8, incx != 0);
        BlasUtils.checkParameter("SGEMV", 11, incy != 0);
        if (m == 0 || n == 0 || (BlasUtils.isZero(alpha) && Float.compare(beta, 1.0f) == 0)) {
            return;
        }
        boolean transFlag = Lsame.lsame(trans, "N");
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * ((transFlag ? n : m) - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * ((transFlag ? m : n) - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (n - 1) * lda + m - 1, a.length);

        if (Float.compare(beta, 1.0f) != 0) {
            SblasLevel2.sMulBeta(transFlag ? m : n, beta, y, yOffset, incy);
        }
        if (BlasUtils.isZero(alpha)) {
            return;
        }
        if (transFlag) {
            if (incy == 1) {
                if (incx == 1) {
                    vecSgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset);
                } else {
                    vecSgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset);
                }
            } else {
                norSgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset, incy);
            }
        } else {
            if (incx == 1) {
                if (incy == 1) {
                    vecSgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset);
                } else {
                    vecSgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset, incy);
                }
            } else {
                norSgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset, incy);
            }
        }
    }

    private static void vecSgemvN(int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, float[] y, int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        int rowUnrollLoopBound = loopBound(m, SSPECIES.length() * 4);
        int rowLoopBound = loopBound(m, SSPECIES.length());
        for (; col < colLoopBound; col += 4) {
            FloatVector xv0 = FloatVector.broadcast(SSPECIES, alpha * x[col + xOffset]);
            FloatVector xv1 = FloatVector.broadcast(SSPECIES, alpha * x[col + 1 + xOffset]);
            FloatVector xv2 = FloatVector.broadcast(SSPECIES, alpha * x[col + 2 + xOffset]);
            FloatVector xv3 = FloatVector.broadcast(SSPECIES, alpha * x[col + 3 + xOffset]);
            int row = 0;
            for (; row < rowUnrollLoopBound; row += SSPECIES.length() * 4) {
                FloatVector yv0 = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                FloatVector yv1 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() + yOffset);
                FloatVector yv2 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() * 2 + yOffset);
                FloatVector yv3 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() * 3 + yOffset);

                FloatVector av00 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector av10 = FloatVector.fromArray(
                        SSPECIES, a, row + SSPECIES.length() + col * lda + aOffset);
                FloatVector av20 = FloatVector.fromArray(
                        SSPECIES, a, (row + SSPECIES.length() * 2) + col * lda + aOffset);
                FloatVector av30 = FloatVector.fromArray(
                        SSPECIES, a, (row + SSPECIES.length() * 3) + col * lda + aOffset);

                FloatVector av01 = FloatVector.fromArray(SSPECIES, a, row + (col + 1) * lda + aOffset);
                FloatVector av11 = FloatVector.fromArray(
                        SSPECIES, a, row + SSPECIES.length() + (col + 1) * lda + aOffset);
                FloatVector av21 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 2) + (col + 1) * lda + aOffset);
                FloatVector av31 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 3) + (col + 1) * lda + aOffset);

                FloatVector av02 = FloatVector.fromArray(SSPECIES, a, row + (col + 2) * lda + aOffset);
                FloatVector av12 = FloatVector.fromArray(
                        SSPECIES, a, row + SSPECIES.length() + (col + 2) * lda + aOffset);
                FloatVector av22 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 2) + (col + 2) * lda + aOffset);
                FloatVector av32 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 3) + (col + 2) * lda + aOffset);

                FloatVector av03 = FloatVector.fromArray(SSPECIES, a, row + (col + 3) * lda + aOffset);
                FloatVector av13 = FloatVector.fromArray(
                        SSPECIES, a, row + SSPECIES.length() + (col + 3) * lda + aOffset);
                FloatVector av23 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 2) + (col + 3) * lda + aOffset);
                FloatVector av33 = FloatVector.fromArray(
                    SSPECIES, a, (row + SSPECIES.length() * 3) + (col + 3) * lda + aOffset);

                av00.fma(xv0, av01.fma(xv1, av02.fma(xv2, av03.fma(xv3, yv0)))).intoArray(y, row + yOffset);
                av10.fma(xv0, av11.fma(xv1, av12.fma(xv2, av13.fma(xv3, yv1))))
                    .intoArray(y, row + SSPECIES.length() + yOffset);
                av20.fma(xv0, av21.fma(xv1, av22.fma(xv2, av23.fma(xv3, yv2))))
                    .intoArray(y, row + SSPECIES.length() * 2 + yOffset);
                av30.fma(xv0, av31.fma(xv1, av32.fma(xv2, av33.fma(xv3, yv3))))
                    .intoArray(y, row + SSPECIES.length() * 3 + yOffset);
            }
            for (; row < rowLoopBound; row += SSPECIES.length()) {
                FloatVector yv = FloatVector.fromArray(SSPECIES, y, row + yOffset);

                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, row + (col + 1) * lda + aOffset);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, row + (col + 2) * lda + aOffset);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, row + (col + 3) * lda + aOffset);

                av0.fma(xv0, av1.fma(xv1, av2.fma(xv2, av3.fma(xv3, yv)))).intoArray(y, row + yOffset);
            }
            float x0 = alpha * x[col + xOffset];
            float x1 = alpha * x[col + 1 + xOffset];
            float x2 = alpha * x[col + 2 + xOffset];
            float x3 = alpha * x[col + 3 + xOffset];
            for (; row < m; row++) {
                y[row + yOffset] += x0 * a[row + col * lda + aOffset]
                        + x1 * a[row + (col + 1) * lda + aOffset]
                        + x2 * a[row + (col + 2) * lda + aOffset]
                        + x3 * a[row + (col + 3) * lda + aOffset];
            }
        }
        for (; col < n; col++) {
            if (!BlasUtils.isZero(x[col + xOffset])) {
                FloatVector bv = FloatVector.broadcast(SSPECIES, alpha * x[col + xOffset]);
                int row = 0;
                for (; row < rowUnrollLoopBound; row += SSPECIES.length() * 4) {
                    FloatVector yv0 = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                    FloatVector yv1 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() + yOffset);
                    FloatVector yv2 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() * 2 + yOffset);
                    FloatVector yv3 = FloatVector.fromArray(SSPECIES, y, row + SSPECIES.length() * 3 + yOffset);

                    FloatVector av0 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                    FloatVector av1 = FloatVector.fromArray(
                            SSPECIES, a, row + SSPECIES.length() + col * lda + aOffset);
                    FloatVector av2 = FloatVector.fromArray(
                        SSPECIES, a, (row + SSPECIES.length() * 2) + col * lda + aOffset);
                    FloatVector av3 = FloatVector.fromArray(
                        SSPECIES, a, (row + SSPECIES.length() * 3) + col * lda + aOffset);

                    av0.fma(bv, yv0).intoArray(y, row + yOffset);
                    av1.fma(bv, yv1).intoArray(y, row + SSPECIES.length() + yOffset);
                    av2.fma(bv, yv2).intoArray(y, row + SSPECIES.length() * 2 + yOffset);
                    av3.fma(bv, yv3).intoArray(y, row + SSPECIES.length() * 3 + yOffset);
                }
                for (; row < rowLoopBound; row += SSPECIES.length()) {
                    FloatVector yv = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                    FloatVector av = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                    bv.fma(av, yv).intoArray(y, row + yOffset);
                }
                float alphaX = alpha * x[col + xOffset];
                for (; row < m; row++) {
                    y[row + yOffset] += alphaX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void vecSgemvN(int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx, float[] y, int yOffset) {
        int xIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        int rowLoopBound = SSPECIES.loopBound(m);
        for (int col = 0; col < n; col++, xIndex += incx) {
            if (!BlasUtils.isZero(x[xIndex + xOffset])) {
                float alphaMulX = alpha * x[xIndex + xOffset];
                FloatVector alphaMulXv = FloatVector.broadcast(SSPECIES, alphaMulX);
                int row = 0;
                for (; row < rowLoopBound; row += SSPECIES.length()) {
                    FloatVector av = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                    FloatVector cv = FloatVector.fromArray(SSPECIES, y, row + yOffset);
                    av.fma(alphaMulXv, cv).intoArray(y, row + yOffset);
                }
                for (; row < m; row++) {
                    y[row + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void norSgemvN(int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        int xIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        for (int col = 0; col < n; col++, xIndex += incx) {
            if (!BlasUtils.isZero(x[xIndex + xOffset])) {
                float alphaMulX = alpha * x[xIndex + xOffset];
                int yIndex = incy > 0 ? 0 : (m - 1) * (-incy);
                for (int row = 0; row < m; row++, yIndex += incy) {
                    y[yIndex + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void vecSgemvT(int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        float[] y, int yOffset, int incy) {
        int yIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        int colLoopBound = SSPECIES.loopBound(m);
        for (int row = 0; row < n; row++, yIndex += incy) {
            FloatVector cv = FloatVector.zero(SSPECIES);
            int col = 0;
            for (; col < colLoopBound; col += SSPECIES.length()) {
                FloatVector av = FloatVector.fromArray(SSPECIES, a, col + row * lda + aOffset);
                FloatVector bv = FloatVector.fromArray(SSPECIES, x, col + xOffset);
                cv = av.fma(bv, cv);
            }
            float accum = cv.reduceLanes(VectorOperators.ADD);
            for (; col < m; col++) {
                accum += a[col + row * lda + aOffset] * x[col + xOffset];
            }
            y[yIndex + yOffset] += alpha * accum;
        }
    }

    private static void vecSgemvT(int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, float[] y, int yOffset) {
        int row = 0;
        int rowLoopBound = loopBound(n, 4);
        int colUnrollLoopBound = loopBound(m, SSPECIES.length() * 4);
        int colLoopBound = loopBound(m, SSPECIES.length());
        for (; row < rowLoopBound; row += 4) {
            FloatVector yv0 = FloatVector.zero(SSPECIES);
            FloatVector yv1 = FloatVector.zero(SSPECIES);
            FloatVector yv2 = FloatVector.zero(SSPECIES);
            FloatVector yv3 = FloatVector.zero(SSPECIES);
            int col = 0;
            for (; col < colUnrollLoopBound; col += SSPECIES.length() * 4) {
                FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, col + xOffset);
                FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, col + SSPECIES.length() + xOffset);
                FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, col + (SSPECIES.length() * 2) + xOffset);
                FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, col + (SSPECIES.length() * 3) + xOffset);

                FloatVector av00 = FloatVector.fromArray(SSPECIES, a, col + row * lda + aOffset);
                FloatVector av10 = FloatVector.fromArray(
                        SSPECIES, a, col + SSPECIES.length() + row * lda + aOffset);
                FloatVector av20 = FloatVector.fromArray(
                        SSPECIES, a, col + (SSPECIES.length() * 2) + row * lda + aOffset);
                FloatVector av30 = FloatVector.fromArray(
                        SSPECIES, a, col + (SSPECIES.length() * 3) + row * lda + aOffset);
                yv0 = av00.fma(xv0, av10.fma(xv1, av20.fma(xv2, av30.fma(xv3, yv0))));

                FloatVector av01 = FloatVector.fromArray(SSPECIES, a, col + (row + 1) * lda + aOffset);
                FloatVector av11 = FloatVector.fromArray(
                        SSPECIES, a, col + SSPECIES.length() + (row + 1) * lda + aOffset);
                FloatVector av21 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 2) + (row + 1) * lda + aOffset);
                FloatVector av31 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 3) + (row + 1) * lda + aOffset);
                yv1 = av01.fma(xv0, av11.fma(xv1, av21.fma(xv2, av31.fma(xv3, yv1))));

                FloatVector av02 = FloatVector.fromArray(SSPECIES, a, col + (row + 2) * lda + aOffset);
                FloatVector av12 = FloatVector.fromArray(
                        SSPECIES, a, col + SSPECIES.length() + (row + 2) * lda + aOffset);
                FloatVector av22 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 2) + (row + 2) * lda + aOffset);
                FloatVector av32 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 3) + (row + 2) * lda + aOffset);
                yv2 = av02.fma(xv0, av12.fma(xv1, av22.fma(xv2, av32.fma(xv3, yv2))));

                FloatVector av03 = FloatVector.fromArray(SSPECIES, a, col + (row + 3) * lda + aOffset);
                FloatVector av13 = FloatVector.fromArray(
                        SSPECIES, a, col + SSPECIES.length() + (row + 3) * lda + aOffset);
                FloatVector av23 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 2) + (row + 3) * lda + aOffset);
                FloatVector av33 = FloatVector.fromArray(
                    SSPECIES, a, col + (SSPECIES.length() * 3) + (row + 3) * lda + aOffset);
                yv3 = av03.fma(xv0, av13.fma(xv1, av23.fma(xv2, av33.fma(xv3, yv3))));
            }
            for (; col < colLoopBound; col += SSPECIES.length()) {
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, col + xOffset);

                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, col + row * lda + aOffset);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, col + (row + 1) * lda + aOffset);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, col + (row + 2) * lda + aOffset);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, col + (row + 3) * lda + aOffset);

                yv0 = av0.fma(xv, yv0);
                yv1 = av1.fma(xv, yv1);
                yv2 = av2.fma(xv, yv2);
                yv3 = av3.fma(xv, yv3);
            }
            float accum0 = yv0.reduceLanes(VectorOperators.ADD);
            float accum1 = yv1.reduceLanes(VectorOperators.ADD);
            float accum2 = yv2.reduceLanes(VectorOperators.ADD);
            float accum3 = yv3.reduceLanes(VectorOperators.ADD);
            for (; col < m; col++) {
                accum0 += a[col + row * lda + aOffset] * x[col + xOffset];
                accum1 += a[col + (row + 1) * lda + aOffset] * x[col + xOffset];
                accum2 += a[col + (row + 2) * lda + aOffset] * x[col + xOffset];
                accum3 += a[col + (row + 3) * lda + aOffset] * x[col + xOffset];
            }
            y[row + yOffset] += alpha * accum0;
            y[row + 1 + yOffset] += alpha * accum1;
            y[row + 2 + yOffset] += alpha * accum2;
            y[row + 3 + yOffset] += alpha * accum3;
        }
        for (; row < n; row++) {
            FloatVector yv = FloatVector.zero(SSPECIES);
            int col = 0;
            for (; col < colUnrollLoopBound; col += SSPECIES.length() * 4) {
                FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, col + xOffset);
                FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, col + SSPECIES.length() + xOffset);
                FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, col + (SSPECIES.length() * 2) + xOffset);
                FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, col + (SSPECIES.length() * 3) + xOffset);

                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, col + row * lda + aOffset);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, col + SSPECIES.length() + row * lda + aOffset);
                FloatVector av2 = FloatVector.fromArray(
                        SSPECIES, a, col + (SSPECIES.length() * 2) + row * lda + aOffset);
                FloatVector av3 = FloatVector.fromArray(
                        SSPECIES, a, col + (SSPECIES.length() * 3) + row * lda + aOffset);

                yv = av0.fma(xv0, av1.fma(xv1, av2.fma(xv2, av3.fma(xv3, yv))));
            }
            for (; col < colLoopBound; col += SSPECIES.length()) {
                FloatVector xv = FloatVector.fromArray(SSPECIES, x, col + xOffset);
                FloatVector av = FloatVector.fromArray(SSPECIES, a, col + row * lda + aOffset);
                yv = xv.fma(av, yv);
            }
            float accum = yv.reduceLanes(VectorOperators.ADD);
            for (; col < m; col++) {
                accum += x[col + xOffset] * a[col + row * lda + aOffset];
            }
            y[row + yOffset] += alpha * accum;
        }
    }

    private static void norSgemvT(int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        int incx, float[] y, int yOffset, int incy) {
        int yIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        for (int row = 0; row < n; row++, yIndex += incy) {
            float accum = 0.0f;
            int xIndex = incx > 0 ? 0 : (m - 1) * (-incx);
            for (int col = 0; col < m; col++, xIndex += incx) {
                accum += a[col + row * lda + aOffset] * x[xIndex + xOffset];
            }
            y[yIndex + yOffset] += alpha * accum;
        }
    }
}
