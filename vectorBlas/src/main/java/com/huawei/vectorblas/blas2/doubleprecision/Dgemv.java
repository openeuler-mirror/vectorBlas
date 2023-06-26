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

public class Dgemv {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void dgemv(String trans, int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        BlasUtils.checkParameter("DGEMV", 1, Lsame.lsame(trans, "N") || Lsame.lsame(trans, "T"));
        BlasUtils.checkParameter("DGEMV", 2, m >= 0);
        BlasUtils.checkParameter("DGEMV", 3, n >= 0);
        BlasUtils.checkParameter("DGEMV", 6, lda >= Math.max(1, m));
        BlasUtils.checkParameter("DGEMV", 8, incx != 0);
        BlasUtils.checkParameter("DGEMV", 11, incy != 0);
        if (m == 0 || n == 0 || (BlasUtils.isZero(alpha) && Double.compare(beta, 1.0) == 0)) {
            return;
        }
        boolean transFlag = Lsame.lsame(trans, "N");
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * ((transFlag ? n : m) - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * ((transFlag ? m : n) - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (n - 1) * lda + m - 1, a.length);

        if (Double.compare(beta, 1.0) != 0) {
            DblasLevel2.dMulBeta(transFlag ? m : n, beta, y, yOffset, incy);
        }
        if (BlasUtils.isZero(alpha)) {
            return;
        }
        if (transFlag) {
            if (incy == 1) {
                if (incx == 1) {
                    vecDgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset);
                } else {
                    vecDgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset);
                }
            } else {
                norDgemvN(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset, incy);
            }
        } else {
            if (incx == 1) {
                if (incy == 1) {
                    vecDgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset);
                } else {
                    vecDgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, y, yOffset, incy);
                }
            } else {
                norDgemvT(m, n, alpha, a, aOffset, lda, x, xOffset, incx, y, yOffset, incy);
            }
        }
    }

    private static void vecDgemvN(int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, double[] y, int yOffset) {
        int col = 0;
        int colLoopBound = loopBound(n, 4);
        int rowUnrollLoopBound = loopBound(m, DSPECIES.length() * 4);
        int rowLoopBound = loopBound(m, DSPECIES.length());
        for (; col < colLoopBound; col += 4) {
            DoubleVector xv0 = DoubleVector.broadcast(DSPECIES, alpha * x[col + xOffset]);
            DoubleVector xv1 = DoubleVector.broadcast(DSPECIES, alpha * x[col + 1 + xOffset]);
            DoubleVector xv2 = DoubleVector.broadcast(DSPECIES, alpha * x[col + 2 + xOffset]);
            DoubleVector xv3 = DoubleVector.broadcast(DSPECIES, alpha * x[col + 3 + xOffset]);
            int row = 0;
            for (; row < rowUnrollLoopBound; row += DSPECIES.length() * 4) {
                DoubleVector yv0 = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                DoubleVector yv1 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() + yOffset);
                DoubleVector yv2 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() * 2 + yOffset);
                DoubleVector yv3 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() * 3 + yOffset);

                DoubleVector av00 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector av10 = DoubleVector.fromArray(
                    DSPECIES, a, row + DSPECIES.length() + col * lda + aOffset);
                DoubleVector av20 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 2) + col * lda + aOffset);
                DoubleVector av30 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 3) + col * lda + aOffset);

                DoubleVector av01 = DoubleVector.fromArray(DSPECIES, a, row + (col + 1) * lda + aOffset);
                DoubleVector av11 = DoubleVector.fromArray(
                    DSPECIES, a, row + DSPECIES.length() + (col + 1) * lda + aOffset);
                DoubleVector av21 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 2) + (col + 1) * lda + aOffset);
                DoubleVector av31 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 3) + (col + 1) * lda + aOffset);

                DoubleVector av02 = DoubleVector.fromArray(DSPECIES, a, row + (col + 2) * lda + aOffset);
                DoubleVector av12 = DoubleVector.fromArray(
                    DSPECIES, a, row + DSPECIES.length() + (col + 2) * lda + aOffset);
                DoubleVector av22 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 2) + (col + 2) * lda + aOffset);
                DoubleVector av32 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 3) + (col + 2) * lda + aOffset);

                DoubleVector av03 = DoubleVector.fromArray(DSPECIES, a, row + (col + 3) * lda + aOffset);
                DoubleVector av13 = DoubleVector.fromArray(
                    DSPECIES, a, row + DSPECIES.length() + (col + 3) * lda + aOffset);
                DoubleVector av23 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 2) + (col + 3) * lda + aOffset);
                DoubleVector av33 = DoubleVector.fromArray(
                    DSPECIES, a, (row + DSPECIES.length() * 3) + (col + 3) * lda + aOffset);

                av00.fma(xv0, av01.fma(xv1, av02.fma(xv2, av03.fma(xv3, yv0)))).intoArray(y, row + yOffset);
                av10.fma(xv0, av11.fma(xv1, av12.fma(xv2, av13.fma(xv3, yv1))))
                    .intoArray(y, row + DSPECIES.length() + yOffset);
                av20.fma(xv0, av21.fma(xv1, av22.fma(xv2, av23.fma(xv3, yv2))))
                    .intoArray(y, row + DSPECIES.length() * 2 + yOffset);
                av30.fma(xv0, av31.fma(xv1, av32.fma(xv2, av33.fma(xv3, yv3))))
                    .intoArray(y, row + DSPECIES.length() * 3 + yOffset);
            }
            for (; row < rowLoopBound; row += DSPECIES.length()) {
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, row + yOffset);

                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, row + (col + 1) * lda + aOffset);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a, row + (col + 2) * lda + aOffset);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a, row + (col + 3) * lda + aOffset);

                av0.fma(xv0, av1.fma(xv1, av2.fma(xv2, av3.fma(xv3, yv)))).intoArray(y, row + yOffset);
            }
            double x0 = alpha * x[col + xOffset];
            double x1 = alpha * x[col + 1 + xOffset];
            double x2 = alpha * x[col + 2 + xOffset];
            double x3 = alpha * x[col + 3 + xOffset];
            for (; row < m; row++) {
                y[row + yOffset] += x0 * a[row + col * lda + aOffset]
                    + x1 * a[row + (col + 1) * lda + aOffset]
                    + x2 * a[row + (col + 2) * lda + aOffset]
                    + x3 * a[row + (col + 3) * lda + aOffset];
            }
        }
        for (; col < n; col++) {
            if (!BlasUtils.isZero(x[col + xOffset])) {
                DoubleVector bv = DoubleVector.broadcast(DSPECIES, alpha * x[col + xOffset]);
                int row = 0;
                for (; row < rowUnrollLoopBound; row += DSPECIES.length() * 4) {
                    DoubleVector yv0 = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                    DoubleVector yv1 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() + yOffset);
                    DoubleVector yv2 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() * 2 + yOffset);
                    DoubleVector yv3 = DoubleVector.fromArray(DSPECIES, y, row + DSPECIES.length() * 3 + yOffset);

                    DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                    DoubleVector av1 = DoubleVector.fromArray(
                        DSPECIES, a, row + DSPECIES.length() + col * lda + aOffset);
                    DoubleVector av2 = DoubleVector.fromArray(
                        DSPECIES, a, (row + DSPECIES.length() * 2) + col * lda + aOffset);
                    DoubleVector av3 = DoubleVector.fromArray(
                        DSPECIES, a, (row + DSPECIES.length() * 3) + col * lda + aOffset);

                    av0.fma(bv, yv0).intoArray(y, row + yOffset);
                    av1.fma(bv, yv1).intoArray(y, row + DSPECIES.length() + yOffset);
                    av2.fma(bv, yv2).intoArray(y, row + DSPECIES.length() * 2 + yOffset);
                    av3.fma(bv, yv3).intoArray(y, row + DSPECIES.length() * 3 + yOffset);
                }
                for (; row < rowLoopBound; row += DSPECIES.length()) {
                    DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                    DoubleVector av = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                    av.fma(bv, yv).intoArray(y, row + yOffset);
                }
                double alphaX = alpha * x[col + xOffset];
                for (; row < m; row++) {
                    y[row + yOffset] += alphaX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void vecDgemvN(int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx, double[] y, int yOffset) {
        int xIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        for (int col = 0; col < n; col++, xIndex += incx) {
            if (!BlasUtils.isZero(x[xIndex + xOffset])) {
                double alphaMulX = alpha * x[xIndex + xOffset];
                DoubleVector alphaMulXv = DoubleVector.broadcast(DSPECIES, alphaMulX);
                int row = 0;
                int rowLoopBound = DSPECIES.loopBound(m);
                for (; row < rowLoopBound; row += DSPECIES.length()) {
                    DoubleVector av = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                    DoubleVector cv = DoubleVector.fromArray(DSPECIES, y, row + yOffset);
                    av.fma(alphaMulXv, cv).intoArray(y, row + yOffset);
                }
                for (; row < m; row++) {
                    y[row + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void norDgemvN(int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        int xIndex = incx > 0 ? 0 : (n - 1) * (-incx);
        for (int col = 0; col < n; col++, xIndex += incx) {
            if (!BlasUtils.isZero(x[xIndex + xOffset])) {
                double alphaMulX = alpha * x[xIndex + xOffset];
                int yIndex = incy > 0 ? 0 : (m - 1) * (-incy);
                for (int row = 0; row < m; row++, yIndex += incy) {
                    y[yIndex + yOffset] += alphaMulX * a[row + col * lda + aOffset];
                }
            }
        }
    }

    private static void vecDgemvT(int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, double[] y, int yOffset, int incy) {
        int yIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        for (int row = 0; row < n; row++, yIndex += incy) {
            DoubleVector cv = DoubleVector.zero(DSPECIES);
            int col = 0;
            int colLoopBound = DSPECIES.loopBound(m);
            for (; col < colLoopBound; col += DSPECIES.length()) {
                DoubleVector av = DoubleVector.fromArray(DSPECIES, a, col + row * lda + aOffset);
                DoubleVector bv = DoubleVector.fromArray(DSPECIES, x, col + xOffset);
                cv = av.fma(bv, cv);
            }
            double accum = cv.reduceLanes(VectorOperators.ADD);
            for (; col < m; col++) {
                accum += a[col + row * lda + aOffset] * x[col + xOffset];
            }
            y[yIndex + yOffset] += alpha * accum;
        }
    }

    private static void vecDgemvT(int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, double[] y, int yOffset) {
        int row = 0;
        int rowLoopBound = loopBound(n, 4);
        int colUnrollLoopBound = loopBound(m, DSPECIES.length() * 4);
        int colLoopBound = loopBound(m, DSPECIES.length());
        for (; row < rowLoopBound; row += 4) {
            DoubleVector yv0 = DoubleVector.zero(DSPECIES);
            DoubleVector yv1 = DoubleVector.zero(DSPECIES);
            DoubleVector yv2 = DoubleVector.zero(DSPECIES);
            DoubleVector yv3 = DoubleVector.zero(DSPECIES);
            int col = 0;
            for (; col < colUnrollLoopBound; col += DSPECIES.length() * 4) {
                DoubleVector xv0 = DoubleVector.fromArray(DSPECIES, x, col + xOffset);
                DoubleVector xv1 = DoubleVector.fromArray(DSPECIES, x, col + DSPECIES.length() + xOffset);
                DoubleVector xv2 = DoubleVector.fromArray(DSPECIES, x, col + (DSPECIES.length() * 2) + xOffset);
                DoubleVector xv3 = DoubleVector.fromArray(DSPECIES, x, col + (DSPECIES.length() * 3) + xOffset);

                DoubleVector av00 = DoubleVector.fromArray(DSPECIES, a, col + row * lda + aOffset);
                DoubleVector av10 = DoubleVector.fromArray(
                    DSPECIES, a, col + DSPECIES.length() + row * lda + aOffset);
                DoubleVector av20 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 2) + row * lda + aOffset);
                DoubleVector av30 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 3) + row * lda + aOffset);
                yv0 = av00.fma(xv0, av10.fma(xv1, av20.fma(xv2, av30.fma(xv3, yv0))));

                DoubleVector av01 = DoubleVector.fromArray(DSPECIES, a, col + (row + 1) * lda + aOffset);
                DoubleVector av11 = DoubleVector.fromArray(
                    DSPECIES, a, col + DSPECIES.length() + (row + 1) * lda + aOffset);
                DoubleVector av21 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 2) + (row + 1) * lda + aOffset);
                DoubleVector av31 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 3) + (row + 1) * lda + aOffset);
                yv1 = av01.fma(xv0, av11.fma(xv1, av21.fma(xv2, av31.fma(xv3, yv1))));

                DoubleVector av02 = DoubleVector.fromArray(DSPECIES, a, col + (row + 2) * lda + aOffset);
                DoubleVector av12 = DoubleVector.fromArray(
                    DSPECIES, a, col + DSPECIES.length() + (row + 2) * lda + aOffset);
                DoubleVector av22 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 2) + (row + 2) * lda + aOffset);
                DoubleVector av32 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 3) + (row + 2) * lda + aOffset);
                yv2 = av02.fma(xv0, av12.fma(xv1, av22.fma(xv2, av32.fma(xv3, yv2))));

                DoubleVector av03 = DoubleVector.fromArray(DSPECIES, a, col + (row + 3) * lda + aOffset);
                DoubleVector av13 = DoubleVector.fromArray(
                    DSPECIES, a, col + DSPECIES.length() + (row + 3) * lda + aOffset);
                DoubleVector av23 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 2) + (row + 3) * lda + aOffset);
                DoubleVector av33 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 3) + (row + 3) * lda + aOffset);
                yv3 = av03.fma(xv0, av13.fma(xv1, av23.fma(xv2, av33.fma(xv3, yv3))));
            }
            for (; col < colLoopBound; col += DSPECIES.length()) {
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, col + xOffset);

                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, col + row * lda + aOffset);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, col + (row + 1) * lda + aOffset);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a, col + (row + 2) * lda + aOffset);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a, col + (row + 3) * lda + aOffset);

                yv0 = av0.fma(xv, yv0);
                yv1 = av1.fma(xv, yv1);
                yv2 = av2.fma(xv, yv2);
                yv3 = av3.fma(xv, yv3);
            }
            double accum0 = yv0.reduceLanes(VectorOperators.ADD);
            double accum1 = yv1.reduceLanes(VectorOperators.ADD);
            double accum2 = yv2.reduceLanes(VectorOperators.ADD);
            double accum3 = yv3.reduceLanes(VectorOperators.ADD);
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
            DoubleVector yv = DoubleVector.zero(DSPECIES);
            int col = 0;
            for (; col < colUnrollLoopBound; col += DSPECIES.length() * 4) {
                DoubleVector xv0 = DoubleVector.fromArray(DSPECIES, x, col + xOffset);
                DoubleVector xv1 = DoubleVector.fromArray(DSPECIES, x, col + DSPECIES.length() + xOffset);
                DoubleVector xv2 = DoubleVector.fromArray(DSPECIES, x, col + (DSPECIES.length() * 2) + xOffset);
                DoubleVector xv3 = DoubleVector.fromArray(DSPECIES, x, col + (DSPECIES.length() * 3) + xOffset);

                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, col + row * lda + aOffset);
                DoubleVector av1 = DoubleVector.fromArray(
                    DSPECIES, a, col + DSPECIES.length() + row * lda + aOffset);
                DoubleVector av2 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 2) + row * lda + aOffset);
                DoubleVector av3 = DoubleVector.fromArray(
                    DSPECIES, a, col + (DSPECIES.length() * 3) + row * lda + aOffset);

                yv = av0.fma(xv0, av1.fma(xv1, av2.fma(xv2, av3.fma(xv3, yv))));
            }
            for (; col < colLoopBound; col += DSPECIES.length()) {
                DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, col + xOffset);
                DoubleVector av = DoubleVector.fromArray(DSPECIES, a, col + row * lda + aOffset);
                yv = av.fma(xv, yv);
            }
            double accum = yv.reduceLanes(VectorOperators.ADD);
            for (; col < m; col++) {
                accum += a[col + row * lda + aOffset] * x[col + xOffset];
            }
            y[row + yOffset] += alpha * accum;
        }
    }

    private static void norDgemvT(int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double[] y, int yOffset, int incy) {
        int yIndex = incy > 0 ? 0 : (n - 1) * (-incy);
        for (int j = 0; j < n; j++, yIndex += incy) {
            double accum = 0.0d;
            int xIndex = incx > 0 ? 0 : (m - 1) * (-incx);
            for (int i = 0; i < m; i++, xIndex += incx) {
                accum += a[i + j * lda + aOffset] * x[xIndex + xOffset];
            }
            y[yIndex + yOffset] += alpha * accum;
        }
    }
}
