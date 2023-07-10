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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Dger {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;
    private static final int UNROLL_SIZE = 4;

    public static void dger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset,
                            int incy, double[] a, int aOffset, int lda) {
        BlasUtils.checkParameter("DGER", 1, m >= 0);
        BlasUtils.checkParameter("DGER", 2, n >= 0);
        BlasUtils.checkParameter("DGER", 5, incx != 0);
        BlasUtils.checkParameter("DGER", 7, incy != 0);
        BlasUtils.checkParameter("DGER", 9, lda >= Math.max(1, m));

        if (m == 0 || n == 0 || BlasUtils.isZero(alpha)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (m - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (m - 1) + (n - 1) * lda, a.length);

        if (incx == 1 && incy == 1) {
            vecDger(m, n, alpha, x, xOffset, y, yOffset, a, aOffset, lda);
        } else {
            normalDger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
        }
    }

    private static void vecDger(int m, int n, double alpha, double[] x, int xOffset, double[] y, int yOffset,
        double[] a, int aOffset, int lda) {
        int colLoopBound = loopBound(n, UNROLL_SIZE);
        int rowLoopBound = loopBound(m, UNROLL_SIZE * DSPECIES.length());
        int col = 0;
        for (; col < colLoopBound; col += UNROLL_SIZE) {
            DoubleVector alphaMulYv0 = DoubleVector.broadcast(DSPECIES, alpha * y[col + yOffset]);
            DoubleVector alphaMulYv1 = DoubleVector.broadcast(DSPECIES, alpha * y[col + 1 + yOffset]);
            DoubleVector alphaMulYv2 = DoubleVector.broadcast(DSPECIES, alpha * y[col + 2 + yOffset]);
            DoubleVector alphaMulYv3 = DoubleVector.broadcast(DSPECIES, alpha * y[col + 3 + yOffset]);
            int row = 0;
            for (; row < rowLoopBound; row += UNROLL_SIZE * DSPECIES.length()) {
                DoubleVector xv0 = DoubleVector.fromArray(DSPECIES, x, row + xOffset);
                DoubleVector xv1 = DoubleVector.fromArray(DSPECIES, x, row + DSPECIES.length() + xOffset);
                DoubleVector xv2 = DoubleVector.fromArray(DSPECIES, x, row + 2 * DSPECIES.length() + xOffset);
                DoubleVector xv3 = DoubleVector.fromArray(DSPECIES, x, row + 3 * DSPECIES.length() + xOffset);

                DoubleVector av00 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector av01 = DoubleVector.fromArray(DSPECIES, a,
                        row + DSPECIES.length() + col * lda + aOffset);
                DoubleVector av02 = DoubleVector.fromArray(DSPECIES, a,
                        row + 2 * DSPECIES.length() + col * lda + aOffset);
                DoubleVector av03 = DoubleVector.fromArray(DSPECIES, a,
                        row + 3 * DSPECIES.length() + col * lda + aOffset);

                xv0.fma(alphaMulYv0, av00).intoArray(a, row + col * lda + aOffset);
                xv1.fma(alphaMulYv0, av01).intoArray(a, row + DSPECIES.length() + col * lda + aOffset);
                xv2.fma(alphaMulYv0, av02).intoArray(a, row + 2 * DSPECIES.length() + col * lda + aOffset);
                xv3.fma(alphaMulYv0, av03).intoArray(a, row + 3 * DSPECIES.length() + col * lda + aOffset);

                DoubleVector av10 = DoubleVector.fromArray(DSPECIES, a, row + (col + 1) * lda + aOffset);
                DoubleVector av11 = DoubleVector.fromArray(DSPECIES, a,
                        row + DSPECIES.length() + (col + 1) * lda + aOffset);
                DoubleVector av12 = DoubleVector.fromArray(DSPECIES, a,
                        row + 2 * DSPECIES.length() + (col + 1) * lda + aOffset);
                DoubleVector av13 = DoubleVector.fromArray(DSPECIES, a,
                        row + 3 * DSPECIES.length() + (col + 1) * lda + aOffset);

                xv0.fma(alphaMulYv1, av10).intoArray(a, row + (col + 1) * lda + aOffset);
                xv1.fma(alphaMulYv1, av11).intoArray(a, row + DSPECIES.length() + (col + 1) * lda + aOffset);
                xv2.fma(alphaMulYv1, av12).intoArray(a, row + 2 * DSPECIES.length() + (col + 1) * lda + aOffset);
                xv3.fma(alphaMulYv1, av13).intoArray(a, row + 3 * DSPECIES.length() + (col + 1) * lda + aOffset);

                DoubleVector av20 = DoubleVector.fromArray(DSPECIES, a, row + (col + 2) * lda + aOffset);
                DoubleVector av21 = DoubleVector.fromArray(DSPECIES, a,
                        row + DSPECIES.length() + (col + 2) * lda + aOffset);
                DoubleVector av22 = DoubleVector.fromArray(DSPECIES, a,
                        row + 2 * DSPECIES.length() + (col + 2) * lda + aOffset);
                DoubleVector av23 = DoubleVector.fromArray(DSPECIES, a,
                        row + 3 * DSPECIES.length() + (col + 2) * lda + aOffset);

                xv0.fma(alphaMulYv2, av20).intoArray(a, row + (col + 2) * lda + aOffset);
                xv1.fma(alphaMulYv2, av21).intoArray(a, row + DSPECIES.length() + (col + 2) * lda + aOffset);
                xv2.fma(alphaMulYv2, av22).intoArray(a, row + 2 * DSPECIES.length() + (col + 2) * lda + aOffset);
                xv3.fma(alphaMulYv2, av23).intoArray(a, row + 3 * DSPECIES.length() + (col + 2) * lda + aOffset);

                DoubleVector av30 = DoubleVector.fromArray(DSPECIES, a, row + (col + 3) * lda + aOffset);
                DoubleVector av31 = DoubleVector.fromArray(DSPECIES, a,
                        row + DSPECIES.length() + (col + 3) * lda + aOffset);
                DoubleVector av32 = DoubleVector.fromArray(DSPECIES, a,
                        row + 2 * DSPECIES.length() + (col + 3) * lda + aOffset);
                DoubleVector av33 = DoubleVector.fromArray(DSPECIES, a,
                        row + 3 * DSPECIES.length() + (col + 3) * lda + aOffset);

                xv0.fma(alphaMulYv3, av30).intoArray(a, row + (col + 3) * lda + aOffset);
                xv1.fma(alphaMulYv3, av31).intoArray(a, row + DSPECIES.length() + (col + 3) * lda + aOffset);
                xv2.fma(alphaMulYv3, av32).intoArray(a, row + 2 * DSPECIES.length() + (col + 3) * lda + aOffset);
                xv3.fma(alphaMulYv3, av33).intoArray(a, row + 3 * DSPECIES.length() + (col + 3) * lda + aOffset);
            }
            double alphaMulY0 = alpha * y[col + yOffset];
            double alphaMulY1 = alpha * y[col + 1 + yOffset];
            double alphaMulY2 = alpha * y[col + 2 + yOffset];
            double alphaMulY3 = alpha * y[col + 3 + yOffset];
            for (; row < m; row++) {
                a[row + col * lda + aOffset] += alphaMulY0 * x[row + xOffset];
                a[row + (col + 1) * lda + aOffset] += alphaMulY1 * x[row + xOffset];
                a[row + (col + 2) * lda + aOffset] += alphaMulY2 * x[row + xOffset];
                a[row + (col + 3) * lda + aOffset] += alphaMulY3 * x[row + xOffset];
            }
        }
        for (; col < n; col++) {
            DoubleVector alphaMulYv = DoubleVector.broadcast(DSPECIES, alpha * y[col + yOffset]);
            int row = 0;
            for (; row < rowLoopBound; row += UNROLL_SIZE * DSPECIES.length()) {
                DoubleVector av0 = DoubleVector.fromArray(DSPECIES, a, row + col * lda + aOffset);
                DoubleVector av1 = DoubleVector.fromArray(DSPECIES, a, row + DSPECIES.length() + col * lda + aOffset);
                DoubleVector av2 = DoubleVector.fromArray(DSPECIES, a,
                        row + 2 * DSPECIES.length() + col * lda + aOffset);
                DoubleVector av3 = DoubleVector.fromArray(DSPECIES, a,
                        row + 3 * DSPECIES.length() + col * lda + aOffset);

                DoubleVector xv0 = DoubleVector.fromArray(DSPECIES, x, row + xOffset);
                DoubleVector xv1 = DoubleVector.fromArray(DSPECIES, x, row + DSPECIES.length() + xOffset);
                DoubleVector xv2 = DoubleVector.fromArray(DSPECIES, x, row + 2 * DSPECIES.length() + xOffset);
                DoubleVector xv3 = DoubleVector.fromArray(DSPECIES, x, row + 3 * DSPECIES.length() + xOffset);

                xv0.fma(alphaMulYv, av0).intoArray(a, row + col * lda + aOffset);
                xv1.fma(alphaMulYv, av1).intoArray(a, row + DSPECIES.length() + col * lda + aOffset);
                xv2.fma(alphaMulYv, av2).intoArray(a, row + 2 * DSPECIES.length() + col * lda + aOffset);
                xv3.fma(alphaMulYv, av3).intoArray(a, row + 3 * DSPECIES.length() + col * lda + aOffset);
            }
            double alphaMulY0 = alpha * y[col + yOffset];
            for (; row < m; row++) {
                a[row + col * lda + aOffset] += alphaMulY0 * x[row + xOffset];
            }
        }
    }

    private static void normalDger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y,
        int yOffset, int incy, double[] a, int aOffset, int lda) {
        int xStartIndx = incx > 0 ? 0 : -(m - 1) * incx;
        int yStartIndx = incy > 0 ? 0 : -(n - 1) * incy;

        for (int j = 0; j < n; j++, yStartIndx += incy) {
            if (!BlasUtils.isZero(y[yStartIndx + yOffset])) {
                for (int i = 0, xIndx = xStartIndx; i < m; i++, xIndx += incx) {
                    a[i + j * lda + aOffset] += alpha * x[xIndx + xOffset] * y[yStartIndx + yOffset];
                }
            }
        }
    }
}
