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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class Sger {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;
    private static final int UNROLL_SIZE = 4;

    public static void sger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
    int incy, float[] a, int aOffset, int lda) {
        BlasUtils.checkParameter("SGER", 1, m >= 0);
        BlasUtils.checkParameter("SGER", 2, n >= 0);
        BlasUtils.checkParameter("SGER", 5, incx != 0);
        BlasUtils.checkParameter("SGER", 7, incy != 0);
        BlasUtils.checkParameter("SGER", 9, lda >= Math.max(1, m));

        if (m == 0 || n == 0 || BlasUtils.isZero(alpha)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (m - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("a", aOffset, (m - 1) + (n - 1) * lda, a.length);

        if (incx == 1 && incy == 1) {
            vecSger(m, n, alpha, x, xOffset, y, yOffset, a, aOffset, lda);
        } else {
            normalSger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
        }
    }

    private static void vecSger(int m, int n, float alpha, float[] x, int xOffset, float[] y, int yOffset, float[] a,
    int aOffset, int lda) {
        int colLoopBound = loopBound(n, UNROLL_SIZE);
        int rowLoopBound = loopBound(m, UNROLL_SIZE * SSPECIES.length());
        int col = 0;
        for (; col < colLoopBound; col += UNROLL_SIZE) {
            FloatVector alphaMulYv0 = FloatVector.broadcast(SSPECIES, alpha * y[col + yOffset]);
            FloatVector alphaMulYv1 = FloatVector.broadcast(SSPECIES, alpha * y[col + 1 + yOffset]);
            FloatVector alphaMulYv2 = FloatVector.broadcast(SSPECIES, alpha * y[col + 2 + yOffset]);
            FloatVector alphaMulYv3 = FloatVector.broadcast(SSPECIES, alpha * y[col + 3 + yOffset]);
            int row = 0;
            for (; row < rowLoopBound; row += UNROLL_SIZE * SSPECIES.length()) {
                FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, row + xOffset);
                FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, row + SSPECIES.length() + xOffset);
                FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, row + 2 * SSPECIES.length() + xOffset);
                FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, row + 3 * SSPECIES.length() + xOffset);

                FloatVector av00 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector av01 = FloatVector.fromArray(SSPECIES, a, row + SSPECIES.length() + col * lda + aOffset);
                FloatVector av02 = FloatVector.fromArray(SSPECIES, a,
                        row + 2 * SSPECIES.length() + col * lda + aOffset);
                FloatVector av03 = FloatVector.fromArray(SSPECIES, a,
                        row + 3 * SSPECIES.length() + col * lda + aOffset);

                xv0.fma(alphaMulYv0, av00).intoArray(a, row + col * lda + aOffset);
                xv1.fma(alphaMulYv0, av01).intoArray(a, row + SSPECIES.length() + col * lda + aOffset);
                xv2.fma(alphaMulYv0, av02).intoArray(a, row + 2 * SSPECIES.length() + col * lda + aOffset);
                xv3.fma(alphaMulYv0, av03).intoArray(a, row + 3 * SSPECIES.length() + col * lda + aOffset);

                FloatVector av10 = FloatVector.fromArray(SSPECIES, a, row + (col + 1) * lda + aOffset);
                FloatVector av11 = FloatVector.fromArray(SSPECIES, a,
                        row + SSPECIES.length() + (col + 1) * lda + aOffset);
                FloatVector av12 = FloatVector.fromArray(SSPECIES, a,
                        row + 2 * SSPECIES.length() + (col + 1) * lda + aOffset);
                FloatVector av13 = FloatVector.fromArray(SSPECIES, a,
                        row + 3 * SSPECIES.length() + (col + 1) * lda + aOffset);

                xv0.fma(alphaMulYv1, av10).intoArray(a, row + (col + 1) * lda + aOffset);
                xv1.fma(alphaMulYv1, av11).intoArray(a, row + SSPECIES.length() + (col + 1) * lda + aOffset);
                xv2.fma(alphaMulYv1, av12).intoArray(a, row + 2 * SSPECIES.length() + (col + 1) * lda + aOffset);
                xv3.fma(alphaMulYv1, av13).intoArray(a, row + 3 * SSPECIES.length() + (col + 1) * lda + aOffset);

                FloatVector av20 = FloatVector.fromArray(SSPECIES, a, row + (col + 2) * lda + aOffset);
                FloatVector av21 = FloatVector.fromArray(SSPECIES, a,
                        row + SSPECIES.length() + (col + 2) * lda + aOffset);
                FloatVector av22 = FloatVector.fromArray(SSPECIES, a,
                        row + 2 * SSPECIES.length() + (col + 2) * lda + aOffset);
                FloatVector av23 = FloatVector.fromArray(SSPECIES, a,
                        row + 3 * SSPECIES.length() + (col + 2) * lda + aOffset);

                xv0.fma(alphaMulYv2, av20).intoArray(a, row + (col + 2) * lda + aOffset);
                xv1.fma(alphaMulYv2, av21).intoArray(a, row + SSPECIES.length() + (col + 2) * lda + aOffset);
                xv2.fma(alphaMulYv2, av22).intoArray(a, row + 2 * SSPECIES.length() + (col + 2) * lda + aOffset);
                xv3.fma(alphaMulYv2, av23).intoArray(a, row + 3 * SSPECIES.length() + (col + 2) * lda + aOffset);

                FloatVector av30 = FloatVector.fromArray(SSPECIES, a, row + (col + 3) * lda + aOffset);
                FloatVector av31 = FloatVector.fromArray(SSPECIES, a,
                        row + SSPECIES.length() + (col + 3) * lda + aOffset);
                FloatVector av32 = FloatVector.fromArray(SSPECIES, a,
                        row + 2 * SSPECIES.length() + (col + 3) * lda + aOffset);
                FloatVector av33 = FloatVector.fromArray(SSPECIES, a,
                        row + 3 * SSPECIES.length() + (col + 3) * lda + aOffset);

                xv0.fma(alphaMulYv3, av30).intoArray(a, row + (col + 3) * lda + aOffset);
                xv1.fma(alphaMulYv3, av31).intoArray(a, row + SSPECIES.length() + (col + 3) * lda + aOffset);
                xv2.fma(alphaMulYv3, av32).intoArray(a, row + 2 * SSPECIES.length() + (col + 3) * lda + aOffset);
                xv3.fma(alphaMulYv3, av33).intoArray(a, row + 3 * SSPECIES.length() + (col + 3) * lda + aOffset);
            }
            float alphaMulY0 = alpha * y[col + yOffset];
            float alphaMulY1 = alpha * y[col + 1 + yOffset];
            float alphaMulY2 = alpha * y[col + 2 + yOffset];
            float alphaMulY3 = alpha * y[col + 3 + yOffset];
            for (; row < m; row++) {
                a[row + col * lda + aOffset] += alphaMulY0 * x[row + xOffset];
                a[row + (col + 1) * lda + aOffset] += alphaMulY1 * x[row + xOffset];
                a[row + (col + 2) * lda + aOffset] += alphaMulY2 * x[row + xOffset];
                a[row + (col + 3) * lda + aOffset] += alphaMulY3 * x[row + xOffset];
            }
        }
        for (; col < n; col++) {
            int row;
            FloatVector alphaMulYv = FloatVector.broadcast(SSPECIES, alpha * y[col + yOffset]);
            for (row = 0; row < rowLoopBound; row += UNROLL_SIZE * SSPECIES.length()) {
                FloatVector av0 = FloatVector.fromArray(SSPECIES, a, row + col * lda + aOffset);
                FloatVector av1 = FloatVector.fromArray(SSPECIES, a, row + SSPECIES.length() + col * lda + aOffset);
                FloatVector av2 = FloatVector.fromArray(SSPECIES, a, row + 2 * SSPECIES.length() + col * lda + aOffset);
                FloatVector av3 = FloatVector.fromArray(SSPECIES, a, row + 3 * SSPECIES.length() + col * lda + aOffset);

                FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, row + xOffset);
                FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, row + SSPECIES.length() + xOffset);
                FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, row + 2 * SSPECIES.length() + xOffset);
                FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, row + 3 * SSPECIES.length() + xOffset);

                xv0.fma(alphaMulYv, av0).intoArray(a, row + col * lda + aOffset);
                xv1.fma(alphaMulYv, av1).intoArray(a, row + SSPECIES.length() + col * lda + aOffset);
                xv2.fma(alphaMulYv, av2).intoArray(a, row + 2 * SSPECIES.length() + col * lda + aOffset);
                xv3.fma(alphaMulYv, av3).intoArray(a, row + 3 * SSPECIES.length() + col * lda + aOffset);
            }
            float alphaMulY0 = alpha * y[col + yOffset];
            for (; row < m; row++) {
                a[row + col * lda + aOffset] += alphaMulY0 * x[row + xOffset];
            }
        }
    }

    private static void normalSger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
        int incy, float[] a, int aOffset, int lda) {
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
