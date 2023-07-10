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

package com.huawei.vectorblas.blas1.doubleprecision;

import static com.huawei.vectorblas.utils.ArrayUtil.loopBound;

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Drot {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void drot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double c,
        double s) {
        if (n < 1) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            vecDrot(n, x, xOffset, y, yOffset, c, s);
        } else {
            norDrot(n, x, xOffset, incx, y, yOffset, incy, c, s);
        }
    }

    private static void vecDrot(int n, double[] x, int xOffset, double[] y, int yOffset, double c, double s) {
        DoubleVector cv = DoubleVector.broadcast(DSPECIES, c);
        DoubleVector sv = DoubleVector.broadcast(DSPECIES, s);
        DoubleVector nsv = DoubleVector.broadcast(DSPECIES, -s);
        int index = 0;
        int idxLoopBound = loopBound(n, DSPECIES.length() * 4);
        for (; index < idxLoopBound; index += DSPECIES.length() * 4) {
            DoubleVector xv0 = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            DoubleVector xv1 = DoubleVector.fromArray(DSPECIES, x, index + DSPECIES.length() + xOffset);
            DoubleVector xv2 = DoubleVector.fromArray(DSPECIES, x, index + DSPECIES.length() * 2 + xOffset);
            DoubleVector xv3 = DoubleVector.fromArray(DSPECIES, x, index + DSPECIES.length() * 3 + xOffset);

            DoubleVector yv0 = DoubleVector.fromArray(DSPECIES, y, index + yOffset);
            DoubleVector yv1 = DoubleVector.fromArray(DSPECIES, y, index + DSPECIES.length() + yOffset);
            DoubleVector yv2 = DoubleVector.fromArray(DSPECIES, y, index + DSPECIES.length() * 2 + yOffset);
            DoubleVector yv3 = DoubleVector.fromArray(DSPECIES, y, index + DSPECIES.length() * 3 + yOffset);

            xv0.fma(cv, yv0.mul(sv)).intoArray(x, index + xOffset);
            xv1.fma(cv, yv1.mul(sv)).intoArray(x, index + DSPECIES.length() + xOffset);
            xv2.fma(cv, yv2.mul(sv)).intoArray(x, index + DSPECIES.length() * 2 + xOffset);
            xv3.fma(cv, yv3.mul(sv)).intoArray(x, index + DSPECIES.length() * 3 + xOffset);

            xv0.fma(nsv, yv0.mul(cv)).intoArray(y, index + yOffset);
            xv1.fma(nsv, yv1.mul(cv)).intoArray(y, index + DSPECIES.length() + yOffset);
            xv2.fma(nsv, yv2.mul(cv)).intoArray(y, index + DSPECIES.length() * 2 + yOffset);
            xv3.fma(nsv, yv3.mul(cv)).intoArray(y, index + DSPECIES.length() * 3 + yOffset);
        }
        for (; index < DSPECIES.loopBound(n); index += DSPECIES.length()) {
            DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, index + yOffset);

            xv.fma(cv, yv.mul(sv)).intoArray(x, index + xOffset);
            xv.fma(nsv, yv.mul(cv)).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            double tmp = x[index + xOffset];
            x[index + xOffset] = c * x[index + xOffset] + s * y[index + yOffset];
            y[index + yOffset] = c * y[index + yOffset] - s * tmp;
        }
    }

    private static void norDrot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy,
                                double c, double s) {
        int xInitIndex = incx < 0 ? (-n + 1) * incx : 0;
        int yInitIndex = incy < 0 ? (-n + 1) * incy : 0;
        for (int num = n; num > 0; --num) {
            double tmp = x[xInitIndex + xOffset];
            x[xInitIndex + xOffset] = c * tmp + s * y[yInitIndex + yOffset];
            y[yInitIndex + yOffset] = -s * tmp + c * y[yInitIndex + yOffset];
            xInitIndex += incx;
            yInitIndex += incy;
        }
    }
}
