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

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Drotm {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void drotm(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy,
        double[] param, int paramOffset) {
        if (n < 1) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("param", paramOffset, 4, param.length);
        if (incx == 1 && incy == 1) {
            vecDrotm(n, x, xOffset, y, yOffset, param, paramOffset);
        } else {
            norDrotm(n, x, xOffset, incx, y, yOffset, incy, param, paramOffset);
        }
    }

    private static void vecDrotm(int n, double[] x, int xOffset, double[] y, int yOffset, double[] param,
        int paramOffset) {
        double flag = param[paramOffset];
        if (Double.compare(flag, -2.0d) == 0) { // If flag equals -2.0, do nothing and return directly.
            return;
        }
        double h11 = param[paramOffset + 1];
        double h12 = 1.0d;
        double h21 = -1.0d;
        double h22 = param[paramOffset + 4];
        if (Double.compare(flag, -1.0d) == 0) {
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
        } else if (BlasUtils.isZero(flag)) {
            h11 = 1.0d;
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
            h22 = 1.0d;
        }
        DoubleVector h11v = DoubleVector.broadcast(DSPECIES, h11);
        DoubleVector h12v = DoubleVector.broadcast(DSPECIES, h12);
        DoubleVector h21v = DoubleVector.broadcast(DSPECIES, h21);
        DoubleVector h22v = DoubleVector.broadcast(DSPECIES, h22);
        int index = 0;
        int idxLoopBound = DSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += DSPECIES.length()) {
            DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, index + yOffset);
            (xv.mul(h11v)).add(yv.mul(h12v)).intoArray(x, index + xOffset);
            (xv.mul(h21v)).add(yv.mul(h22v)).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            double xTmp = x[index + xOffset];
            x[index + xOffset] = h11 * xTmp + h12 * y[index + yOffset];
            y[index + yOffset] = h21 * xTmp + h22 * y[index + yOffset];
        }
    }

    private static void norDrotm(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy,
        double[] param, int paramOffset) {
        double flag = param[paramOffset];
        if (Double.compare(flag, -2.0d) == 0) { // If flag equals -2.0, do nothing and return directly.
            return;
        }
        double h11 = param[paramOffset + 1];
        double h12 = 1.0d;
        double h21 = -1.0d;
        double h22 = param[paramOffset + 4];
        if (Double.compare(flag, -1.0d) == 0) {
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
        } else if (BlasUtils.isZero(flag)) {
            h11 = 1.0d;
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
            h22 = 1.0d;
        }
        int xInitIndex = incx < 0 ? (-n + 1) * incx : 0;
        int yInitIndex = incy < 0 ? (-n + 1) * incy : 0;
        for (int num = n; num > 0; --num) {
            double xTmp = x[xInitIndex + xOffset];
            x[xInitIndex + xOffset] = h11 * xTmp + h12 * y[yInitIndex + yOffset];
            y[yInitIndex + yOffset] = h21 * xTmp + h22 * y[yInitIndex + yOffset];
            xInitIndex += incx;
            yInitIndex += incy;
        }
    }
}
