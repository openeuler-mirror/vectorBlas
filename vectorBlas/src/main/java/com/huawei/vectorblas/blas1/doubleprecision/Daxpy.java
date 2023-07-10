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

public class Daxpy {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void daxpy(int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset,
        int incy) {
        if (n < 1 || BlasUtils.isZero(alpha)) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            vecDaxpy(n, alpha, x, xOffset, y, yOffset);
        } else {
            norDaxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
        }
    }

    private static void vecDaxpy(int n, double alpha, double[] x, int xOffset, double[] y, int yOffset) {
        DoubleVector alphaVec = DoubleVector.broadcast(DSPECIES, alpha);
        int index = 0;
        int loopBound = DSPECIES.loopBound(n);
        for (; index < loopBound; index += DSPECIES.length()) {
            DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            DoubleVector yv = DoubleVector.fromArray(DSPECIES, y, index + yOffset);
            alphaVec.fma(xv, yv).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            y[index + yOffset] += alpha * x[index + xOffset];
        }
    }

    private static void norDaxpy(int n, double alpha, double[] x, int xOffset, int incx,
        double[] y, int yOffset, int incy) {
        int xIndex = incx >= 0 ? 0 : (n - 1) * -incx;
        int yIndex = incy >= 0 ? 0 : (n - 1) * -incy;
        for (int count = 0; count < n; count++) {
            y[yIndex + yOffset] += alpha * x[xIndex + xOffset];
            xIndex += incx;
            yIndex += incy;
        }
    }
}
