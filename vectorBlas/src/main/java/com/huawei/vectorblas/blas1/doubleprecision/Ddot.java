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
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Ddot {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static double ddot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        if (n < 1) {
            return 0.0d;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            if (xOffset == 0 && yOffset == 0) {
                return vecDdot(n, x, y);
            }
            return vecDdot(n, x, xOffset, y, yOffset);
        }
        return norDdot(n, x, xOffset, incx, y, yOffset, incy);
    }

    private static double vecDdot(int n, double[] x, double[] y) {
        int index = 0;
        DoubleVector sumVec = DoubleVector.zero(DSPECIES);
        int idxLoopBound = DSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += DSPECIES.length()) {
            DoubleVector av = DoubleVector.fromArray(DSPECIES, x, index);
            DoubleVector bv = DoubleVector.fromArray(DSPECIES, y, index);
            sumVec = av.fma(bv, sumVec);
        }
        double sum = sumVec.reduceLanes(VectorOperators.ADD);
        for (; index < n; index++) {
            sum += x[index] * y[index];
        }
        return sum;
    }

    private static double vecDdot(int n, double[] x, int xOffset, double[] y, int yOffset) {
        int index = 0;
        DoubleVector sumVec = DoubleVector.zero(DSPECIES);
        int idxLoopBound = DSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += DSPECIES.length()) {
            DoubleVector av = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            DoubleVector bv = DoubleVector.fromArray(DSPECIES, y, index + yOffset);
            sumVec = av.fma(bv, sumVec);
        }
        double sum = sumVec.reduceLanes(VectorOperators.ADD);
        for (; index < n; index++) {
            sum += x[index + xOffset] * y[index + yOffset];
        }
        return sum;
    }

    private static double norDdot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        int xIndex = incx >= 0 ? 0 : (n - 1) * -incx;
        int yIndex = incy >= 0 ? 0 : (n - 1) * -incy;
        double sum = 0.0d;
        for (int count = 0; count < n; count++) {
            sum += y[yIndex + yOffset] * x[xIndex + xOffset];
            xIndex += incx;
            yIndex += incy;
        }
        return sum;
    }
}
