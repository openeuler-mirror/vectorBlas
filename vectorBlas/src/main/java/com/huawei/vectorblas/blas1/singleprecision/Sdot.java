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

package com.huawei.vectorblas.blas1.singleprecision;

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Sdot {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static float sdot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        if (n < 1) {
            return 0.0f;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            if (xOffset == 0 && yOffset == 0) {
                return vecSdot(n, x, y);
            }
            return vecSdot(n, x, xOffset, y, yOffset);
        }
        return norSdot(n, x, xOffset, incx, y, yOffset, incy);
    }

    private static float vecSdot(int n, float[] x, float[] y) {
        FloatVector sumVec = FloatVector.zero(SSPECIES);
        int index = 0;
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += SSPECIES.length()) {
            FloatVector av = FloatVector.fromArray(SSPECIES, x, index);
            FloatVector bv = FloatVector.fromArray(SSPECIES, y, index);
            sumVec = av.fma(bv, sumVec);
        }
        float sum = sumVec.reduceLanes(VectorOperators.ADD);
        for (; index < n; index++) {
            sum += x[index] * y[index];
        }
        return sum;
    }

    private static float vecSdot(int n, float[] x, int xOffset, float[] y, int yOffset) {
        FloatVector sumVec = FloatVector.zero(SSPECIES);
        int index = 0;
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += SSPECIES.length()) {
            FloatVector av = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector bv = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            sumVec = av.fma(bv, sumVec);
        }
        float sum = sumVec.reduceLanes(VectorOperators.ADD);
        for (; index < n; index++) {
            sum += x[index + xOffset] * y[index + yOffset];
        }
        return sum;
    }

    private static float norSdot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        int xIndex = incx >= 0 ? 0 : (n - 1) * -incx;
        int yIndex = incy >= 0 ? 0 : (n - 1) * -incy;
        float sum = 0.0f;
        for (int count = 0; count < n; count++) {
            sum += y[yIndex + yOffset] * x[xIndex + xOffset];
            xIndex += incx;
            yIndex += incy;
        }
        return sum;
    }
}
