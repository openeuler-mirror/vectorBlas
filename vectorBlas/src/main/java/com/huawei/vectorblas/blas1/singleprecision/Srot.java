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

import static com.huawei.vectorblas.utils.ArrayUtil.loopBound;

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class Srot {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void srot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float c,
        float s) {
        if (n < 1) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            vecSrot(n, x, xOffset, y, yOffset, c, s);
        } else {
            norSrot(n, x, xOffset, incx, y, yOffset, incy, c, s);
        }
    }

    private static void norSrot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy,
                                float c, float s) {
        int xIndex = incx < 0 ? (-n + 1) * incx : 0;
        int yIndex = incy < 0 ? (-n + 1) * incy : 0;
        for (int num = n; num > 0; --num) {
            float tmp = x[xIndex + xOffset];
            x[xIndex + xOffset] = c * tmp + s * y[yIndex + yOffset];
            y[yIndex + yOffset] = -s * tmp + c * y[yIndex + yOffset];
            xIndex += incx;
            yIndex += incy;
        }
    }

    private static void vecSrot(int n, float[] x, int xOffset, float[] y, int yOffset, float c, float s) {
        FloatVector cv = FloatVector.broadcast(SSPECIES, c);
        FloatVector sv = FloatVector.broadcast(SSPECIES, s);
        FloatVector nsv = FloatVector.broadcast(SSPECIES, -s);
        int index = 0;
        int idxLoopBound = loopBound(n, SSPECIES.length() * 4);
        for (; index < idxLoopBound; index += SSPECIES.length() * 4) {
            FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() + xOffset);
            FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() * 2 + xOffset);
            FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() * 3 + xOffset);

            FloatVector yv0 = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            FloatVector yv1 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() + yOffset);
            FloatVector yv2 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() * 2 + yOffset);
            FloatVector yv3 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() * 3 + yOffset);

            xv0.fma(cv, yv0.mul(sv)).intoArray(x, index + xOffset);
            xv1.fma(cv, yv1.mul(sv)).intoArray(x, index + SSPECIES.length() + xOffset);
            xv2.fma(cv, yv2.mul(sv)).intoArray(x, index + SSPECIES.length() * 2 + xOffset);
            xv3.fma(cv, yv3.mul(sv)).intoArray(x, index + SSPECIES.length() * 3 + xOffset);

            xv0.fma(nsv, yv0.mul(cv)).intoArray(y, index + yOffset);
            xv1.fma(nsv, yv1.mul(cv)).intoArray(y, index + SSPECIES.length() + yOffset);
            xv2.fma(nsv, yv2.mul(cv)).intoArray(y, index + SSPECIES.length() * 2 + yOffset);
            xv3.fma(nsv, yv3.mul(cv)).intoArray(y, index + SSPECIES.length() * 3 + yOffset);
        }
        for (; index < SSPECIES.loopBound(n); index += SSPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector yv = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            xv.fma(cv, yv.mul(sv)).intoArray(x, index + xOffset);
            xv.fma(nsv, yv.mul(cv)).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            float tmp = x[index + xOffset];
            x[index + xOffset] = c * tmp + s * y[index + yOffset];
            y[index + yOffset] = c * y[index + yOffset] - s * tmp;
        }
    }
}
