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

public class Saxpy {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void saxpy(int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        if (n < 1 || BlasUtils.isZero(alpha)) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            vecSaxpy(n, alpha, x, xOffset, y, yOffset);
        } else {
            norSaxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
        }
    }

    private static void vecSaxpy(int n, float alpha, float[] x, int xOffset, float[] y, int yOffset) {
        FloatVector alphaVec = FloatVector.broadcast(SSPECIES, alpha);
        int index = 0;
        int idxLoopBound = loopBound(n, (SSPECIES.length() * 4));
        for (; index < idxLoopBound; index += SSPECIES.length() * 4) {
            FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector xv1 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() + xOffset);
            FloatVector xv2 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() * 2 + xOffset);
            FloatVector xv3 = FloatVector.fromArray(SSPECIES, x, index + SSPECIES.length() * 3 + xOffset);

            FloatVector yv0 = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            FloatVector yv1 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() + yOffset);
            FloatVector yv2 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() * 2 + yOffset);
            FloatVector yv3 = FloatVector.fromArray(SSPECIES, y, index + SSPECIES.length() * 3 + yOffset);

            xv0.fma(alphaVec, yv0).intoArray(y, index + yOffset);
            xv1.fma(alphaVec, yv1).intoArray(y, index + SSPECIES.length() + yOffset);
            xv2.fma(alphaVec, yv2).intoArray(y, index + SSPECIES.length() * 2 + yOffset);
            xv3.fma(alphaVec, yv3).intoArray(y, index + SSPECIES.length() * 3 + yOffset);
        }
        for (; index < SSPECIES.loopBound(n); index += SSPECIES.length()) {
            FloatVector xv0 = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector yv0 = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            xv0.fma(alphaVec, yv0).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            y[index + yOffset] += alpha * x[index + xOffset];
        }
    }

    private static void norSaxpy(int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
        int incy) {
        int xIndex = incx >= 0 ? 0 : (n - 1) * -incx;
        int yIndex = incy >= 0 ? 0 : (n - 1) * -incy;
        for (int count = 0; count < n; count++) {
            y[yIndex + yOffset] += alpha * x[xIndex + xOffset];
            xIndex += incx;
            yIndex += incy;
        }
    }
}
