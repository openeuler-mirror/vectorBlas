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
import jdk.incubator.vector.VectorSpecies;

public class Srotm {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void srotm(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float[] param,
        int paramOffset) {
        if (n < 1) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        BlasUtils.checkBlasArray("param", paramOffset, 4, param.length);
        if (incx == 1 && incy == 1) {
            vecSrotm(n, x, xOffset, y, yOffset, param, paramOffset);
        } else {
            norSrotm(n, x, xOffset, incx, y, yOffset, incy, param, paramOffset);
        }
    }

    private static void norSrotm(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy,
        float[] param, int paramOffset) {
        float flag = param[paramOffset];
        if (Float.compare(flag, -2.0f) == 0) { // If flag equals -2.0, do nothing and return directly.
            return;
        }
        float h11 = param[paramOffset + 1];
        float h12 = 1.0f;
        float h21 = -1.0f;
        float h22 = param[paramOffset + 4];
        if (Float.compare(flag, -1.0f) == 0) {
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
        } else if (BlasUtils.isZero(flag)) {
            h11 = 1.0f;
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
            h22 = 1.0f;
        }
        int xIndex = incx < 0 ? (-n + 1) * incx : 0;
        int yIndex = incy < 0 ? (-n + 1) * incy : 0;
        for (int num = n; num > 0; --num) {
            float xTmp = x[xIndex + xOffset];
            x[xIndex + xOffset] = h11 * xTmp + h12 * y[yIndex + yOffset];
            y[yIndex + yOffset] = h21 * xTmp + h22 * y[yIndex + yOffset];
            xIndex += incx;
            yIndex += incy;
        }
    }

    private static void vecSrotm(int n, float[] x, int xOffset, float[] y, int yOffset, float[] param,
        int paramOffset) {
        float flag = param[paramOffset];
        if (Float.compare(flag, -2.0f) == 0) { // If flag equals -2.0, do nothing and return directly.
            return;
        }
        float h11 = param[paramOffset + 1];
        float h12 = 1.0f;
        float h21 = -1.0f;
        float h22 = param[paramOffset + 4];
        if (Float.compare(flag, -1.0f) == 0) {
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
        } else if (BlasUtils.isZero(flag)) {
            h11 = 1.0f;
            h12 = param[paramOffset + 3];
            h21 = param[paramOffset + 2];
            h22 = 1.0f;
        }
        FloatVector h11v = FloatVector.broadcast(SSPECIES, h11);
        FloatVector h12v = FloatVector.broadcast(SSPECIES, h12);
        FloatVector h21v = FloatVector.broadcast(SSPECIES, h21);
        FloatVector h22v = FloatVector.broadcast(SSPECIES, h22);
        int index = 0;
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += SSPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            FloatVector yv = FloatVector.fromArray(SSPECIES, y, index + yOffset);
            (xv.mul(h11v)).add(yv.mul(h12v)).intoArray(x, index + xOffset);
            (xv.mul(h21v)).add(yv.mul(h22v)).intoArray(y, index + yOffset);
        }
        for (; index < n; index++) {
            float xTmp = x[index + xOffset];
            x[index + xOffset] = h11 * xTmp + h12 * y[index + yOffset];
            y[index + yOffset] = h21 * xTmp + h22 * y[index + yOffset];
        }
    }
}
