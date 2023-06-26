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

public class Sscal {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static void sscal(int n, float alpha, float[] x, int xOffset, int incx) {
        if (n < 1 || incx < 1 || Double.compare(alpha, 1.0) == 0) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        if (incx == 1) {
            vecSscal(n, alpha, x, xOffset);
        } else {
            norSscal(n, alpha, x, xOffset, incx);
        }
    }

    private static void vecSscal(int n, float alpha, float[] x, int xOffset) {
        FloatVector alphaVec = FloatVector.broadcast(SSPECIES, alpha);
        int index = 0;
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += SSPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            xv.mul(alphaVec).intoArray(x, index + xOffset);
        }
        for (; index < n; index += 1) {
            x[index + xOffset] *= alpha;
        }
    }

    private static void norSscal(int n, float alpha, float[] x, int xOffset, int incx) {
        int xIndex = 0;
        for (int num = n; num > 0; --num) {
            x[xIndex + xOffset] = alpha * x[xIndex + xOffset];
            xIndex += incx;
        }
    }
}
