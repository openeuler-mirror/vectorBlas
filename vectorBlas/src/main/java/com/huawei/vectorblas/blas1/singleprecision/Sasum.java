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

public class Sasum {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static float sasum(int n, float[] x, int xOffset, int incx) {
        if (n < 1 || incx < 1) {
            return 0.0f;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        if (incx == 1) {
            return vecSasum(n, x, xOffset);
        }
        return norSasum(n, x, xOffset, incx);
    }

    private static float vecSasum(int n, float[] x, int xOffset) {
        int xIndex = 0;
        FloatVector resVec = FloatVector.zero(SSPECIES);
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; xIndex < idxLoopBound; xIndex += SSPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(SSPECIES, x, xIndex + xOffset);
            resVec = resVec.add(xv.abs());
        }
        float result = resVec.reduceLanes(VectorOperators.ADD);
        for (; xIndex < n; xIndex++) {
            result += Math.abs(x[xIndex + xOffset]);
        }
        return result;
    }

    private static float norSasum(int n, float[] x, int xOffset, int incx) {
        float result = 0.0f;
        int xIndex = 0;
        for (int count = 0; count < n; count++) {
            result += Math.abs(x[xIndex + xOffset]);
            xIndex += incx;
        }
        return result;
    }
}
