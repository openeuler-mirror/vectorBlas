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

public class Isamax {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    public static int isamax(int n, float[] x, int xOffset, int incx) {
        if (n <= 0 || incx <= 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        if (incx == 1) {
            return vecIsamax(n, x, xOffset);
        } else {
            return norIsamax(n, x, xOffset, incx);
        }
    }

    private static int vecIsamax(int n, float[] x, int xOffset) {
        float max = 0.0f;
        int indexOfMaxVec = 0;
        int index = 0;
        int idxLoopBound = SSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += SSPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(SSPECIES, x, index + xOffset);
            float maxOfLanes = xv.abs().reduceLanes(VectorOperators.MAX);
            if (max < maxOfLanes) {
                max = maxOfLanes;
                indexOfMaxVec = index;
            }
        }
        int indexOfMaxValue = 0;
        for (int j = indexOfMaxVec; j < indexOfMaxVec + SSPECIES.length(); j++) {
            if (max <= Math.abs(x[j + xOffset])) {
                indexOfMaxValue = j + 1;
                break;
            }
        }
        for (; index < n; index++) {
            if (max < Math.abs(x[index + xOffset])) {
                max = Math.abs(x[index + xOffset]);
                indexOfMaxValue = index + 1;
            }
        }
        return indexOfMaxValue;
    }

    private static int norIsamax(int n, float[] x, int xOffset, int incx) {
        int indexOfMaxValue = 1;
        float max = Math.abs(x[xOffset]);
        int xIndex = incx;
        for (int j = 2; j <= n; j++) {
            float val = Math.abs(x[xIndex + xOffset]);
            if (val > max) {
                indexOfMaxValue = j;
                max = val;
            }
            xIndex += incx;
        }
        return indexOfMaxValue;
    }
}
