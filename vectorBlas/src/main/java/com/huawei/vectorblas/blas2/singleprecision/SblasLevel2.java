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

package com.huawei.vectorblas.blas2.singleprecision;

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class SblasLevel2 {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;

    protected static void sMulBeta(int size, float beta, float[] sy, int yOffset, int incy) {
        if (incy == 1) {
            FloatVector betaVec = FloatVector.broadcast(SSPECIES, beta);
            int idx = 0;
            for (; idx < SSPECIES.loopBound(size); idx += SSPECIES.length()) {
                FloatVector yv = FloatVector.fromArray(SSPECIES, sy, idx + yOffset);
                betaVec.mul(yv).intoArray(sy, idx + yOffset);
            }
            for (; idx < size; idx++) {
                sy[idx + yOffset] = beta * sy[idx + yOffset];
            }
        } else {
            int yIndex = incy >= 0 ? 0 : (1 - size) * incy;
            if (BlasUtils.isZero(beta)) {
                for (int i = 0; i < size; i++, yIndex += incy) {
                    sy[yIndex + yOffset] = 0.0f;
                }
            } else {
                for (int i = 0; i < size; i++, yIndex += incy) {
                    sy[yIndex + yOffset] = beta * sy[yIndex + yOffset];
                }
            }
        }
    }
}
