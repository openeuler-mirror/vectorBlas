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

package com.huawei.vectorblas.blas2.doubleprecision;

import com.huawei.vectorblas.utils.BlasUtils;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class DblasLevel2 {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    protected static void dMulBeta(int size, double beta, double[] dy, int yOffset, int incy) {
        if (incy == 1) {
            DoubleVector betaVec = DoubleVector.broadcast(DSPECIES, beta);
            int idx = 0;
            int idxLoopBound = DSPECIES.loopBound(size);
            for (; idx < idxLoopBound; idx += DSPECIES.length()) {
                DoubleVector yv = DoubleVector.fromArray(DSPECIES, dy, idx + yOffset);
                betaVec.mul(yv).intoArray(dy, idx + yOffset);
            }
            for (; idx < size; idx++) {
                dy[idx + yOffset] = beta * dy[idx + yOffset];
            }
        } else {
            int yIndex = incy >= 0 ? 0 : (1 - size) * incy;
            if (BlasUtils.isZero(beta)) {
                for (int i = 0; i < size; i++, yIndex += incy) {
                    dy[yIndex + yOffset] = 0.0d;
                }
            } else {
                for (int i = 0; i < size; i++, yIndex += incy) {
                    dy[yIndex + yOffset] = beta * dy[yIndex + yOffset];
                }
            }
        }
    }
}
