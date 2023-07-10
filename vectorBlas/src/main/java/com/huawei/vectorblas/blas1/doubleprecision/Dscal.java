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
import jdk.incubator.vector.VectorSpecies;

public class Dscal {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;

    public static void dscal(int n, double alpha, double[] x, int xOffset, int incx) {
        if (n < 1 || incx < 1 || Double.compare(alpha, 1.0) == 0) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        if (incx == 1) {
            vecDscal(n, alpha, x, xOffset);
        } else {
            norDscal(n, alpha, x, xOffset, incx);
        }
    }

    private static void vecDscal(int n, double alpha, double[] x, int xOffset) {
        DoubleVector alpv = DoubleVector.broadcast(DSPECIES, alpha);
        int index = 0;
        int idxLoopBound = DSPECIES.loopBound(n);
        for (; index < idxLoopBound; index += DSPECIES.length()) {
            DoubleVector xv = DoubleVector.fromArray(DSPECIES, x, index + xOffset);
            xv.mul(alpv).intoArray(x, index + xOffset);
        }
        for (; index < n; index++) {
            x[index + xOffset] *= alpha;
        }
    }

    private static void norDscal(int n, double alpha, double[] x, int xOffset, int incx) {
        int xInitIndex = 0;
        for (int num = 0; num < n; num++) {
            x[xInitIndex + xOffset] = alpha * x[xInitIndex + xOffset];
            xInitIndex += incx;
        }
    }
}
