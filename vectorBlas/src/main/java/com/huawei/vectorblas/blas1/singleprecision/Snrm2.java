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

public class Snrm2 {
    private static final int MINEXPONENT = -125; // -125 is the minimum exponent in the model of the type of float.
    private static final int MAXEXPONENT = 128; // 128 is the maximum exponent in the model of the type of float.
    private static final int DIGITS = 24; // 24 is the number of significant binary digits of float.
    public static float snrm2(int n, float[] x, int xOffset, int incx) {
        if (n < 1 || incx < 1) {
            return 0.0f;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        return norSnrm2(n, x, xOffset, incx);
    }

    private static float norSnrm2(int n, float[] x, int xOffset, int incx) {
        /*
         * tSml, tBig, sSml, sBig are Blue's scaling constants.
         */
        float tSml = (float) Math.pow(2, Math.ceil((MINEXPONENT - 1) * 0.5f));
        float tBig = (float) Math.pow(2, Math.floor((MAXEXPONENT - DIGITS + 1) * 0.5f));
        float sSml = (float) Math.pow(2, -1 * Math.floor((MINEXPONENT - DIGITS) * 0.5f));
        float sBig = (float) Math.pow(2, -1 * Math.ceil((MAXEXPONENT + DIGITS - 1) * 0.5f));
        boolean notBig = true;
        float aSml = 0.0f;
        float aMed = 0.0f;
        float aBig = 0.0f;

        int xIndex = 0;
        for (int count = 0; count < n; count++) {
            float ax = Math.abs(x[xOffset + xIndex]);
            if (ax > tBig) {
                aBig += (ax * sBig) * (ax * sBig);
                notBig = false;
            } else if (ax < tSml) {
                if (notBig) {
                    aSml += (ax * sSml) * (ax * sSml);
                }
            } else {
                aMed += ax * ax;
            }
            xIndex += incx;
        }

        float maxN = Float.MAX_VALUE;
        float scaleVal;
        float sumSq;
        if (aBig > 0.0) {
            if ((aMed > 0.0) || (aMed > maxN) || (Float.compare(aMed, aMed) != 0)) {
                aBig += (aMed * sBig) * sBig;
            }
            scaleVal = 1.0f / sBig;
            sumSq = aBig;
        } else if (aSml > 0.0) {
            if ((aMed > 0.0) || (aMed > maxN) || (Float.compare(aMed, aMed) != 0)) {
                aMed = (float) Math.sqrt(aMed);
                aSml = (float) Math.sqrt(aSml) / sSml;
                float yMin = aSml > aMed ? aMed : aSml;
                float yMax = aSml > aMed ? aSml : aMed;
                scaleVal = 1.0f;
                float yMinDevideMax = yMin / yMax;
                sumSq = yMax * yMax * (1.0f + yMinDevideMax * yMinDevideMax);
            } else {
                scaleVal = 1.0f / sSml;
                sumSq = aSml;
            }
        } else {
            scaleVal = 1.0f;
            sumSq = aMed;
        }
        return scaleVal * (float) Math.sqrt(sumSq);
    }
}
