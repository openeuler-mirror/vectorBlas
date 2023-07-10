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

public class Dnrm2 {
    private static final int MINEXPONENT = -1021; // -1021 is the minimum exponent in the model of the type of double.
    private static final int MAXEXPONENT = 1024; // 1024 is the maximum exponent in the model of the type of double.
    private static final int DIGITS = 53; // 53 is the number of significant binary digits of double.
    public static double dnrm2(int n, double[] x, int xOffset, int incx) {
        if (n < 1 || incx < 1) {
            return 0.0;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        return norDnrm2(n, x, xOffset, incx);
    }

    private static double norDnrm2(int n, double[] x, int xOffset, int incx) {
        /*
         * tSml, tBig, sSml, sBig are Blue's scaling constants.
         */
        double tSml = Math.pow(2, Math.ceil((MINEXPONENT - 1) * 0.5d));
        double tBig = Math.pow(2, Math.floor((MAXEXPONENT - DIGITS + 1) * 0.5d));
        double sSml = Math.pow(2, -1 * Math.floor((MINEXPONENT - DIGITS) * 0.5d));
        double sBig = Math.pow(2, -1 * Math.ceil((MAXEXPONENT + DIGITS - 1) * 0.5d));
        boolean notBig = true;
        double aSml = 0.0d;
        double aMed = 0.0d;
        double aBig = 0.0d;

        int xIndex = 0;
        for (int count = 0; count < n; count++) {
            double ax = Math.abs(x[xOffset + xIndex]);
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

        double maxN = Double.MAX_VALUE;
        double scaleVal;
        double sumSq;
        if (aBig > 0.0) {
            if ((aMed > 0.0) || (aMed > maxN) || (Double.compare(aMed, aMed) != 0)) {
                aBig += (aMed * sBig) * sBig;
            }
            scaleVal = 1.0d / sBig;
            sumSq = aBig;
        } else if (aSml > 0.0) {
            if ((aMed > 0.0) || (aMed > maxN) || (Double.compare(aMed, aMed) != 0)) {
                aMed = Math.sqrt(aMed);
                aSml = Math.sqrt(aSml) / sSml;
                double yMin = aSml > aMed ? aMed : aSml;
                double yMax = aSml > aMed ? aSml : aMed;
                scaleVal = 1.0d;
                double yMinDevideMax = yMin / yMax;
                sumSq = yMax * yMax * (1.0d + yMinDevideMax * yMinDevideMax);
            } else {
                scaleVal = 1.0d / sSml;
                sumSq = aSml;
            }
        } else {
            scaleVal = 1.0d;
            sumSq = aMed;
        }
        return scaleVal * Math.sqrt(sumSq);
    }
}
