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

public class Scopy {
    public static void scopy(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        if (n <= 0) {
            return;
        }
        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("y", yOffset, Math.abs(incy) * (n - 1), y.length);
        if (incx == 1 && incy == 1) {
            System.arraycopy(x, xOffset, y, yOffset, n);
        } else {
            norScopy(n, x, xOffset, incx, y, yOffset, incy);
        }
    }

    private static void norScopy(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        int xIndex = incx < 0 ? (-n + 1) * incx : 0;
        int yIndex = incy < 0 ? (-n + 1) * incy : 0;
        for (int i = n; i > 0; --i) {
            y[yIndex + yOffset] = x[xIndex + xOffset];
            xIndex += incx;
            yIndex += incy;
        }
    }
}
