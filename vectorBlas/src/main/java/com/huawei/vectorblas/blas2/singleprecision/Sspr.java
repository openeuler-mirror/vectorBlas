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

import com.huawei.vectorblas.blas1.singleprecision.Saxpy;
import com.huawei.vectorblas.utils.BlasUtils;
import com.huawei.vectorblas.utils.Lsame;

public class Sspr {
    public static void sspr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] ap,
            int aOffset) {
        BlasUtils.checkParameter("SSPR", 1, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        BlasUtils.checkParameter("SSPR", 2, n >= 0);
        BlasUtils.checkParameter("SSPR", 5, incx != 0);

        if (n == 0 || BlasUtils.isZero(alpha)) {
            return;
        }

        BlasUtils.checkBlasArray("x", xOffset, Math.abs(incx) * (n - 1), x.length);
        BlasUtils.checkBlasArray("a", aOffset, (1 + n) * n / 2 - 1, ap.length);

        boolean uploFlag = Lsame.lsame(uplo, "U");
        int xStartIndx = 0;
        if (incx <= 0) {
            xStartIndx = -(n - 1) * incx;
        }

        int cnt = 0;
        if (incx >= 0) {
            for (int j = 0, xIndx = xStartIndx; j < n; j++, xIndx += incx) {
                int colCnt = uploFlag ? j + 1 : n - j;
                if (!BlasUtils.isZero(x[xIndx + xOffset])) {
                    int kIndx = uploFlag ? 0 : xIndx;
                    Saxpy.saxpy(colCnt, alpha * x[xIndx + xOffset], x, xOffset + kIndx, incx, ap, aOffset + cnt, 1);
                }
                cnt += colCnt;
            }
        } else {
            for (int j = 0, xIndx = xStartIndx; j < n; j++, xIndx += incx) {
                int colCnt = uploFlag ? j + 1 : n - j;
                if (!BlasUtils.isZero(x[xIndx + xOffset])) {
                    int kIndx = uploFlag ? xIndx : 0;
                    Saxpy.saxpy(colCnt, alpha * x[xIndx + xOffset], x, xOffset + kIndx, incx, ap, aOffset + cnt, 1);
                }
                cnt += colCnt;
            }
        }
    }
}
