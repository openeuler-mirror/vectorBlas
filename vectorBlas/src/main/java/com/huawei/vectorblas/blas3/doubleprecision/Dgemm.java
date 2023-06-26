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

package com.huawei.vectorblas.blas3.doubleprecision;

import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.DGEMM_P;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.DGEMM_Q;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.DGEMM_R;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.DGEMM_UNROLL_N;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.VECTOR_LENGTH;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.VECTOR_LENGTH2;
import static com.huawei.vectorblas.blas3.doubleprecision.DblasLevel3.VECTOR_LENGTH4;

import com.huawei.vectorblas.utils.BlasUtils;
import com.huawei.vectorblas.utils.Lsame;

public class Dgemm {
    public static void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int aOffset,
        int lda, double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        BlasUtils.checkParameter("DGEMM", 1, Lsame.lsame(transa, "N") || Lsame.lsame(transa, "T"));
        BlasUtils.checkParameter("DGEMM", 2, Lsame.lsame(transb, "N") || Lsame.lsame(transb, "T"));
        boolean transaFlag = Lsame.lsame(transa, "N");
        boolean transbFlag = Lsame.lsame(transb, "N");
        BlasUtils.checkParameter("DGEMM", 3, m >= 0);
        BlasUtils.checkParameter("DGEMM", 4, n >= 0);
        BlasUtils.checkParameter("DGEMM", 5, k >= 0);
        BlasUtils.checkParameter("DGEMM", 8, lda >= Math.max(1, (transaFlag ? m : k)));
        BlasUtils.checkParameter("DGEMM", 10, ldb >= Math.max(1, (transbFlag ? k : n)));
        BlasUtils.checkParameter("DGEMM", 13, ldc >= Math.max(1, m));

        if (m == 0 || n == 0) {
            return;
        }
        if (Double.compare(beta, 1.0d) != 0) {
            BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
            DblasLevel3.betaMulC(m, n, beta, c, cOffset, ldc);
        }
        if (BlasUtils.isZero(alpha) || k == 0) {
            return;
        }
        BlasUtils.checkBlasArray("a", aOffset, ((transaFlag ? m : k) - 1) + ((transaFlag ? k : m) - 1) * lda, a.length);
        BlasUtils.checkBlasArray("b", bOffset, ((transbFlag ? k : n) - 1) + ((transbFlag ? n : k) - 1) * ldb, b.length);
        BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
        dgemmVector(transa, transb, m, n, k, a, aOffset, lda, alpha, b, bOffset, ldb, c, cOffset, ldc);
    }

    private static void dgemmVector(String transa, String transb, int sizeM, int sizeN, int sizeK, double[] da,
        int aOffset, int lda, double alpha, double[] db, int bOffset, int ldb, double[] dc, int cOffset, int ldc) {
        int mc = Math.min(DGEMM_P, sizeM);
        int nc = Math.min(DGEMM_R, sizeN);
        int kc = Math.min(DGEMM_Q, sizeK);
        double[] packa = new double[kc * mc];
        double[] packb = new double[kc * nc];
        for (int ns = 0; ns < sizeN; ns += nc) {
            nc = Math.min(nc, sizeN - ns);
            for (int ks = 0; ks < sizeK; ks += kc) {
                kc = Math.min(kc, sizeK - ks);
                if (Lsame.lsame(transb, "N")) {
                    DblasLevel3.onCopy(kc, nc, db, ks, ns, bOffset, ldb, packb, 0); // packing matrix b
                } else {
                    otCopy(nc, kc, db, ns, ks, bOffset, ldb, packb, 0);
                }
                for (int ms = 0; ms < sizeM; ms += mc) {
                    mc = Math.min(mc, sizeM - ms);
                    if (Lsame.lsame(transa, "N")) {
                        DblasLevel3.itCopy(mc, kc, da, ms, ks, aOffset, lda, packa, 0); // packing matrix a
                    } else {
                        inCopy(kc, mc, da, ks, ms, aOffset, lda, packa, 0);
                    }
                    DblasLevel3.kernelOperation8x4(mc, nc, kc, alpha, packa, packb, 0, dc, ldc, cOffset, ms, ns);
                }
            }
        }
    }

    /**
     * otCopy method is used for transpose packing matrix in the right.
     * For example, when DGEMM_UNROLL_N = 4,
     *       before packing             after packing
     *        1  6  11 16                1  5  9  13
     *        2  7  12 17      --->      2  6  10 14
     *        3  8  13 18                3  7  11 15
     *        4  9  14 19                4  8  12 16
     *        5  10 15 20                17 18 19 20
     */
    private static void otCopy(int sizeM, int sizeN, double[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        double[] dst, int dstOffset) {
        int row = 0;
        int colPackSize = DGEMM_UNROLL_N;
        int dstIndex = 0;
        for (; row < sizeM - sizeM % colPackSize; row += colPackSize) {
            int col = 0;
            for (; col < sizeN; col += 1) {
                System.arraycopy(src, (srcRow + row) + (srcCol + col) * srcLd + srcOffset, dst,
                    dstOffset + dstIndex, DGEMM_UNROLL_N);
                dstIndex += colPackSize;
            }
        }
        for (; row < sizeM; row += 1) {
            int col = 0;
            for (; col < sizeN; col += 1) {
                dst[dstOffset + dstIndex] = src[(srcRow + row) + (srcCol + col) * srcLd + srcOffset];
                dstIndex += 1;
            }
        }
    }

    /**
     * inCopy is used for normally packing matrix in the left.
     * For example, when DGEMM_UNROLL_M = 4,
     *        before packing                 after packing
     *        1  6  11 16 21                 1  2  3  4  21
     *        2  7  12 17 22       --->      5  6  7  8  22
     *        3  8  13 18 23                 9  10 11 12 23
     *        4  9  14 19 24                 13 14 15 16 24
     *        5  10 15 20 25                 17 18 19 20 25
     */
    private static void inCopy(int sizeM, int sizeN, double[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        double[] dst, int dstOffset) {
        int col = 0;
        int dstIndex = 0;
        int[] vectorLengthList = {VECTOR_LENGTH4, VECTOR_LENGTH2, VECTOR_LENGTH, 1};
        for (int vectorLen : vectorLengthList) {
            while (col + vectorLen <= sizeN) {
                int row = 0;
                for (; row < sizeM; row++) {
                    for (int count = 0; count < vectorLen; count++) {
                        dst[dstOffset + dstIndex + count] = src[srcOffset + (srcRow + row) + (srcCol + (col + count))
                            * srcLd];
                    }
                    dstIndex += vectorLen;
                }
                col += vectorLen;
            }
        }
    }
}
