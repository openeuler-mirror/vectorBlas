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

package com.huawei.vectorblas.blas3.singleprecision;

import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.SGEMM_P;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.SGEMM_Q;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.SGEMM_R;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.SGEMM_UNROLL_N;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.VECTOR_LENGTH;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.VECTOR_LENGTH2;
import static com.huawei.vectorblas.blas3.singleprecision.SblasLevel3.VECTOR_LENGTH4;

import com.huawei.vectorblas.utils.BlasUtils;
import com.huawei.vectorblas.utils.Lsame;

public class Sgemm {
    public static void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int aOffset,
        int lda, float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        BlasUtils.checkParameter("SGEMM", 1, Lsame.lsame(transa, "N") || Lsame.lsame(transa, "T"));
        BlasUtils.checkParameter("SGEMM", 2, Lsame.lsame(transb, "N") || Lsame.lsame(transb, "T"));
        boolean transaFlag = Lsame.lsame(transa, "N");
        boolean transbFlag = Lsame.lsame(transb, "N");
        BlasUtils.checkParameter("SGEMM", 3, m >= 0);
        BlasUtils.checkParameter("SGEMM", 4, n >= 0);
        BlasUtils.checkParameter("SGEMM", 5, k >= 0);
        BlasUtils.checkParameter("SGEMM", 8, lda >= Math.max(1, (transaFlag ? m : k)));
        BlasUtils.checkParameter("SGEMM", 10, ldb >= Math.max(1, (transbFlag ? k : n)));
        BlasUtils.checkParameter("SGEMM", 13, ldc >= Math.max(1, m));

        if (m == 0 || n == 0) {
            return;
        }
        if (Float.compare(beta, 1.0f) != 0) {
            BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
            SblasLevel3.betaMulC(m, n, beta, c, cOffset, ldc);
        }
        if (BlasUtils.isZero(alpha) || k == 0) {
            return;
        }
        BlasUtils.checkBlasArray("a", aOffset, ((transaFlag ? m : k) - 1) + ((transaFlag ? k : m) - 1) * lda, a.length);
        BlasUtils.checkBlasArray("b", bOffset, ((transbFlag ? k : n) - 1) + ((transbFlag ? n : k) - 1) * ldb, b.length);
        BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
        sgemmVector(transa, transb, m, n, k, a, aOffset, lda, alpha, b, bOffset, ldb, c, cOffset, ldc);
    }

    private static void sgemmVector(String transa, String transb, int sizeM, int sizeN, int sizeK, float[] sa,
        int aOffset, int lda, float alpha, float[] sb, int bOffset, int ldb, float[] sc, int cOffset, int ldc) {
        int mc = Math.min(SGEMM_P, sizeM);
        int nc = Math.min(SGEMM_R, sizeN);
        int kc = Math.min(SGEMM_Q, sizeK);
        float[] packa = new float[kc * mc];
        float[] packb = new float[kc * nc];
        for (int ns = 0; ns < sizeN; ns += nc) {
            nc = Math.min(nc, sizeN - ns);
            for (int ks = 0; ks < sizeK; ks += kc) {
                kc = Math.min(kc, sizeK - ks);
                if (Lsame.lsame(transb, "N")) {
                    SblasLevel3.onCopy(kc, nc, sb, ks, ns, bOffset, ldb, packb, 0); // packing matrix b
                } else {
                    otCopy(nc, kc, sb, ns, ks, bOffset, ldb, packb, 0);
                }
                for (int ms = 0; ms < sizeM; ms += mc) {
                    mc = Math.min(mc, sizeM - ms);
                    if (Lsame.lsame(transa, "N")) {
                        SblasLevel3.itCopy(mc, kc, sa, ms, ks, aOffset, lda, packa, 0); // packing matrix a
                    } else {
                        inCopy(kc, mc, sa, ks, ms, aOffset, lda, packa, 0);
                    }
                    SblasLevel3.kernelOperation16x4(mc, nc, kc, alpha, packa, packb, 0, sc, ldc, cOffset, ms, ns);
                }
            }
        }
    }

    /**
     * otCopy method is used for transpose packing matrix in the right.
     */
    private static void otCopy(int sizeM, int sizeN, float[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        float[] dst, int dstOffset) {
        int row = 0;
        int colPackSize = SGEMM_UNROLL_N;
        int dstIndex = 0;
        for (; row < sizeM - sizeM % colPackSize; row += colPackSize) {
            int col = 0;
            for (; col < sizeN; col += 1) {
                System.arraycopy(src, (srcRow + row) + (srcCol + col) * srcLd + srcOffset, dst,
                    dstOffset + dstIndex, SGEMM_UNROLL_N);
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
     */
    private static void inCopy(int sizeM, int sizeN, float[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        float[] dst, int dstOffset) {
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
