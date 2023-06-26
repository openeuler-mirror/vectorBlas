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

public class Dsymm {
    public static void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        BlasUtils.checkParameter("DSYMM", 1, Lsame.lsame(side, "L") || Lsame.lsame(side, "R"));
        BlasUtils.checkParameter("DSYMM", 2, Lsame.lsame(uplo, "U") || Lsame.lsame(uplo, "L"));
        boolean sideFlag = Lsame.lsame(side, "L");
        BlasUtils.checkParameter("DSYMM", 3, m >= 0);
        BlasUtils.checkParameter("DSYMM", 4, n >= 0);
        BlasUtils.checkParameter("DSYMM", 7, lda >= Math.max(1, (sideFlag ? m : n)));
        BlasUtils.checkParameter("DSYMM", 9, ldb >= Math.max(1, m));
        BlasUtils.checkParameter("DSYMM", 12, ldc >= Math.max(1, m));

        if (m == 0 || n == 0) {
            return;
        }
        if (Double.compare(beta, 1.0d) != 0) {
            BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
            DblasLevel3.betaMulC(m, n, beta, c, cOffset, ldc);
        }
        if (BlasUtils.isZero(alpha)) {
            return;
        }
        BlasUtils.checkBlasArray("a", aOffset, ((sideFlag ? m : n) - 1) + ((sideFlag ? m : n) - 1) * lda, a.length);
        BlasUtils.checkBlasArray("b", bOffset, (m - 1) + (n - 1) * ldb, b.length);
        BlasUtils.checkBlasArray("c", cOffset, (m - 1) + (n - 1) * ldc, c.length);
        dsymmVector(side, uplo, m, n, sideFlag ? m : n, a, aOffset, lda, alpha, b, bOffset, ldb, c, cOffset, ldc);
    }

    private static void dsymmVector(String side, String uplo, int sizeM, int sizeN, int sizeK, double[] da, int aOffset,
        int lda, double alpha, double[] db, int bOffset, int ldb, double[] dc, int cOffset, int ldc) {
        int mc = Math.min(DGEMM_P, sizeM);
        int nc = Math.min(DGEMM_R, sizeN);
        int kc = Math.min(DGEMM_Q, sizeK);
        boolean sideFlag = Lsame.lsame(side, "L");
        double[] packa = new double[kc * (sideFlag ? mc : nc)];
        double[] packb = new double[kc * (sideFlag ? nc : mc)];
        for (int ns = 0; ns < sizeN; ns += nc) {
            nc = Math.min(nc, sizeN - ns);
            for (int ks = 0; ks < sizeK; ks += kc) {
                kc = Math.min(kc, sizeK - ks);
                if (Lsame.lsame(side, "L")) {
                    DblasLevel3.onCopy(kc, nc, db, ks, ns, bOffset, ldb, packb, 0);
                } else if (Lsame.lsame(side, "R") && Lsame.lsame(uplo, "U")) {
                    outCopy(kc, nc, da, aOffset, lda, packa, 0, ns, ks);
                } else {
                    oltCopy(kc, nc, da, aOffset, lda, packa, 0, ns, ks);
                }
                for (int ms = 0; ms < sizeM; ms += mc) {
                    mc = Math.min(mc, sizeM - ms);
                    if (Lsame.lsame(side, "L") && Lsame.lsame(uplo, "U")) {
                        iutCopy(kc, mc, da, aOffset, lda, packa, 0, ms, ks);
                        DblasLevel3.kernelOperation8x4(mc, nc, kc, alpha, packa, packb, 0, dc, ldc, cOffset, ms, ns);
                    } else if (Lsame.lsame(side, "L") && Lsame.lsame(uplo, "L")) {
                        iltCopy(kc, mc, da, aOffset, lda, packa, 0, ms, ks);
                        DblasLevel3.kernelOperation8x4(mc, nc, kc, alpha, packa, packb, 0, dc, ldc, cOffset, ms, ns);
                    } else {
                        DblasLevel3.itCopy(mc, kc, db, ms, ks, bOffset, ldb, packb, 0);
                        DblasLevel3.kernelOperation8x4(mc, nc, kc, alpha, packb, packa, 0, dc, ldc, cOffset, ms, ns);
                    }
                }
            }
        }
    }

    /**
     * oltCopy method is used for packing lower matrix in the right.
     */
    private static void oltCopy(int sizeM, int sizeN, double[] src, int srcOffset, int srcLd, double[] dst,
        int dstOffset, int posX, int posY) {
        int dstIndex = 0;
        int countJ = sizeN;
        int[] vectorLenthList = {DGEMM_UNROLL_N, 1};
        for (int vectorLen : vectorLenthList) {
            while (countJ - vectorLen >= 0) {
                int delta = posX - posY;
                int[] offset = new int[vectorLen];
                for (int index = 0; index < vectorLen; index++) {
                    if (delta > -index) {
                        offset[index] = posX + index + posY * srcLd;
                    } else {
                        offset[index] = posY + (posX + index) * srcLd;
                    }
                }

                int countI = sizeM;
                for (; countI > 0; countI--) {
                    // read and write data
                    for (int index = 0; index < vectorLen; index++) {
                        dst[dstOffset + dstIndex] = src[srcOffset + offset[index]];
                        dstIndex += 1;
                        if (delta > -index) {
                            offset[index] += srcLd;
                        } else {
                            offset[index]++;
                        }
                    }
                    delta--;
                }

                posX += vectorLen;
                countJ -= vectorLen;
            }
        }
    }

    /**
     * outCopy method is used for packing upper matrix in the right.
     */
    private static void outCopy(int sizeM, int sizeN, double[] src, int srcOffset, int srcLd, double[] dst,
        int dstOffset, int posX, int posY) {
        int dstIndex = 0;
        int countJ = sizeN;
        int[] vectorLenthList = {DGEMM_UNROLL_N, 1};
        for (int vectorLen : vectorLenthList) {
            while (countJ - vectorLen >= 0) {
                int delta = posX - posY;
                int[] offset = new int[vectorLen];
                for (int index = 0; index < vectorLen; index++) {
                    if (delta > -index) {
                        offset[index] = posY + (posX + index) * srcLd;
                    } else {
                        offset[index] = posX + index + posY * srcLd;
                    }
                }

                int countI = sizeM;
                for (; countI > 0; countI--) {
                    // read and write data
                    for (int index = 0; index < vectorLen; index++) {
                        dst[dstOffset + dstIndex] = src[srcOffset + offset[index]];
                        dstIndex += 1;
                        if (delta > -index) {
                            offset[index]++;
                        } else {
                            offset[index] += srcLd;
                        }
                    }
                    delta--;
                }

                posX += vectorLen;
                countJ -= vectorLen;
            }
        }
    }

    /**
     * iltCopy method is used for packing lower matrix in the left.
     */
    private static void iltCopy(int sizeM, int sizeN, double[] src, int srcOffset, int srcLd, double[] dst,
        int dstOffset, int posX, int posY) {
        int dstIndex = 0;
        int countJ = sizeN;
        int[] vectorLengthList = {VECTOR_LENGTH4, VECTOR_LENGTH2, VECTOR_LENGTH, 1};
        for (int vectorLen : vectorLengthList) {
            while (countJ - vectorLen >= 0) {
                int delta = posX - posY;
                int[] offset = new int[vectorLen];
                for (int index = 0; index < vectorLen; index++) {
                    if (delta > -index) {
                        offset[index] = posX + index + posY * srcLd;
                    } else {
                        offset[index] = posY + (posX + index) * srcLd;
                    }
                }

                int countI = sizeM;
                for (; countI > 0; countI--) {
                    // read and write data
                    for (int index = 0; index < vectorLen; index++) {
                        dst[dstOffset + dstIndex] = src[srcOffset + offset[index]];
                        dstIndex += 1;
                        if (delta > -index) {
                            offset[index] += srcLd;
                        } else {
                            offset[index]++;
                        }
                    }
                    delta--;
                }

                posX += vectorLen;
                countJ -= vectorLen;
            }
        }
    }

    /**
     * iutCopy method is used for packing upper matrix in the left.
     */
    private static void iutCopy(int sizeM, int sizeN, double[] src, int srcOffset, int srcLd, double[] dst,
        int dstOffset, int posX, int posY) {
        int dstIndex = 0;
        int countJ = sizeN;
        int[] vectorLengthList = {VECTOR_LENGTH4, VECTOR_LENGTH2, VECTOR_LENGTH, 1};
        for (int vectorLen : vectorLengthList) {
            while (countJ - vectorLen >= 0) {
                int delta = posX - posY;
                int[] offset = new int[vectorLen];
                for (int index = 0; index < vectorLen; index++) {
                    if (delta > -index) {
                        offset[index] = posY + (posX + index) * srcLd;
                    } else {
                        offset[index] = posX + index + posY * srcLd;
                    }
                }

                int countI = sizeM;
                for (; countI > 0; countI--) {
                    // read and write data
                    for (int index = 0; index < vectorLen; index++) {
                        dst[dstOffset + dstIndex] = src[srcOffset + offset[index]];
                        dstIndex += 1;
                        if (delta > -index) {
                            offset[index]++;
                        } else {
                            offset[index] += srcLd;
                        }
                    }
                    delta--;
                }

                posX += vectorLen;
                countJ -= vectorLen;
            }
        }
    }
}
