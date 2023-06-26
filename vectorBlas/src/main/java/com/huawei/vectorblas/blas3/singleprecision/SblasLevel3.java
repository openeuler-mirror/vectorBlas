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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class SblasLevel3 {
    private static final VectorSpecies<Float> SSPECIES = FloatVector.SPECIES_MAX;
    protected static final int SGEMM_P = 256; // Blocking size for m direction.
    protected static final int SGEMM_Q = 256; // Blocking size for k direction.
    protected static final int SGEMM_R = 8192; // Blocking size for n direction.
    protected static final int VECTOR_LENGTH = SSPECIES.length();
    protected static final int VECTOR_LENGTH2 = 2 * VECTOR_LENGTH;
    protected static final int VECTOR_LENGTH3 = 3 * VECTOR_LENGTH;
    protected static final int VECTOR_LENGTH4 = 4 * VECTOR_LENGTH;
    protected static final int SGEMM_UNROLL_M = 4 * VECTOR_LENGTH;
    protected static final int SGEMM_UNROLL_N = 4;

    protected static void betaMulC(int sizeM, int sizeN, float beta, float[] sc, int cOffset, int ldc) {
        FloatVector betav = FloatVector.broadcast(SSPECIES, beta);
        for (int col = 0; col < sizeN; col++) {
            int row = 0;
            for (; row < sizeM - VECTOR_LENGTH; row += VECTOR_LENGTH) {
                FloatVector cv = FloatVector.fromArray(SSPECIES, sc, row + col * ldc + cOffset);
                cv.mul(betav).intoArray(sc, row + col * ldc + cOffset);
            }
            for (; row < sizeM; row++) {
                sc[row + col * ldc + cOffset] *= beta;
            }
        }
    }

    protected static void kernelOperation16x4(int mc, int nc, int kc, float alpha, float[] sa, float[] sb, int bOffset,
        float[] sc, int ldc, int cOffset, int csRow, int csCol) {
        kernelOperation16x4Main(mc, nc, kc, alpha, sa, sb, bOffset, sc, ldc, cOffset, csRow, csCol);
        kernelOperation16x4NBorder(mc, nc, kc, alpha, sa, sb, bOffset, sc, ldc, cOffset, csRow, csCol);
    }

    private static void kernelOperation16x4NBorder(int mc, int nc, int kc, float alpha, float[] sa, float[] sb,
        int bOffset, float[] sc, int ldc, int cOffset, int csRow, int csCol) {
        FloatVector alphaVec = FloatVector.broadcast(SSPECIES, alpha);
        int cCol = csCol + (nc / SGEMM_UNROLL_N) * SGEMM_UNROLL_N;
        int countJ = nc % SGEMM_UNROLL_N;
        for (; countJ > 0; countJ--) {
            int cRow = csRow;
            int aIndx = 0;
            int countI = mc / SGEMM_UNROLL_M;
            for (; countI > 0; countI--) {
                int bIndx = (nc - countJ) * kc;
                FloatVector c00 = FloatVector.zero(SSPECIES);
                FloatVector c10 = FloatVector.zero(SSPECIES);
                FloatVector c20 = FloatVector.zero(SSPECIES);
                FloatVector c30 = FloatVector.zero(SSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);
                    FloatVector a1 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH);
                    FloatVector a2 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH2);
                    FloatVector a3 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH3);
                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c20 = a2.fma(b0, c20);
                    c30 = a3.fma(b0, c30);

                    aIndx += SGEMM_UNROLL_M;
                    bIndx += 1;
                }
                alphaVec.fma(c00, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc)).intoArray(sc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);
                alphaVec.fma(c20, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH2
                    + cCol * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH2 + cCol * ldc);
                alphaVec.fma(c30, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH3
                    + cCol * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH3 + cCol * ldc);

                cRow += SGEMM_UNROLL_M;
            }
            countI = mc % SGEMM_UNROLL_M;
            if (countI >= VECTOR_LENGTH2) {
                int bIndx = (nc - countJ) * kc;
                FloatVector c00 = FloatVector.zero(SSPECIES);
                FloatVector c10 = FloatVector.zero(SSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);
                    FloatVector a1 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH);
                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);

                    aIndx += VECTOR_LENGTH2;
                    bIndx += 1;
                }
                alphaVec.fma(c00, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc)).intoArray(sc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);

                cRow += VECTOR_LENGTH2;
                countI -= VECTOR_LENGTH2;
            }
            if (countI >= VECTOR_LENGTH) {
                int bIndx = (nc - countJ) * kc;
                FloatVector c00 = FloatVector.zero(SSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);
                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);
                    c00 = a0.fma(b0, c00);
                    aIndx += VECTOR_LENGTH;
                    bIndx += 1;
                }
                alphaVec.fma(c00, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc)).intoArray(sc,
                    cOffset + cRow + cCol * ldc);

                cRow += VECTOR_LENGTH;
                countI -= VECTOR_LENGTH;
            }
            while (countI > 0) {
                int bIndx = (nc - countJ) * kc;
                float[] cTmp = new float[1];
                int countL = kc;
                for (; countL > 0; countL--) {
                    cTmp[0] += sa[aIndx] * sb[bIndx];
                    aIndx += 1;
                    bIndx += 1;
                }
                sc[cOffset + cRow + cCol * ldc] += alpha * cTmp[0];

                cRow += 1;
                countI -= 1;
            }
            cCol += 1;
        }
    }

    private static void kernelOperation16x4Main(int mc, int nc, int kc, float alpha, float[] sa, float[] sb,
        int bOffset, float[] sc, int ldc, int cOffset, int csRow, int csCol) {
        FloatVector alphaVec = FloatVector.broadcast(SSPECIES, alpha);
        int countJ = nc / SGEMM_UNROLL_N;
        int cCol = csCol;
        for (; countJ > 0; countJ--) {
            int cRow = csRow;
            int aIndx = 0;
            int countI = mc / SGEMM_UNROLL_M;
            for (; countI > 0; countI--) {
                FloatVector c00 = FloatVector.zero(SSPECIES);
                FloatVector c10 = FloatVector.zero(SSPECIES);
                FloatVector c20 = FloatVector.zero(SSPECIES);
                FloatVector c30 = FloatVector.zero(SSPECIES);
                FloatVector c01 = FloatVector.zero(SSPECIES);
                FloatVector c11 = FloatVector.zero(SSPECIES);
                FloatVector c21 = FloatVector.zero(SSPECIES);
                FloatVector c31 = FloatVector.zero(SSPECIES);
                FloatVector c02 = FloatVector.zero(SSPECIES);
                FloatVector c12 = FloatVector.zero(SSPECIES);
                FloatVector c22 = FloatVector.zero(SSPECIES);
                FloatVector c32 = FloatVector.zero(SSPECIES);
                FloatVector c03 = FloatVector.zero(SSPECIES);
                FloatVector c13 = FloatVector.zero(SSPECIES);
                FloatVector c23 = FloatVector.zero(SSPECIES);
                FloatVector c33 = FloatVector.zero(SSPECIES);
                int bIndx = (nc / SGEMM_UNROLL_N - countJ) * SGEMM_UNROLL_N * kc;
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);

                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);
                    FloatVector a1 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH);
                    FloatVector a2 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH2);
                    FloatVector a3 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH3);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c20 = a2.fma(b0, c20);
                    c30 = a3.fma(b0, c30);

                    FloatVector b1 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 1]);
                    c01 = a0.fma(b1, c01);
                    c11 = a1.fma(b1, c11);
                    c21 = a2.fma(b1, c21);
                    c31 = a3.fma(b1, c31);

                    FloatVector b2 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 2]);
                    c02 = a0.fma(b2, c02);
                    c12 = a1.fma(b2, c12);
                    c22 = a2.fma(b2, c22);
                    c32 = a3.fma(b2, c32);

                    FloatVector b3 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 3]);
                    c03 = a0.fma(b3, c03);
                    c13 = a1.fma(b3, c13);
                    c23 = a2.fma(b3, c23);
                    c33 = a3.fma(b3, c33);
                    aIndx += SGEMM_UNROLL_M;
                    bIndx += SGEMM_UNROLL_N;
                }
                FloatVector cOri00 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc);
                FloatVector cOri10 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc);
                FloatVector cOri20 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH2
                    + cCol * ldc);
                FloatVector cOri30 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH3
                    + cCol * ldc);

                cOri00 = alphaVec.fma(c00, cOri00);
                cOri10 = alphaVec.fma(c10, cOri10);
                cOri20 = alphaVec.fma(c20, cOri20);
                cOri30 = alphaVec.fma(c30, cOri30);

                cOri00.intoArray(sc, cOffset + cRow + cCol * ldc);
                cOri10.intoArray(sc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);
                cOri20.intoArray(sc, cOffset + cRow + VECTOR_LENGTH2 + cCol * ldc);
                cOri30.intoArray(sc, cOffset + cRow + VECTOR_LENGTH3 + cCol * ldc);

                FloatVector cOri01 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 1) * ldc);
                FloatVector cOri11 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 1) * ldc);
                FloatVector cOri21 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 1) * ldc);
                FloatVector cOri31 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 1) * ldc);

                cOri01 = alphaVec.fma(c01, cOri01);
                cOri11 = alphaVec.fma(c11, cOri11);
                cOri21 = alphaVec.fma(c21, cOri21);
                cOri31 = alphaVec.fma(c31, cOri31);

                cOri01.intoArray(sc, cOffset + cRow + (cCol + 1) * ldc);
                cOri11.intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 1) * ldc);
                cOri21.intoArray(sc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 1) * ldc);
                cOri31.intoArray(sc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 1) * ldc);

                FloatVector cOri02 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 2) * ldc);
                FloatVector cOri12 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 2) * ldc);
                FloatVector cOri22 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 2) * ldc);
                FloatVector cOri32 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 2) * ldc);

                cOri02 = alphaVec.fma(c02, cOri02);
                cOri12 = alphaVec.fma(c12, cOri12);
                cOri22 = alphaVec.fma(c22, cOri22);
                cOri32 = alphaVec.fma(c32, cOri32);

                cOri02.intoArray(sc, cOffset + cRow + (cCol + 2) * ldc);
                cOri12.intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 2) * ldc);
                cOri22.intoArray(sc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 2) * ldc);
                cOri32.intoArray(sc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 2) * ldc);

                FloatVector cOri03 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 3) * ldc);
                FloatVector cOri13 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 3) * ldc);
                FloatVector cOri23 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 3) * ldc);
                FloatVector cOri33 = FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 3) * ldc);

                cOri03 = alphaVec.fma(c03, cOri03);
                cOri13 = alphaVec.fma(c13, cOri13);
                cOri23 = alphaVec.fma(c23, cOri23);
                cOri33 = alphaVec.fma(c33, cOri33);

                cOri03.intoArray(sc, cOffset + cRow + (cCol + 3) * ldc);
                cOri13.intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 3) * ldc);
                cOri23.intoArray(sc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 3) * ldc);
                cOri33.intoArray(sc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 3) * ldc);

                cRow += SGEMM_UNROLL_M;
            }
            countI = mc % SGEMM_UNROLL_M;
            if (countI >= VECTOR_LENGTH2) {
                int bIndx = (nc / SGEMM_UNROLL_N - countJ) * SGEMM_UNROLL_N * kc;
                FloatVector c00 = FloatVector.zero(SSPECIES);
                FloatVector c10 = FloatVector.zero(SSPECIES);
                FloatVector c01 = FloatVector.zero(SSPECIES);
                FloatVector c11 = FloatVector.zero(SSPECIES);
                FloatVector c02 = FloatVector.zero(SSPECIES);
                FloatVector c12 = FloatVector.zero(SSPECIES);
                FloatVector c03 = FloatVector.zero(SSPECIES);
                FloatVector c13 = FloatVector.zero(SSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);
                    FloatVector a1 = FloatVector.fromArray(SSPECIES, sa, aIndx + VECTOR_LENGTH);

                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);
                    FloatVector b1 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 1]);
                    FloatVector b2 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 2]);
                    FloatVector b3 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 3]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c01 = a0.fma(b1, c01);
                    c11 = a1.fma(b1, c11);

                    c02 = a0.fma(b2, c02);
                    c12 = a1.fma(b2, c12);
                    c03 = a0.fma(b3, c03);
                    c13 = a1.fma(b3, c13);

                    aIndx += VECTOR_LENGTH2;
                    bIndx += SGEMM_UNROLL_N;
                }
                alphaVec.fma(c00, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc)).intoArray(sc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);

                alphaVec.fma(c01, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 1) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 1) * ldc);
                alphaVec.fma(c11, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 1) * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 1) * ldc);

                alphaVec.fma(c02, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 2) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 2) * ldc);
                alphaVec.fma(c12, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 2) * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 2) * ldc);

                alphaVec.fma(c03, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 3) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 3) * ldc);
                alphaVec.fma(c13, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 3) * ldc)).intoArray(sc, cOffset + cRow + VECTOR_LENGTH + (cCol + 3) * ldc);

                cRow += VECTOR_LENGTH2;
                countI -= VECTOR_LENGTH2;
            }
            if (countI >= VECTOR_LENGTH) {
                int bIndx = (nc / SGEMM_UNROLL_N - countJ) * SGEMM_UNROLL_N * kc;
                FloatVector c00 = FloatVector.zero(SSPECIES);
                FloatVector c01 = FloatVector.zero(SSPECIES);
                FloatVector c02 = FloatVector.zero(SSPECIES);
                FloatVector c03 = FloatVector.zero(SSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    FloatVector a0 = FloatVector.fromArray(SSPECIES, sa, aIndx);

                    FloatVector b0 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx]);
                    FloatVector b1 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 1]);
                    FloatVector b2 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 2]);
                    FloatVector b3 = FloatVector.broadcast(SSPECIES, sb[bOffset + bIndx + 3]);

                    c00 = a0.fma(b0, c00);
                    c01 = a0.fma(b1, c01);
                    c02 = a0.fma(b2, c02);
                    c03 = a0.fma(b3, c03);

                    aIndx += VECTOR_LENGTH;
                    bIndx += SGEMM_UNROLL_N;
                }
                alphaVec.fma(c00, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + cCol * ldc)).intoArray(sc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c01, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 1) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 1) * ldc);
                alphaVec.fma(c02, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 2) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 2) * ldc);
                alphaVec.fma(c03, FloatVector.fromArray(SSPECIES, sc, cOffset + cRow + (cCol + 3) * ldc)).intoArray(sc,
                    cOffset + cRow + (cCol + 3) * ldc);

                cRow += VECTOR_LENGTH;
                countI -= VECTOR_LENGTH;
            }
            while (countI > 0) {
                int bIndx = (nc / SGEMM_UNROLL_N - countJ) * SGEMM_UNROLL_N * kc;
                float[] cTmp = new float[SGEMM_UNROLL_N];
                int countL = kc;
                for (; countL > 0; countL--) {
                    cTmp[0] += sa[aIndx] * sb[bOffset + bIndx];
                    cTmp[1] += sa[aIndx] * sb[bOffset + bIndx + 1];
                    cTmp[2] += sa[aIndx] * sb[bOffset + bIndx + 2];
                    cTmp[3] += sa[aIndx] * sb[bOffset + bIndx + 3];
                    aIndx += 1;
                    bIndx += SGEMM_UNROLL_N;
                }
                sc[cOffset + cRow + cCol * ldc] += alpha * cTmp[0];
                sc[cOffset + cRow + (cCol + 1) * ldc] += alpha * cTmp[1];
                sc[cOffset + cRow + (cCol + 2) * ldc] += alpha * cTmp[2];
                sc[cOffset + cRow + (cCol + 3) * ldc] += alpha * cTmp[3];

                cRow += 1;
                countI -= 1;
            }
            cCol += SGEMM_UNROLL_N;
        }
    }

    /**
     * onCopy is used for normally packing matrix in the right.
     */
    protected static void onCopy(int sizeM, int sizeN, float[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        float[] dst, int dstOffset) {
        int col = 0;
        int colPackSize = SGEMM_UNROLL_N;
        int dstIndex = 0;
        for (; col < sizeN - sizeN % colPackSize; col += colPackSize) {
            int row = 0;
            for (; row < sizeM; row += 1) {
                dst[dstOffset + dstIndex] = src[(srcRow + row) + (srcCol + col) * srcLd + srcOffset];
                dst[dstOffset + dstIndex + 1] = src[(srcRow + row) + (srcCol + (col + 1)) * srcLd + srcOffset];
                dst[dstOffset + dstIndex + 2] = src[(srcRow + row) + (srcCol + (col + 2)) * srcLd + srcOffset];
                dst[dstOffset + dstIndex + 3] = src[(srcRow + row) + (srcCol + (col + 3)) * srcLd + srcOffset];
                dstIndex += colPackSize;
            }
        }
        for (; col < sizeN; col += 1) {
            int row = 0;
            for (; row < sizeM; row += 1) {
                dst[dstOffset + dstIndex] = src[(srcRow + row) + (srcCol + col) * srcLd + srcOffset];
                dstIndex += 1;
            }
        }
    }

    /**
     * itCopy is used for transpose packing matrix in the left.
     */
    protected static void itCopy(int sizeM, int sizeN, float[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        float[] dst, int dstOffset) {
        int row = 0;
        int dstIndex = 0;
        int[] vectorLengthList = {VECTOR_LENGTH4, VECTOR_LENGTH2, VECTOR_LENGTH, 1};
        for (int vectorLen : vectorLengthList) {
            while (row + vectorLen <= sizeM) {
                int col = 0;
                for (; col < sizeN; col++) {
                    System.arraycopy(src, srcOffset + (srcRow + row) + (srcCol + col) * srcLd, dst,
                        dstOffset + dstIndex, vectorLen);
                    dstIndex += vectorLen;
                }
                row += vectorLen;
            }
        }
    }
}
