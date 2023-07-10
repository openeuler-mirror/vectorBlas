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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class DblasLevel3 {
    private static final VectorSpecies<Double> DSPECIES = DoubleVector.SPECIES_MAX;
    protected static final int DGEMM_P = 256; // Blocking size for m direction.
    protected static final int DGEMM_Q = 240; // Blocking size for k direction.
    protected static final int DGEMM_R = 8192; // Blocking size for n direction.
    protected static final int VECTOR_LENGTH = DSPECIES.length();
    protected static final int VECTOR_LENGTH2 = 2 * VECTOR_LENGTH; // 2 times vector length
    protected static final int VECTOR_LENGTH3 = 3 * VECTOR_LENGTH; // 3 times vector length
    protected static final int VECTOR_LENGTH4 = 4 * VECTOR_LENGTH; // 4 times vector length
    protected static final int DGEMM_UNROLL_M = 4 * VECTOR_LENGTH; // Kernel size for m is 4 * DSPECIES.length().
    protected static final int DGEMM_UNROLL_N = 4; // Kernel size for n direction is 4.

    protected static void betaMulC(int sizeM, int sizeN, double beta, double[] dc, int cOffset, int ldc) {
        DoubleVector betav = DoubleVector.broadcast(DSPECIES, beta);
        for (int col = 0; col < sizeN; col++) {
            int row = 0;
            for (; row < sizeM - VECTOR_LENGTH; row += VECTOR_LENGTH) {
                DoubleVector cv = DoubleVector.fromArray(DSPECIES, dc, row + col * ldc + cOffset);
                cv.mul(betav).intoArray(dc, row + col * ldc + cOffset);
            }
            for (; row < sizeM; row++) {
                dc[row + col * ldc + cOffset] *= beta;
            }
        }
    }

    protected static void kernelOperation8x4(int mc, int nc, int kc, double alpha, double[] da, double[] db,
        int bOffset, double[] dc, int ldc, int cOffset, int csRow, int csCol) {
        kernelOperation8x4Main(mc, nc, kc, alpha, da, db, bOffset, dc, ldc, cOffset, csRow, csCol);
        kernelOperation8x4NBorder(mc, nc, kc, alpha, da, db, bOffset, dc, ldc, cOffset, csRow, csCol);
    }

    private static void kernelOperation8x4NBorder(int mc, int nc, int kc, double alpha, double[] da, double[] db,
        int bOffset, double[] dc, int ldc, int cOffset, int csRow, int csCol) {
        DoubleVector alphaVec = DoubleVector.broadcast(DSPECIES, alpha);
        int cCol = csCol + (nc / DGEMM_UNROLL_N) * DGEMM_UNROLL_N;
        int countJ = nc % DGEMM_UNROLL_N;
        for (; countJ > 0; countJ--) {
            int cRow = csRow;
            int aIndx = 0;
            int countI = mc / DGEMM_UNROLL_M;
            for (; countI > 0; countI--) {
                int bIndx = (nc - countJ) * kc;
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                DoubleVector c10 = DoubleVector.zero(DSPECIES);
                DoubleVector c20 = DoubleVector.zero(DSPECIES);
                DoubleVector c30 = DoubleVector.zero(DSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);
                    DoubleVector a1 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH);
                    DoubleVector a2 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH2);
                    DoubleVector a3 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH3);

                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c20 = a2.fma(b0, c20);
                    c30 = a3.fma(b0, c30);

                    aIndx += DGEMM_UNROLL_M;
                    bIndx += 1;
                }
                alphaVec.fma(c00, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc)).intoArray(dc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);
                alphaVec.fma(c20, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH2
                    + cCol * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH2 + cCol * ldc);
                alphaVec.fma(c30, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH3
                    + cCol * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH3 + cCol * ldc);

                cRow += DGEMM_UNROLL_M;
            }
            countI = mc % DGEMM_UNROLL_M;
            if (countI >= VECTOR_LENGTH2) {
                int bIndx = (nc - countJ) * kc;
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                DoubleVector c10 = DoubleVector.zero(DSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);
                    DoubleVector a1 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH);
                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);

                    aIndx += VECTOR_LENGTH2;
                    bIndx += 1;
                }
                alphaVec.fma(c00, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc)).intoArray(dc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);

                cRow += VECTOR_LENGTH2;
                countI -= VECTOR_LENGTH2;
            }
            if (countI >= VECTOR_LENGTH) {
                int bIndx = (nc - countJ) * kc;
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);
                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);
                    c00 = a0.fma(b0, c00);
                    aIndx += VECTOR_LENGTH;
                    bIndx += 1;
                }
                alphaVec.fma(c00, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc)).intoArray(dc,
                    cOffset + cRow + cCol * ldc);

                cRow += VECTOR_LENGTH;
                countI -= VECTOR_LENGTH;
            }
            while (countI > 0) {
                int bIndx = (nc - countJ) * kc;
                double[] cTmp = new double[1];
                int countL = kc;
                for (; countL > 0; countL--) {
                    cTmp[0] += da[aIndx] * db[bIndx];
                    aIndx += 1;
                    bIndx += 1;
                }
                dc[cOffset + cRow + cCol * ldc] += alpha * cTmp[0];

                cRow += 1;
                countI -= 1;
            }
            cCol += 1;
        }
    }

    private static void kernelOperation8x4Main(int mc, int nc, int kc, double alpha, double[] da, double[] db,
        int bOffset, double[] dc, int ldc, int cOffset, int csRow, int csCol) {
        DoubleVector alphaVec = DoubleVector.broadcast(DSPECIES, alpha);
        int countJ = nc / DGEMM_UNROLL_N;
        int cCol = csCol;
        for (; countJ > 0; countJ--) {
            int cRow = csRow;
            int aIndx = 0;
            int countI = mc / DGEMM_UNROLL_M;
            for (; countI > 0; countI--) {
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                DoubleVector c10 = DoubleVector.zero(DSPECIES);
                DoubleVector c20 = DoubleVector.zero(DSPECIES);
                DoubleVector c30 = DoubleVector.zero(DSPECIES);
                DoubleVector c01 = DoubleVector.zero(DSPECIES);
                DoubleVector c11 = DoubleVector.zero(DSPECIES);
                DoubleVector c21 = DoubleVector.zero(DSPECIES);
                DoubleVector c31 = DoubleVector.zero(DSPECIES);
                DoubleVector c02 = DoubleVector.zero(DSPECIES);
                DoubleVector c12 = DoubleVector.zero(DSPECIES);
                DoubleVector c22 = DoubleVector.zero(DSPECIES);
                DoubleVector c32 = DoubleVector.zero(DSPECIES);
                DoubleVector c03 = DoubleVector.zero(DSPECIES);
                DoubleVector c13 = DoubleVector.zero(DSPECIES);
                DoubleVector c23 = DoubleVector.zero(DSPECIES);
                DoubleVector c33 = DoubleVector.zero(DSPECIES);
                int bIndx = (nc / DGEMM_UNROLL_N - countJ) * DGEMM_UNROLL_N * kc;

                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);
                    DoubleVector a1 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH);
                    DoubleVector a2 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH2);
                    DoubleVector a3 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH3);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c20 = a2.fma(b0, c20);
                    c30 = a3.fma(b0, c30);

                    DoubleVector b1 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 1]);
                    c01 = a0.fma(b1, c01);
                    c11 = a1.fma(b1, c11);
                    c21 = a2.fma(b1, c21);
                    c31 = a3.fma(b1, c31);

                    DoubleVector b2 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 2]);
                    c02 = a0.fma(b2, c02);
                    c12 = a1.fma(b2, c12);
                    c22 = a2.fma(b2, c22);
                    c32 = a3.fma(b2, c32);

                    DoubleVector b3 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 3]);
                    c03 = a0.fma(b3, c03);
                    c13 = a1.fma(b3, c13);
                    c23 = a2.fma(b3, c23);
                    c33 = a3.fma(b3, c33);
                    aIndx += DGEMM_UNROLL_M;
                    bIndx += DGEMM_UNROLL_N;
                }
                DoubleVector cOri00 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc);
                DoubleVector cOri10 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc);
                DoubleVector cOri20 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH2
                    + cCol * ldc);
                DoubleVector cOri30 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH3
                    + cCol * ldc);

                cOri00 = alphaVec.fma(c00, cOri00);
                cOri10 = alphaVec.fma(c10, cOri10);
                cOri20 = alphaVec.fma(c20, cOri20);
                cOri30 = alphaVec.fma(c30, cOri30);

                cOri00.intoArray(dc, cOffset + cRow + cCol * ldc);
                cOri10.intoArray(dc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);
                cOri20.intoArray(dc, cOffset + cRow + VECTOR_LENGTH2 + cCol * ldc);
                cOri30.intoArray(dc, cOffset + cRow + VECTOR_LENGTH3 + cCol * ldc);

                DoubleVector cOri01 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 1) * ldc);
                DoubleVector cOri11 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 1) * ldc);
                DoubleVector cOri21 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 1) * ldc);
                DoubleVector cOri31 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 1) * ldc);

                cOri01 = alphaVec.fma(c01, cOri01);
                cOri11 = alphaVec.fma(c11, cOri11);
                cOri21 = alphaVec.fma(c21, cOri21);
                cOri31 = alphaVec.fma(c31, cOri31);

                cOri01.intoArray(dc, cOffset + cRow + (cCol + 1) * ldc);
                cOri11.intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 1) * ldc);
                cOri21.intoArray(dc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 1) * ldc);
                cOri31.intoArray(dc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 1) * ldc);

                DoubleVector cOri02 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 2) * ldc);
                DoubleVector cOri12 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 2) * ldc);
                DoubleVector cOri22 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 2) * ldc);
                DoubleVector cOri32 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 2) * ldc);

                cOri02 = alphaVec.fma(c02, cOri02);
                cOri12 = alphaVec.fma(c12, cOri12);
                cOri22 = alphaVec.fma(c22, cOri22);
                cOri32 = alphaVec.fma(c32, cOri32);

                cOri02.intoArray(dc, cOffset + cRow + (cCol + 2) * ldc);
                cOri12.intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 2) * ldc);
                cOri22.intoArray(dc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 2) * ldc);
                cOri32.intoArray(dc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 2) * ldc);

                DoubleVector cOri03 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 3) * ldc);
                DoubleVector cOri13 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 3) * ldc);
                DoubleVector cOri23 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH2
                    + (cCol + 3) * ldc);
                DoubleVector cOri33 = DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH3
                    + (cCol + 3) * ldc);

                cOri03 = alphaVec.fma(c03, cOri03);
                cOri13 = alphaVec.fma(c13, cOri13);
                cOri23 = alphaVec.fma(c23, cOri23);
                cOri33 = alphaVec.fma(c33, cOri33);

                cOri03.intoArray(dc, cOffset + cRow + (cCol + 3) * ldc);
                cOri13.intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 3) * ldc);
                cOri23.intoArray(dc, cOffset + cRow + VECTOR_LENGTH2 + (cCol + 3) * ldc);
                cOri33.intoArray(dc, cOffset + cRow + VECTOR_LENGTH3 + (cCol + 3) * ldc);

                cRow += DGEMM_UNROLL_M;
            }
            countI = mc % DGEMM_UNROLL_M;
            if (countI >= VECTOR_LENGTH2) {
                int bIndx = (nc / DGEMM_UNROLL_N - countJ) * DGEMM_UNROLL_N * kc;
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                DoubleVector c01 = DoubleVector.zero(DSPECIES);
                DoubleVector c02 = DoubleVector.zero(DSPECIES);
                DoubleVector c03 = DoubleVector.zero(DSPECIES);
                DoubleVector c10 = DoubleVector.zero(DSPECIES);
                DoubleVector c11 = DoubleVector.zero(DSPECIES);
                DoubleVector c12 = DoubleVector.zero(DSPECIES);
                DoubleVector c13 = DoubleVector.zero(DSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);
                    DoubleVector a1 = DoubleVector.fromArray(DSPECIES, da, aIndx + VECTOR_LENGTH);

                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);
                    DoubleVector b1 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 1]);
                    DoubleVector b2 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 2]);
                    DoubleVector b3 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 3]);

                    c00 = a0.fma(b0, c00);
                    c10 = a1.fma(b0, c10);
                    c01 = a0.fma(b1, c01);
                    c11 = a1.fma(b1, c11);

                    c02 = a0.fma(b2, c02);
                    c12 = a1.fma(b2, c12);
                    c03 = a0.fma(b3, c03);
                    c13 = a1.fma(b3, c13);

                    aIndx += VECTOR_LENGTH2;
                    bIndx += DGEMM_UNROLL_N;
                }
                alphaVec.fma(c00, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc)).intoArray(dc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c10, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + cCol * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + cCol * ldc);

                alphaVec.fma(c01, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 1) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 1) * ldc);
                alphaVec.fma(c11, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 1) * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 1) * ldc);

                alphaVec.fma(c02, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 2) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 2) * ldc);
                alphaVec.fma(c12, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 2) * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 2) * ldc);

                alphaVec.fma(c03, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 3) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 3) * ldc);
                alphaVec.fma(c13, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + VECTOR_LENGTH
                    + (cCol + 3) * ldc)).intoArray(dc, cOffset + cRow + VECTOR_LENGTH + (cCol + 3) * ldc);

                cRow += VECTOR_LENGTH2;
                countI -= VECTOR_LENGTH2;
            }
            if (countI >= VECTOR_LENGTH) {
                int bIndx = (nc / DGEMM_UNROLL_N - countJ) * DGEMM_UNROLL_N * kc;
                DoubleVector c00 = DoubleVector.zero(DSPECIES);
                DoubleVector c01 = DoubleVector.zero(DSPECIES);
                DoubleVector c02 = DoubleVector.zero(DSPECIES);
                DoubleVector c03 = DoubleVector.zero(DSPECIES);
                int countL = kc;
                for (; countL > 0; countL--) {
                    DoubleVector a0 = DoubleVector.fromArray(DSPECIES, da, aIndx);

                    DoubleVector b0 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx]);
                    DoubleVector b1 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 1]);
                    DoubleVector b2 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 2]);
                    DoubleVector b3 = DoubleVector.broadcast(DSPECIES, db[bOffset + bIndx + 3]);

                    c00 = a0.fma(b0, c00);
                    c01 = a0.fma(b1, c01);
                    c02 = a0.fma(b2, c02);
                    c03 = a0.fma(b3, c03);

                    aIndx += VECTOR_LENGTH;
                    bIndx += DGEMM_UNROLL_N;
                }
                alphaVec.fma(c00, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + cCol * ldc)).intoArray(dc,
                    cOffset + cRow + cCol * ldc);
                alphaVec.fma(c01, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 1) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 1) * ldc);
                alphaVec.fma(c02, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 2) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 2) * ldc);
                alphaVec.fma(c03, DoubleVector.fromArray(DSPECIES, dc, cOffset + cRow + (cCol + 3) * ldc)).intoArray(dc,
                    cOffset + cRow + (cCol + 3) * ldc);

                cRow += VECTOR_LENGTH;
                countI -= VECTOR_LENGTH;
            }
            while (countI > 0) {
                int bIndx = (nc / DGEMM_UNROLL_N - countJ) * DGEMM_UNROLL_N * kc;
                double[] cTmp = new double[DGEMM_UNROLL_N];
                int countL = kc;
                for (; countL > 0; countL--) {
                    cTmp[0] += da[aIndx] * db[bOffset + bIndx];
                    cTmp[1] += da[aIndx] * db[bOffset + bIndx + 1];
                    cTmp[2] += da[aIndx] * db[bOffset + bIndx + 2];
                    cTmp[3] += da[aIndx] * db[bOffset + bIndx + 3];
                    aIndx += 1;
                    bIndx += DGEMM_UNROLL_N;
                }
                dc[cOffset + cRow + cCol * ldc] += alpha * cTmp[0];
                dc[cOffset + cRow + (cCol + 1) * ldc] += alpha * cTmp[1];
                dc[cOffset + cRow + (cCol + 2) * ldc] += alpha * cTmp[2];
                dc[cOffset + cRow + (cCol + 3) * ldc] += alpha * cTmp[3];

                cRow += 1;
                countI -= 1;
            }
            cCol += DGEMM_UNROLL_N;
        }
    }

    /**
     * onCopy is used for normally packing matrix in the right.
     * For example, when DGEMM_UNROLL_N = 4,
     *       before packing                 after packing
     *        1  6  11 16 21                1  2  3  4  21
     *        2  7  12 17 22      --->      5  6  7  8  22
     *        3  8  13 18 23                9  10 11 12 23
     *        4  9  14 19 24                13 14 15 16 24
     *        5  10 15 20 25                17 18 19 20 25
     */
    protected static void onCopy(int sizeM, int sizeN, double[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        double[] dst, int dstOffset) {
        int col = 0;
        int colPackSize = DGEMM_UNROLL_N;
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
     * For example, when DGEMM_UNROLL_M = 4,
     *        before packing                 after packing
     *        1  6  11 16 21                 1  5  9  13 17
     *        2  7  12 17 22       --->      2  6  10 14 18
     *        3  8  13 18 23                 3  7  11 15 19
     *        4  9  14 19 24                 4  8  12 16 20
     *        5  10 15 20 25                 21 22 23 24 25
     */
    protected static void itCopy(int sizeM, int sizeN, double[] src, int srcRow, int srcCol, int srcOffset, int srcLd,
        double[] dst, int dstOffset) {
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
