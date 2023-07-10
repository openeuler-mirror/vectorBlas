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

package com.huawei.vectorblas.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Locale;
import java.util.Random;

public class BlasUtils {
    private static final Logger LOG = LoggerFactory.getLogger(BlasUtils.class);
    private static Random rand = new Random(0);

    public static void checkParameter(String name, int index, boolean isValid) {
        if (!isValid) {
            String msg = String.format(Locale.ROOT,
                    "** On entry to %s parameter number %d had an illegal value", name, index);
            throw new IllegalArgumentException(msg);
        }
    }

    public static void checkBlasArray(String arrName, int offset, int index, int length) {
        try {
            checkBound(index + offset, length);
            checkBound(offset, length);
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                "Index " + index + " of array " + arrName + " out of bounds for length: " + length);
        }
    }

    public static void checkBound(int index, int length) {
        if (index < 0 || index >= length) {
            throw new ArrayIndexOutOfBoundsException();
        }
    }

    public static boolean isZero(double val) {
        return Double.compare(val, 0.0d) == 0 || Double.compare(val, -0.0d) == 0;
    }

    public static boolean isZero(float val) {
        return Float.compare(val, 0.0f) == 0 || Float.compare(val, -0.0f) == 0;
    }

    /**
     * Get double precision machine epsilon.
     */
    public static double getEpsd() {
        double eps;
        double half = 0.5d;
        double maxVal;
        double f1 = 0.5d;
        do {
            eps = f1;
            f1 *= half;
            maxVal = 1.0d + f1;
        } while (Double.compare(maxVal, 1.0d) != 0);
        return eps;
    }

    /**
     * Get single precision machine epsilon.
     */
    public static float getEpsf() {
        float eps;
        float half = 0.5f;
        float maxVal;
        float f1 = 0.5f;
        do {
            eps = f1;
            f1 *= half;
            maxVal = 1.0f + f1;
        } while (Float.compare(maxVal, 1.0f) != 0);
        return eps;
    }

    /**
     * Generate general matrix of double precision.
     */
    public static void gegen(int sizeM, int sizeN, double[] da, int aOffset, int lda) {
        try {
            int idx = (lda >= 0 ? 0 : (sizeN - 1) * -lda) + aOffset;
            for (int j = 0; j < sizeN; j++) {
                for (int i = 0; i < sizeM; i++) {
                    da[idx + i] = rand.nextDouble() - 0.5d;
                }
                idx += lda;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
    }

    /**
     * Generate general matrix of single precision.
     */
    public static void gegen(int sizeM, int sizeN, float[] sa, int aOffset, int lda) {
        try {
            int idx = (lda >= 0 ? 0 : (sizeN - 1) * -lda) + aOffset;
            for (int j = 0; j < sizeN; j++) {
                for (int i = 0; i < sizeM; i++) {
                    sa[idx + i] = rand.nextFloat() - 0.5f;
                }
                idx += lda;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
    }

    /**
     * Calculates the infinity norm of single precision vector.
     */
    public static float getInfnrm(int sizeN, float[] sx, int xOffset, int incX) {
        int idx = (incX >= 0 ? 0 : (sizeN - 1) * -incX) + xOffset;
        float max = 0.0f;
        try {
            for (int i = 0; i < sizeN; i++, idx += incX) {
                max = Math.max(Math.abs(sx[idx]), max);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        return max;
    }

    /**
     * Calculates the infinity norm of double precision vector.
     */
    public static double getInfnrm(int sizeN, double[] dx, int xOffset, int incX) {
        int idx = (incX >= 0 ? 0 : (sizeN - 1) * -incX) + xOffset;
        double max = 0.0d;
        try {
            for (int i = 0; i < sizeN; i++, idx += incX) {
                max = Math.max(Math.abs(dx[idx]), max);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        return max;
    }

    /**
     * Calculates the difference between two double precision vectors.
     */
    public static void getVdiff(int sizeN, double[] dx, int xOffset, int incX, double[] dy, int yOffset, int incY,
        double[] dz, int zOffset, int incZ) {
        int xIdx = (incX >= 0 ? 0 : (sizeN - 1) * -incX) + xOffset;
        int yIdx = (incY >= 0 ? 0 : (sizeN - 1) * -incY) + yOffset;
        int zIdx = (incZ >= 0 ? 0 : (sizeN - 1) * -incZ) + zOffset;
        for (int i = sizeN; i > 0; i--, xIdx += incX, yIdx += incY, zIdx += incZ) {
            dz[zIdx] = dx[xIdx] - dy[yIdx];
        }
    }

    /**
     * Calculates the difference between two single precision vectors.
     */
    public static void getVdiff(int sizeN, float[] sx, int xOffset, int incX, float[] sy, int yOffset, int incY,
        float[] sz, int zOffset, int incZ) {
        int xIdx = (incX >= 0 ? 0 : (sizeN - 1) * -incX) + xOffset;
        int yIdx = (incY >= 0 ? 0 : (sizeN - 1) * -incY) + yOffset;
        int zIdx = (incZ >= 0 ? 0 : (sizeN - 1) * -incZ) + zOffset;
        for (int i = sizeN; i > 0; i--, xIdx += incX, yIdx += incY, zIdx += incZ) {
            sz[zIdx] = sx[xIdx] - sy[yIdx];
        }
    }

    /**
     * Calculates the 1-norm of a general rectangular matrix of double precision.
     */
    public static double getGenrm1(int sizeM, int sizeN, double[] da, int aOffset, int lda) {
        double max = 0.0d;
        int offset = aOffset;
        try {
            for (int j = 0; j < sizeN; j++) {
                double t0 = org.netlib.blas.Dasum.dasum(sizeM, da, offset, 1);
                max = Math.max(t0, max);
                offset += lda;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        return max;
    }

    /**
     * Calculates the 1-norm of a general rectangular matrix of single precision.
     */
    public static float getGenrm1(int sizeM, int sizeN, float[] sa, int aOffset, int lda) {
        float max = 0.0f;
        int offset = aOffset;
        try {
            for (int j = 0; j < sizeN; j++) {
                float t0 = org.netlib.blas.Sasum.sasum(sizeM, sa, offset, 1);
                max = Math.max(t0, max);
                offset += lda;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        return max;
    }

    /**
     * Calculates the 1-norm of (A-B) matrix of double precision.
     */
    public static double getGediffnrm1(int sizeM, int sizeN, double[] da, int aOffset, int lda,
        double[] db, int bOffset, int ldb) {
        double max = 0.0d;
        int offset1 = aOffset;
        int offset2 = bOffset;
        for (int j = 0; j < sizeN; j++) {
            double t0 = 0.0d;
            for (int i = 0; i < sizeM; i++) {
                t0 += Math.abs(da[offset1] - db[offset2]);
            }
            max = Math.max(t0, max);
            offset1 += lda;
            offset2 += ldb;
        }
        return max;
    }

    /**
     * Calculates the 1-norm of (A-B) matrix of single precision.
     */
    public static float getGediffnrm1(int sizeM, int sizeN, float[] sa, int aOffset, int lda,
        float[] sb, int bOffset, int ldb) {
        float max = 0.0f;
        int offset1 = aOffset;
        int offset2 = bOffset;
        for (int j = 0; j < sizeN; j++) {
            float t0 = 0.0f;
            for (int i = 0; i < sizeM; i++) {
                t0 += Math.abs(sa[offset1] - sb[offset2]);
            }
            max = Math.max(t0, max);
            offset1 += lda;
            offset2 += ldb;
        }
        return max;
    }

    /**
     * Calculates the norm of a double precision symmetric packed matrix.
     */
    public static double getSpnrm(String uplo, int sizeN, double[] da, int aOffset) {
        if (sizeN <= 0) {
            return 0.0d;
        }
        double[] work = new double[sizeN];
        try {
            if (uplo.equalsIgnoreCase("U")) {
                for (int j = 0, iaij = 0; j < sizeN; j++) {
                    double t0 = 0.0d;
                    for (int i = 0; i < j; i++, iaij++) {
                        work[i] += Math.abs(da[iaij + aOffset]);
                        t0 += Math.abs(da[iaij + aOffset]);
                    }
                    work[j] += Math.abs(da[iaij + aOffset]) + t0;
                    iaij++;
                }
            } else {
                for (int j = 0, iaij = 0; j < sizeN; j++) {
                    double t0 = 0.0d;
                    work[j] = Math.abs(da[iaij + aOffset]);
                    iaij++;
                    for (int i = j + 1; i < sizeN; i++, iaij++) {
                        work[i] += Math.abs(da[iaij + aOffset]);
                        t0 += Math.abs(da[iaij + aOffset]);
                    }
                    work[j] += t0;
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        double max = work[0];
        for (int j = 1; j < sizeN; j++) {
            max = Math.max(max, work[j]);
        }
        return max;
    }

    /**
     * Calculates the norm of a single precision symmetric packed matrix.
     */
    public static float getSpnrm(String uplo, int sizeN, float[] sa, int aOffset) {
        if (sizeN <= 0) {
            return 0.0f;
        }
        float[] work = new float[sizeN];
        try {
            if (uplo.equalsIgnoreCase("U")) {
                for (int j = 0, iaij = 0; j < sizeN; j++) {
                    float t0 = 0.0f;
                    for (int i = 0; i < j; i++, iaij++) {
                        work[i] += Math.abs(sa[iaij + aOffset]);
                        t0 += Math.abs(sa[iaij + aOffset]);
                    }
                    work[j] += Math.abs(sa[iaij + aOffset]) + t0;
                    iaij++;
                }
            } else {
                for (int j = 0, iaij = 0; j < sizeN; j++) {
                    float t0 = 0.0f;
                    work[j] = Math.abs(sa[iaij + aOffset]);
                    iaij++;
                    for (int i = j + 1; i < sizeN; i++, iaij++) {
                        work[i] += Math.abs(sa[iaij + aOffset]);
                        t0 += Math.abs(sa[iaij + aOffset]);
                    }
                    work[j] += t0;
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            LOG.error(e.toString());
        }
        float max = work[0];
        for (int j = 1; j < sizeN; j++) {
            max = Math.max(max, work[j]);
        }
        return max;
    }

    /**
     * Calculates the norm of a upper or lower triangular part of the double precision symmetric matrix.
     */
    public static double getSynrm(String uplo, int sizeN, double[] da, int aOffset, int lda) {
        int ldap12 = lda + 1;
        if (sizeN <= 0) {
            return 0.0d;
        }
        double[] work = new double[sizeN];
        if (uplo.equalsIgnoreCase("U")) {
            for (int j = 0, jaj = 0; j < sizeN; j++, jaj += lda) {
                double t0 = 0.0d;
                int iaij = jaj;
                for (int i = 0; i < j; i++, iaij++) {
                    work[i] += Math.abs(da[iaij + aOffset]);
                    t0 += Math.abs(da[iaij + aOffset]);
                }
                work[j] += Math.abs(da[iaij + aOffset]) + t0;
            }
        } else {
            for (int j = 0, jaj = 0; j < sizeN; j++, jaj += ldap12) {
                double t0 = 0.0d;
                work[j] = Math.abs(da[jaj + aOffset]);
                for (int i = j + 1, iaij = jaj + 1; i < sizeN; i++, iaij++) {
                    work[i] += Math.abs(da[iaij + aOffset]);
                    t0 += Math.abs(da[iaij + aOffset]);
                }
                work[j] += t0;
            }
        }
        double max = work[0];
        for (int j = 1; j < sizeN; j++) {
            max = Math.max(work[j], max);
        }
        return max;
    }

    /**
     * Calculates the norm of a upper or lower triangular part of the single precision symmetric matrix.
     */
    public static float getSynrm(String uplo, int sizeN, float[] sa, int aOffset, int lda) {
        int ldap12 = lda + 1;
        if (sizeN <= 0) {
            return 0.0f;
        }
        float[] work = new float[sizeN];
        if (uplo.equalsIgnoreCase("U")) {
            for (int j = 0, jaj = 0; j < sizeN; j++, jaj += lda) {
                float t0 = 0.0f;
                int iaij = jaj;
                for (int i = 0; i < j; i++, iaij++) {
                    work[i] += Math.abs(sa[iaij + aOffset]);
                    t0 += Math.abs(sa[iaij + aOffset]);
                }
                work[j] += Math.abs(sa[iaij + aOffset]) + t0;
            }
        } else {
            for (int j = 0, jaj = 0; j < sizeN; j++, jaj += ldap12) {
                float t0 = 0.0f;
                work[j] = Math.abs(sa[jaj + aOffset]);
                for (int i = j + 1, iaij = jaj + 1; i < sizeN; i++, iaij++) {
                    work[i] += Math.abs(sa[iaij + aOffset]);
                    t0 += Math.abs(sa[iaij + aOffset]);
                }
                work[j] += t0;
            }
        }
        float max = work[0];
        for (int j = 1; j < sizeN; j++) {
            max = Math.max(work[j], max);
        }
        return max;
    }
}
