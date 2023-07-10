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

package com.huawei.vectorblas;

public class F2jBLAS implements BLAS {
    /**
     * F2jblas dasum fixed version, use long to store (n * incx) to avoid int overflow.
     */
    @Override
    public double dasum(int n, double[] x, int xOffset, int incx) {
        int unrollSize = 6;
        double dasum = 0.0D;
        if (n <= 0 || incx <= 0) {
            return dasum;
        } else {
            int index;
            if (incx == 1) {
                int restm = n % unrollSize;
                if (restm != 0) {
                    index = 1;
                    for (int i = restm; i > 0; --i) {
                        dasum += Math.abs(x[index - 1 + xOffset]);
                        ++index;
                    }
                    if (n < unrollSize) {
                        return dasum;
                    }
                }
                int mp1 = restm + 1;
                index = mp1;

                for (int i = (n - mp1 + unrollSize) / unrollSize; i > 0; --i) {
                    dasum = dasum + Math.abs(x[index - 1 + xOffset]) + Math.abs(x[index + xOffset])
                        + Math.abs(x[index + 1 + xOffset]) + Math.abs(x[index + 2 + xOffset])
                        + Math.abs(x[index + 3 + xOffset]) + Math.abs(x[index + 4 + xOffset]);
                    index += unrollSize;
                }
                return dasum;
            } else {
                long nIncx = (long) n * incx;
                index = 1;
                for (long i = (nIncx - 1 + incx) / incx; i > 0; --i) {
                    dasum += Math.abs(x[index - 1 + xOffset]);
                    index += incx;
                }
                return dasum;
            }
        }
    }

    /**
     * dasum without offset
     */
    @Override
    public double dasum(int n, double[] x, int incx) {
        return dasum(n, x, 0, incx);
    }


    @Override
    public float sasum(int n, float[] x, int incx) {
        return sasum(n, x, 0, incx);
    }

    /**
     * F2jblas sasum fixed version, use long to store (n * incx) to avoid int overflow.
     */
    @Override
    public float sasum(int n, float[] x, int xOffset, int incx) {
        int unrollSize = 6;
        float sasum = 0.0F;
        if (n <= 0 || incx <= 0) {
            return sasum;
        } else {
            int index;
            if (incx == 1) {
                int restm = n % unrollSize;
                if (restm != 0) {
                    index = 1;
                    for (int i = restm; i > 0; --i) {
                        sasum += Math.abs(x[index - 1 + xOffset]);
                        ++index;
                    }
                    if (n < unrollSize) {
                        return sasum;
                    }
                }
                int mp1 = restm + 1;
                index = mp1;

                for (int i = (n - mp1 + unrollSize) / unrollSize; i > 0; --i) {
                    sasum = sasum + Math.abs(x[index - 1 + xOffset]) + Math.abs(x[index + xOffset])
                        + Math.abs(x[index + 1 + xOffset]) + Math.abs(x[index + 2 + xOffset])
                        + Math.abs(x[index + 3 + xOffset]) + Math.abs(x[index + 4 + xOffset]);
                    index += unrollSize;
                }
                return sasum;
            } else {
                long nIncx = (long) n * incx;
                index = 1;
                for (long i = (nIncx - 1 + incx) / incx; i > 0; --i) {
                    sasum += Math.abs(x[index - 1 + xOffset]);
                    index += incx;
                }
                return sasum;
            }
        }
    }

    @Override
    public void daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy) {
        org.netlib.blas.Daxpy.daxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    @Override
    public void daxpy(int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        org.netlib.blas.Daxpy.daxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy) {
        org.netlib.blas.Saxpy.saxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    @Override
    public void saxpy(int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        org.netlib.blas.Saxpy.saxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void dcopy(int n, double[] x, int incx, double[] y, int incy) {
        org.netlib.blas.Dcopy.dcopy(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void dcopy(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dcopy.dcopy(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void scopy(int n, float[] x, int incx, float[] y, int incy) {
        org.netlib.blas.Scopy.scopy(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void scopy(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        org.netlib.blas.Scopy.scopy(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public double ddot(int n, double[] x, int incx, double[] y, int incy) {
        return org.netlib.blas.Ddot.ddot(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public double ddot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        return org.netlib.blas.Ddot.ddot(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public float sdot(int n, float[] x, int incx, float[] y, int incy) {
        return org.netlib.blas.Sdot.sdot(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public float sdot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        return org.netlib.blas.Sdot.sdot(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public double dnrm2(int n, double[] x, int incx) {
        return dnrm2(n, x, 0, incx);
    }

    /**
     * F2jblas dnrm2 fixed version, use long to store (n * incx) to avoid int overflow.
     */
    @Override
    public double dnrm2(int n, double[] x, int xOffset, int incx) {
        double absxi = 0.0D;
        double norm = 0.0D;
        double scale = 0.0D;
        double ssq = 0.0D;
        if (n < 1 || incx < 1) {
            norm = 0.0;
        } else if (n == 1) {
            norm = Math.abs(x[xOffset]);
        } else {
            scale = 0.0;
            ssq = 1.0D;
            int ix = 1;
            for (long i = ((long) n * incx) / incx; i > 0; --i) {
                if (x[ix - 1 + xOffset] != 0.0) {
                    absxi = Math.abs(x[ix - 1 + xOffset]);
                    if (scale < absxi) {
                        ssq = 1.0D + ssq * Math.pow(scale / absxi, (double) 2);
                        scale = absxi;
                    } else {
                        ssq += Math.pow(absxi / scale, (double) 2);
                    }
                }
                ix += incx;
            }
            norm = scale * Math.sqrt(ssq);
        }
        return norm;
    }

    @Override
    public float snrm2(int n, float[] x, int incx) {
        return snrm2(n, x, 0, incx);
    }

    /**
     * F2jblas snrm2 fixed version, use long to store (n * incx) to avoid int overflow.
     */
    @Override
    public float snrm2(int n, float[] x, int xOffset, int incx) {
        float absxi = 0.0F;
        float norm = 0.0F;
        float scale = 0.0F;
        float ssq = 0.0F;
        if (n < 1 || incx < 1) {
            norm = 0.0F;
        } else if (n == 1) {
            norm = Math.abs(x[xOffset]);
        } else {
            scale = 0.0F;
            ssq = 1.0F;
            int ix = 1;
            for (long i = ((long) n * incx) / incx; i > 0; --i) {
                if (x[ix - 1 + xOffset] != 0.0F) {
                    absxi = Math.abs(x[ix - 1 + xOffset]);
                    if (scale < absxi) {
                        ssq = 1.0F + ssq * (float) Math.pow(scale / absxi, 2);
                        scale = absxi;
                    } else {
                        ssq += Math.pow(absxi / scale, 2);
                    }
                }
                ix += incx;
            }
            norm = scale * (float) Math.sqrt(ssq);
        }
        return norm;
    }

    @Override
    public void srot(int n, float[] x, int incx, float[] y, int incy, float c, float s) {
        org.netlib.blas.Srot.srot(n, x, 0, incx, y, 0, incy, c, s);
    }

    @Override
    public void srot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float c, float s) {
        org.netlib.blas.Srot.srot(n, x, xOffset, incx, y, yOffset, incy, c, s);
    }

    @Override
    public void drot(int n, double[] x, int incx, double[] y, int incy, double c, double s) {
        org.netlib.blas.Drot.drot(n, x, 0, incx, y, 0, incy, c, s);
    }

    @Override
    public void drot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double c, double s) {
        org.netlib.blas.Drot.drot(n, x, xOffset, incx, y, yOffset, incy, c, s);
    }

    @Override
    public void srotm(int n, float[] x, int incx, float[] y, int incy, float[] param) {
        srotm(n, x, 0, incx, y, 0, incy, param, 0);
    }

    /**
     * f2jblas srotm fixed version, use long to store (n * incx) to avoid int overflow
     */
    @Override
    public void srotm(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float[] param,
        int paramOffset) {
        float flag = 0.0F;
        float h11 = 0.0F;
        float h12 = 0.0F;
        float h21 = 0.0F;
        float h22 = 0.0F;
        float wi = 0.0F;
        float zi = 0.0F;
        flag = param[paramOffset];
        if (n > 0 && Float.compare(flag, -2.0F) != 0) { // If flag equals -2.0, do nothing and return directly.
            int index;
            if ((incx == incy && incx > 0) ^ true) {
                int xIndex = 1;
                int yIndex = 1;
                if (incx < 0) {
                    xIndex = 1 + (1 - n) * incx;
                }
                if (incy < 0) {
                    yIndex = 1 + (1 - n) * incy;
                }
                if (flag < 0.0) {
                    h11 = param[2 - 1 + paramOffset];
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi * h11 + zi * h12;
                        y[yIndex - 1 + yOffset] = wi * h21 + zi * h22;
                        xIndex += incx;
                        yIndex += incy;
                        ++index;
                    }
                } else if (flag == 0.0) {
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    index = 1;
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi + zi * h12;
                        y[yIndex - 1 + yOffset] = wi * h21 + zi;
                        xIndex += incx;
                        yIndex += incy;
                        ++index;
                    }
                } else {
                    h11 = param[2 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi * h11 + zi;
                        y[yIndex - 1 + yOffset] = -wi + h22 * zi;
                        xIndex += incx;
                        yIndex += incy;
                    }
                }
            } else {
                long nSteps = (long) n * incx;
                if (flag < 0.0) {
                    h11 = param[2 - 1 + paramOffset];
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi * h11 + zi * h12;
                        y[index - 1 + yOffset] = wi * h21 + zi * h22;
                        index += incx;
                    }
                } else if (flag == 0.0) {
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi + zi * h12;
                        y[index - 1 + yOffset] = wi * h21 + zi;
                        index += incx;
                    }
                } else {
                    h11 = param[2 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi * h11 + zi;
                        y[index - 1 + yOffset] = -wi + h22 * zi;
                        index += incx;
                    }
                }
            }
        }
    }

    @Override
    public void drotm(int n, double[] x, int incx, double[] y, int incy, double[] param) {
        drotm(n, x, 0, incx, y, 0, incy, param, 0);
    }

    /**
     * F2jblas drotm fixed version, use long to store (n * incx) to avoid int overflow.
     */
    @Override
    public void drotm(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double[] param,
        int paramOffset) {
        double flag = 0.0D;
        double h11 = 0.0D;
        double h12 = 0.0D;
        double h21 = 0.0D;
        double h22 = 0.0D;
        double wi = 0.0D;
        double zi = 0.0D;
        flag = param[paramOffset];
        if (n > 0 && Double.compare(flag, -2.0D) != 0) { // If flag equals -2.0, do nothing and return directly.
            int index;
            if ((incx == incy && incx > 0) ^ true) {
                int xIndex = 1;
                int yIndex = 1;
                if (incx < 0) {
                    xIndex = 1 + (1 - n) * incx;
                }
                if (incy < 0) {
                    yIndex = 1 + (1 - n) * incy;
                }
                if (flag < 0.0) {
                    h11 = param[2 - 1 + paramOffset];
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi * h11 + zi * h12;
                        y[yIndex - 1 + yOffset] = wi * h21 + zi * h22;
                        xIndex += incx;
                        yIndex += incy;
                        ++index;
                    }
                } else if (flag == 0.0) {
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    index = 1;
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi + zi * h12;
                        y[yIndex - 1 + yOffset] = wi * h21 + zi;
                        xIndex += incx;
                        yIndex += incy;
                        ++index;
                    }
                } else {
                    h11 = param[2 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    for (int i = n; i > 0; --i) {
                        wi = x[xIndex - 1 + xOffset];
                        zi = y[yIndex - 1 + yOffset];
                        x[xIndex - 1 + xOffset] = wi * h11 + zi;
                        y[yIndex - 1 + yOffset] = -wi + h22 * zi;
                        xIndex += incx;
                        yIndex += incy;
                    }
                }
            } else {
                long nSteps = (long) n * incx;
                if (flag < 0.0) {
                    h11 = param[2 - 1 + paramOffset];
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi * h11 + zi * h12;
                        y[index - 1 + yOffset] = wi * h21 + zi * h22;
                        index += incx;
                    }
                } else if (flag == 0.0) {
                    h12 = param[4 - 1 + paramOffset];
                    h21 = param[3 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi + zi * h12;
                        y[index - 1 + yOffset] = wi * h21 + zi;
                        index += incx;
                    }
                } else {
                    h11 = param[2 - 1 + paramOffset];
                    h22 = param[5 - 1 + paramOffset];
                    index = 1;
                    for (long i = (nSteps - 1 + incx) / incx; i > 0; --i) {
                        wi = x[index - 1 + xOffset];
                        zi = y[index - 1 + yOffset];
                        x[index - 1 + xOffset] = wi * h11 + zi;
                        y[index - 1 + yOffset] = -wi + h22 * zi;
                        index += incx;
                    }
                }
            }
        }
    }

    @Override
    public void sscal(int n, float alp, float[] x, int incx) {
        org.netlib.blas.Sscal.sscal(n, alp, x, 0, incx);
    }

    @Override
    public void sscal(int n, float alp, float[] x, int xOffset, int incx) {
        org.netlib.blas.Sscal.sscal(n, alp, x, xOffset, incx);
    }

    @Override
    public void dscal(int n, double alp, double[] x, int incx) {
        org.netlib.blas.Dscal.dscal(n, alp, x, 0, incx);
    }

    @Override
    public void dscal(int n, double alp, double[] x, int xOffset, int incx) {
        org.netlib.blas.Dscal.dscal(n, alp, x, xOffset, incx);
    }

    @Override
    public void sswap(int n, float[] x, int incx, float[] y, int incy) {
        org.netlib.blas.Sswap.sswap(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void sswap(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        org.netlib.blas.Sswap.sswap(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void dswap(int n, double[] x, int incx, double[] y, int incy) {
        org.netlib.blas.Dswap.dswap(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void dswap(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dswap.dswap(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public int isamax(int n, float[] x, int incx) {
        return org.netlib.blas.Isamax.isamax(n, x, 0, incx);
    }

    @Override
    public int isamax(int n, float[] x, int xOffset, int incx) {
        return org.netlib.blas.Isamax.isamax(n, x, xOffset, incx);
    }

    @Override
    public int idamax(int n, double[] x, int incx) {
        return org.netlib.blas.Idamax.idamax(n, x, 0, incx);
    }

    @Override
    public int idamax(int n, double[] x, int xOffset, int incx) {
        return org.netlib.blas.Idamax.idamax(n, x, xOffset, incx);
    }

    @Override
    public void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int lda,
        double[] x, int incx, double beta, double[] y, int incy) {
        org.netlib.blas.Dgbmv.dgbmv(trans, m, n, kl, ku, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dgbmv.dgbmv(
            trans, m, n, kl, ku, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int lda, float[] x, int incx,
        float beta, float[] y, int incy) {
        org.netlib.blas.Sgbmv.sgbmv(trans, m, n, kl, ku, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx, float beta, float[] y, int yOffset, int incy) {
        org.netlib.blas.Sgbmv.sgbmv(
            trans, m, n, kl, ku, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dgemv(String trans, int m, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy) {
        org.netlib.blas.Dgemv.dgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dgemv(String trans, int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dgemv.dgemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void sgemv(String trans, int m, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy) {
        org.netlib.blas.Sgemv.sgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void sgemv(String trans, int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy) {
        org.netlib.blas.Sgemv.sgemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dger(int m, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda) {
        org.netlib.blas.Dger.dger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void dger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy,
        double[] a, int aOffset, int lda) {
        org.netlib.blas.Dger.dger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void sger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda) {
        org.netlib.blas.Sger.sger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void sger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy,
        float[] a, int aOffset, int lda) {
        org.netlib.blas.Sger.sger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void dsbmv(String uplo, int n, int k, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy) {
        org.netlib.blas.Dsbmv.dsbmv(uplo, n, k, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dsbmv(String uplo, int n, int k, double alpha, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dsbmv.dsbmv(uplo, n, k, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void ssbmv(String uplo, int n, int k, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy) {
        org.netlib.blas.Ssbmv.ssbmv(uplo, n, k, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void ssbmv(String uplo, int n, int k, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy) {
        org.netlib.blas.Ssbmv.ssbmv(uplo, n, k, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dspmv(String uplo, int n, double alpha, double[] a, double[] x, int incx, double beta,
        double[] y, int incy) {
        org.netlib.blas.Dspmv.dspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dspmv(String uplo, int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, int incx,
        double beta, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dspmv.dspmv(uplo, n, alpha, a, aOffset, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void sspmv(String uplo, int n, float alpha, float[] a, float[] x, int incx, float beta,
        float[] y, int incy) {
        org.netlib.blas.Sspmv.sspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void sspmv(String uplo, int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, int incx,
        float beta, float[] y, int yOffset, int incy) {
        org.netlib.blas.Sspmv.sspmv(uplo, n, alpha, a, aOffset, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dspr(String uplo, int n, double alpha, double[] x, int incx, double[] ap) {
        org.netlib.blas.Dspr.dspr(uplo, n, alpha, x, 0, incx, ap, 0);
    }

    @Override
    public void dspr(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] ap, int aOffset) {
        org.netlib.blas.Dspr.dspr(uplo, n, alpha, x, xOffset, incx, ap, aOffset);
    }

    @Override
    public void sspr(String uplo, int n, float alpha, float[] x, int incx, float[] ap) {
        org.netlib.blas.Sspr.sspr(uplo, n, alpha, x, 0, incx, ap, 0);
    }

    @Override
    public void sspr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] ap, int aOffset) {
        org.netlib.blas.Sspr.sspr(uplo, n, alpha, x, xOffset, incx, ap, aOffset);
    }

    @Override
    public void dspr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a) {
        org.netlib.blas.Dspr2.dspr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0);
    }

    @Override
    public void dspr2(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset,
        int incy, double[] a, int aOffset) {
        org.netlib.blas.Dspr2.dspr2(uplo, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset);
    }

    @Override
    public void sspr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a) {
        org.netlib.blas.Sspr2.sspr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0);
    }

    @Override
    public void sspr2(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
        int incy, float[] a, int aOffset) {
        org.netlib.blas.Sspr2.sspr2(uplo, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset);
    }

    @Override
    public void dsymv(String uplo, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy) {
        org.netlib.blas.Dsymv.dsymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dsymv(String uplo, int n, double alpha, double[] a, int aOffset, int lda, double[] x, int xOffset,
        int incx, double beta, double[] y, int yOffset, int incy) {
        org.netlib.blas.Dsymv.dsymv(uplo, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void ssymv(String uplo, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy) {
        org.netlib.blas.Ssymv.ssymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void ssymv(String uplo, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        int incx, float beta, float[] y, int yOffset, int incy) {
        org.netlib.blas.Ssymv.ssymv(uplo, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dsyr(String uplo, int n, double alpha, double[] x, int incx, double[] a, int lda) {
        org.netlib.blas.Dsyr.dsyr(uplo, n, alpha, x, 0, incx, a, 0, lda);
    }

    @Override
    public void dsyr(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] a, int aOffset,
        int lda) {
        org.netlib.blas.Dsyr.dsyr(uplo, n, alpha, x, xOffset, incx, a, aOffset, lda);
    }

    @Override
    public void ssyr(String uplo, int n, float alpha, float[] x, int incx, float[] a, int lda) {
        org.netlib.blas.Ssyr.ssyr(uplo, n, alpha, x, 0, incx, a, 0, lda);
    }

    @Override
    public void ssyr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] a, int aOffset,
        int lda) {
        org.netlib.blas.Ssyr.ssyr(uplo, n, alpha, x, xOffset, incx, a, aOffset, lda);
    }

    @Override
    public void dsyr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy,
        double[] a, int lda) {
        org.netlib.blas.Dsyr2.dsyr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void dsyr2(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset,
        int incy, double[] a, int aOffset, int lda) {
        org.netlib.blas.Dsyr2.dsyr2(uplo, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void ssyr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda) {
        org.netlib.blas.Ssyr2.ssyr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void ssyr2(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
        int incy, float[] a, int aOffset, int lda) {
        org.netlib.blas.Ssyr2.ssyr2(uplo, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx) {
        org.netlib.blas.Dtbmv.dtbmv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
    }

    @Override
    public void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx) {
        org.netlib.blas.Dtbmv.dtbmv(uplo, trans, diag, n, k, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx) {
        org.netlib.blas.Stbmv.stbmv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
    }

    @Override
    public void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx) {
        org.netlib.blas.Stbmv.stbmv(uplo, trans, diag, n, k, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx) {
        org.netlib.blas.Dtbsv.dtbsv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
    }

    @Override
    public void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx) {
        org.netlib.blas.Dtbsv.dtbsv(uplo, trans, diag, n, k, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx) {
        org.netlib.blas.Stbsv.stbsv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
    }

    @Override
    public void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx) {
        org.netlib.blas.Stbsv.stbsv(uplo, trans, diag, n, k, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void dtpmv(String uplo, String transa, String diag, int n, double[] a, double[] x, int incx) {
        org.netlib.blas.Dtpmv.dtpmv(uplo, transa, diag, n, a, 0, x, 0, incx);
    }

    @Override
    public void dtpmv(String uplo, String transa, String diag, int n, double[] a, int aOffset, double[] x,
        int xOffset, int incx) {
        org.netlib.blas.Dtpmv.dtpmv(uplo, transa, diag, n, a, aOffset, x, xOffset, incx);
    }

    @Override
    public void stpmv(String uplo, String transa, String diag, int n, float[] a, float[] x, int incx) {
        org.netlib.blas.Stpmv.stpmv(uplo, transa, diag, n, a, 0, x, 0, incx);
    }

    @Override
    public void stpmv(String uplo, String transa, String diag, int n, float[] a, int aOffset, float[] x,
        int xOffset, int incx) {
        org.netlib.blas.Stpmv.stpmv(uplo, transa, diag, n, a, aOffset, x, xOffset, incx);
    }

    @Override
    public void dtpsv(String uplo, String transa, String diag, int n, double[] a, double[] x, int incx) {
        org.netlib.blas.Dtpsv.dtpsv(uplo, transa, diag, n, a, 0, x, 0, incx);
    }

    @Override
    public void dtpsv(String uplo, String transa, String diag, int n, double[] a, int aOffset, double[] x,
        int xOffset, int incx) {
        org.netlib.blas.Dtpsv.dtpsv(uplo, transa, diag, n, a, aOffset, x, xOffset, incx);
    }

    @Override
    public void stpsv(String uplo, String transa, String diag, int n, float[] a, float[] x, int incx) {
        org.netlib.blas.Stpsv.stpsv(uplo, transa, diag, n, a, 0, x, 0, incx);
    }

    @Override
    public void stpsv(String uplo, String transa, String diag, int n, float[] a, int aOffset, float[] x,
        int xOffset, int incx) {
        org.netlib.blas.Stpsv.stpsv(uplo, transa, diag, n, a, aOffset, x, xOffset, incx);
    }

    @Override
    public void dtrmv(String uplo, String trans, String diag, int n, double[] a, int lda, double[] x, int incx) {
        org.netlib.blas.Dtrmv.dtrmv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
    }

    @Override
    public void dtrmv(String uplo, String trans, String diag, int n, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx) {
        org.netlib.blas.Dtrmv.dtrmv(uplo, trans, diag, n, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void strmv(String uplo, String trans, String diag, int n, float[] a, int lda, float[] x, int incx) {
        org.netlib.blas.Strmv.strmv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
    }

    @Override
    public void strmv(String uplo, String trans, String diag, int n, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx) {
        org.netlib.blas.Strmv.strmv(uplo, trans, diag, n, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void dtrsv(String uplo, String transa, String diag, int n, double[] a, int lda, double[] x, int incx) {
        org.netlib.blas.Dtrsv.dtrsv(uplo, transa, diag, n, a, 0, lda, x, 0, incx);
    }

    @Override
    public void dtrsv(String uplo, String transa, String diag, int n, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx) {
        org.netlib.blas.Dtrsv.dtrsv(uplo, transa, diag, n, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void strsv(String uplo, String transa, String diag, int n, float[] a, int lda, float[] x, int incx) {
        org.netlib.blas.Strsv.strsv(uplo, transa, diag, n, a, 0, lda, x, 0, incx);
    }

    @Override
    public void strsv(String uplo, String transa, String diag, int n, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx) {
        org.netlib.blas.Strsv.strsv(uplo, transa, diag, n, a, aOffset, lda, x, xOffset, incx);
    }

    @Override
    public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc) {
        org.netlib.blas.Dgemm.dgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int aOffset,
        int lda, double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        org.netlib.blas.Dgemm.dgemm(
            transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc) {
        org.netlib.blas.Sgemm.sgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int aOffset,
        int lda, float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        org.netlib.blas.Sgemm.sgemm(
            transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc) {
        org.netlib.blas.Dsymm.dsymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        org.netlib.blas.Dsymm.dsymm(side, uplo, m, n, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc) {
        org.netlib.blas.Ssymm.ssymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        org.netlib.blas.Ssymm.ssymm(side, uplo, m, n, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc) {
        org.netlib.blas.Dsyr2k.dsyr2k(uplo, trans, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        org.netlib.blas.Dsyr2k.dsyr2k(
            uplo, trans, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc) {
        org.netlib.blas.Ssyr2k.ssyr2k(uplo, trans, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        org.netlib.blas.Ssyr2k.ssyr2k(
            uplo, trans, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int lda, double beta,
        double[] c, int ldc) {
        org.netlib.blas.Dsyrk.dsyrk(uplo, trans, n, k, alpha, a, 0, lda, beta, c, 0, ldc);
    }

    @Override
    public void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int aOffset, int lda,
        double beta, double[] c, int cOffset, int ldc) {
        org.netlib.blas.Dsyrk.dsyrk(uplo, trans, n, k, alpha, a, aOffset, lda, beta, c, cOffset, ldc);
    }

    @Override
    public void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int lda, float beta,
        float[] c, int ldc) {
        org.netlib.blas.Ssyrk.ssyrk(uplo, trans, n, k, alpha, a, 0, lda, beta, c, 0, ldc);
    }

    @Override
    public void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int aOffset, int lda,
        float beta, float[] c, int cOffset, int ldc) {
        org.netlib.blas.Ssyrk.ssyrk(uplo, trans, n, k, alpha, a, aOffset, lda, beta, c, cOffset, ldc);
    }

    @Override
    public void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb) {
        org.netlib.blas.Dtrmm.dtrmm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
    }

    @Override
    public void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a,
        int aOffset, int lda, double[] b, int bOffset, int ldb) {
        org.netlib.blas.Dtrmm.dtrmm(side, uplo, transa, diag, m, n, alpha, a, aOffset, lda, b, bOffset, ldb);
    }

    @Override
    public void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha,
        float[] a, int lda, float[] b, int ldb) {
        org.netlib.blas.Strmm.strmm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
    }

    @Override
    public void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int aOffset, int lda, float[] b, int bOffset, int ldb) {
        org.netlib.blas.Strmm.strmm(side, uplo, transa, diag, m, n, alpha, a, aOffset, lda, b, bOffset, ldb);
    }

    @Override
    public void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb) {
        org.netlib.blas.Dtrsm.dtrsm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
    }

    @Override
    public void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a,
        int aOffset, int lda, double[] b, int bOffset, int ldb) {
        org.netlib.blas.Dtrsm.dtrsm(side, uplo, transa, diag, m, n, alpha, a, aOffset, lda, b, bOffset, ldb);
    }

    @Override
    public void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha,
        float[] a, int lda, float[] b, int ldb) {
        org.netlib.blas.Strsm.strsm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
    }

    @Override
    public void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int aOffset, int lda, float[] b, int bOffset, int ldb) {
        org.netlib.blas.Strsm.strsm(side, uplo, transa, diag, m, n, alpha, a, aOffset, lda, b, bOffset, ldb);
    }
}
