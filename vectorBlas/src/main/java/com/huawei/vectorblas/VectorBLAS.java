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

import com.huawei.vectorblas.blas1.doubleprecision.Dasum;
import com.huawei.vectorblas.blas1.doubleprecision.Daxpy;
import com.huawei.vectorblas.blas1.doubleprecision.Dcopy;
import com.huawei.vectorblas.blas1.doubleprecision.Ddot;
import com.huawei.vectorblas.blas1.doubleprecision.Dnrm2;
import com.huawei.vectorblas.blas1.doubleprecision.Drot;
import com.huawei.vectorblas.blas1.doubleprecision.Drotm;
import com.huawei.vectorblas.blas1.doubleprecision.Dscal;
import com.huawei.vectorblas.blas1.doubleprecision.Dswap;
import com.huawei.vectorblas.blas1.doubleprecision.Idamax;
import com.huawei.vectorblas.blas1.singleprecision.Isamax;
import com.huawei.vectorblas.blas1.singleprecision.Sasum;
import com.huawei.vectorblas.blas1.singleprecision.Saxpy;
import com.huawei.vectorblas.blas1.singleprecision.Scopy;
import com.huawei.vectorblas.blas1.singleprecision.Sdot;
import com.huawei.vectorblas.blas1.singleprecision.Snrm2;
import com.huawei.vectorblas.blas1.singleprecision.Srot;
import com.huawei.vectorblas.blas1.singleprecision.Srotm;
import com.huawei.vectorblas.blas1.singleprecision.Sscal;
import com.huawei.vectorblas.blas1.singleprecision.Sswap;
import com.huawei.vectorblas.blas2.doubleprecision.Dgemv;
import com.huawei.vectorblas.blas2.doubleprecision.Dger;
import com.huawei.vectorblas.blas2.doubleprecision.Dspmv;
import com.huawei.vectorblas.blas2.doubleprecision.Dspr;
import com.huawei.vectorblas.blas2.doubleprecision.Dsymv;
import com.huawei.vectorblas.blas2.singleprecision.Sgemv;
import com.huawei.vectorblas.blas2.singleprecision.Sger;
import com.huawei.vectorblas.blas2.singleprecision.Sspmv;
import com.huawei.vectorblas.blas2.singleprecision.Sspr;
import com.huawei.vectorblas.blas2.singleprecision.Ssymv;
import com.huawei.vectorblas.blas3.doubleprecision.Dgemm;
import com.huawei.vectorblas.blas3.doubleprecision.Dsymm;
import com.huawei.vectorblas.blas3.singleprecision.Sgemm;
import com.huawei.vectorblas.blas3.singleprecision.Ssymm;

public class VectorBLAS extends F2jBLAS {
    @Override
    public double dasum(int n, double[] x, int incx) {
        return Dasum.dasum(n, x, 0, incx);
    }

    @Override
    public double dasum(int n, double[] x, int xOffset, int incx) {
        return Dasum.dasum(n, x, xOffset, incx);
    }

    @Override
    public float sasum(int n, float[] x, int incx) {
        return Sasum.sasum(n, x, 0, incx);
    }

    @Override
    public float sasum(int n, float[] x, int xOffset, int incx) {
        return Sasum.sasum(n, x, xOffset, incx);
    }

    @Override
    public void daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy) {
        Daxpy.daxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    @Override
    public void daxpy(int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        Daxpy.daxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy) {
        Saxpy.saxpy(n, alpha, x, 0, incx, y, 0, incy);
    }

    @Override
    public void saxpy(int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        Saxpy.saxpy(n, alpha, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void dcopy(int n, double[] x, int incx, double[] y, int incy) {
        Dcopy.dcopy(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void dcopy(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        Dcopy.dcopy(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void scopy(int n, float[] x, int incx, float[] y, int incy) {
        Scopy.scopy(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void scopy(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        Scopy.scopy(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public double ddot(int n, double[] x, int incx, double[] y, int incy) {
        return Ddot.ddot(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public double ddot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        return Ddot.ddot(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public float sdot(int n, float[] x, int incx, float[] y, int incy) {
        return Sdot.sdot(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public float sdot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        return Sdot.sdot(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public float snrm2(int n, float[] x, int incx) {
        return Snrm2.snrm2(n, x, 0, incx);
    }

    @Override
    public float snrm2(int n, float[] x, int xOffset, int incx) {
        return Snrm2.snrm2(n, x, xOffset, incx);
    }

    @Override
    public double dnrm2(int n, double[] x, int incx) {
        return Dnrm2.dnrm2(n, x, 0, incx);
    }

    @Override
    public double dnrm2(int n, double[] x, int xOffset, int incx) {
        return Dnrm2.dnrm2(n, x, xOffset, incx);
    }

    @Override
    public void srot(int n, float[] x, int incx, float[] y, int incy, float c, float s) {
        Srot.srot(n, x, 0, incx, y, 0, incy, c, s);
    }

    @Override
    public void srot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float c, float s) {
        Srot.srot(n, x, xOffset, incx, y, yOffset, incy, c, s);
    }

    @Override
    public void drot(int n, double[] x, int incx, double[] y, int incy, double c, double s) {
        Drot.drot(n, x, 0, incx, y, 0, incy, c, s);
    }

    @Override
    public void drot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double c, double s) {
        Drot.drot(n, x, xOffset, incx, y, yOffset, incy, c, s);
    }

    @Override
    public void srotm(int n, float[] x, int incx, float[] y, int incy, float[] param) {
        Srotm.srotm(n, x, 0, incx, y, 0, incy, param, 0);
    }

    @Override
    public void srotm(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float[] param,
        int paramOffset) {
        Srotm.srotm(n, x, xOffset, incx, y, yOffset, incy, param, paramOffset);
    }

    @Override
    public void drotm(int n, double[] x, int incx, double[] y, int incy, double[] param) {
        Drotm.drotm(n, x, 0, incx, y, 0, incy, param, 0);
    }

    @Override
    public void drotm(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double[] param,
        int paramOffset) {
        Drotm.drotm(n, x, xOffset, incx, y, yOffset, incy, param, paramOffset);
    }

    @Override
    public void sscal(int n, float alp, float[] x, int incx) {
        Sscal.sscal(n, alp, x, 0, incx);
    }

    @Override
    public void sscal(int n, float alp, float[] x, int xOffset, int incx) {
        Sscal.sscal(n, alp, x, xOffset, incx);
    }

    @Override
    public void dscal(int n, double alp, double[] x, int incx) {
        Dscal.dscal(n, alp, x, 0, incx);
    }

    @Override
    public void dscal(int n, double alp, double[] x, int xOffset, int incx) {
        Dscal.dscal(n, alp, x, xOffset, incx);
    }

    @Override
    public void sswap(int n, float[] x, int incx, float[] y, int incy) {
        Sswap.sswap(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void sswap(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy) {
        Sswap.sswap(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public void dswap(int n, double[] x, int incx, double[] y, int incy) {
        Dswap.dswap(n, x, 0, incx, y, 0, incy);
    }

    @Override
    public void dswap(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy) {
        Dswap.dswap(n, x, xOffset, incx, y, yOffset, incy);
    }

    @Override
    public int isamax(int n, float[] x, int incx) {
        return Isamax.isamax(n, x, 0, incx);
    }

    @Override
    public int isamax(int n, float[] x, int xOffset, int incx) {
        return Isamax.isamax(n, x, xOffset, incx);
    }

    @Override
    public int idamax(int n, double[] x, int incx) {
        return Idamax.idamax(n, x, 0, incx);
    }

    @Override
    public int idamax(int n, double[] x, int xOffset, int incx) {
        return Idamax.idamax(n, x, xOffset, incx);
    }

    @Override
    public void dgemv(String trans, int m, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy) {
        Dgemv.dgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dgemv(String trans, int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy) {
        Dgemv.dgemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void sgemv(String trans, int m, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy) {
        Sgemv.sgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void sgemv(String trans, int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy) {
        Sgemv.sgemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dger(int m, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda) {
        Dger.dger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void dger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy,
        double[] a, int aOffset, int lda) {
        Dger.dger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void sger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda) {
        Sger.sger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
    }

    @Override
    public void sger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy,
        float[] a, int aOffset, int lda) {
        Sger.sger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda);
    }

    @Override
    public void dspmv(String uplo, int n, double alpha, double[] a, double[] x, int incx, double beta,
        double[] y, int incy) {
        Dspmv.dspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dspmv(String uplo, int n, double alpha, double[] a, int aOffset, double[] x, int xOffset, int incx,
        double beta, double[] y, int yOffset, int incy) {
        Dspmv.dspmv(uplo, n, alpha, a, aOffset, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void sspmv(String uplo, int n, float alpha, float[] a, float[] x, int incx, float beta,
        float[] y, int incy) {
        Sspmv.sspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void sspmv(String uplo, int n, float alpha, float[] a, int aOffset, float[] x, int xOffset, int incx,
        float beta, float[] y, int yOffset, int incy) {
        Sspmv.sspmv(uplo, n, alpha, a, aOffset, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dspr(String uplo, int n, double alpha, double[] x, int incx, double[] ap) {
        Dspr.dspr(uplo, n, alpha, x, 0, incx, ap, 0);
    }

    @Override
    public void dspr(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] ap, int aOffset) {
        Dspr.dspr(uplo, n, alpha, x, xOffset, incx, ap, aOffset);
    }

    @Override
    public void sspr(String uplo, int n, float alpha, float[] x, int incx, float[] ap) {
        Sspr.sspr(uplo, n, alpha, x, 0, incx, ap, 0);
    }

    @Override
    public void sspr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] ap, int aOffset) {
        Sspr.sspr(uplo, n, alpha, x, xOffset, incx, ap, aOffset);
    }

    @Override
    public void dsymv(String uplo, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy) {
        Dsymv.dsymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void dsymv(String uplo, int n, double alpha, double[] a, int aOffset, int lda, double[] x, int xOffset,
        int incx, double beta, double[] y, int yOffset, int incy) {
        Dsymv.dsymv(uplo, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void ssymv(String uplo, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy) {
        Ssymv.ssymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
    }

    @Override
    public void ssymv(String uplo, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        int incx, float beta, float[] y, int yOffset, int incy) {
        Ssymv.ssymv(uplo, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y, yOffset, incy);
    }

    @Override
    public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc) {
        Dgemm.dgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int aOffset,
        int lda, double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        Dgemm.dgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc) {
        Sgemm.sgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int aOffset,
        int lda, float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        Sgemm.sgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc) {
        Dsymm.dsymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc) {
        Dsymm.dsymm(side, uplo, m, n, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }

    @Override
    public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc) {
        Ssymm.ssymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    }

    @Override
    public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc) {
        Ssymm.ssymm(side, uplo, m, n, alpha, a, aOffset, lda, b, bOffset, ldb, beta, c, cOffset, ldc);
    }
}
