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

public interface BLAS {
    // BLAS 1
    double dasum(int n, double[] x, int incx);

    double dasum(int n, double[] x, int xOffset, int incx);

    float sasum(int n, float[] x, int incx);

    float sasum(int n, float[] x, int xOffset, int incx);

    void daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy);

    void daxpy(int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy);

    void saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy);

    void saxpy(int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy);

    void dcopy(int n, double[] x, int incx, double[] y, int incy);

    void dcopy(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy);

    void scopy(int n, float[] x, int incx, float[] y, int incy);

    void scopy(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy);

    double ddot(int n, double[] x, int incx, double[] y, int incy);

    double ddot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy);

    float sdot(int n, float[] x, int incx, float[] y, int incy);

    float sdot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy);

    float snrm2(int n, float[] x, int incx);

    float snrm2(int n, float[] x, int xOffset, int incx);

    double dnrm2(int n, double[] x, int incx);

    double dnrm2(int n, double[] x, int xOffset, int incx);

    void srot(int n, float[] x, int incx, float[] y, int incy, float c, float s);

    void srot(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float c, float s);

    void drot(int n, double[] x, int incx, double[] y, int incy, double c, double s);

    void drot(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double c, double s);

    void srotm(int n, float[] x, int incx, float[] y, int incy, float[] param);

    void srotm(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy, float[] param,
        int paramOffset);

    void drotm(int n, double[] x, int incx, double[] y, int incy, double[] param);

    void drotm(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy, double[] param,
        int paramOffset);

    void sscal(int n, float alp, float[] x, int incx);

    void sscal(int n, float alp, float[] x, int xOffset, int incx);

    void dscal(int n, double alp, double[] x, int incx);

    void dscal(int n, double alp, double[] x, int xOffset, int incx);

    void sswap(int n, float[] x, int incx, float[] y, int incy);

    void sswap(int n, float[] x, int xOffset, int incx, float[] y, int yOffset, int incy);

    void dswap(int n, double[] x, int incx, double[] y, int incy);

    void dswap(int n, double[] x, int xOffset, int incx, double[] y, int yOffset, int incy);

    int isamax(int n, float[] x, int incx);

    int isamax(int n, float[] x, int xOffset, int incx);

    int idamax(int n, double[] x, int incx);

    int idamax(int n, double[] x, int xOffset, int incx);

    // BLAS 2
    void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int lda, double[] x,
        int incx, double beta, double[] y, int incy);

    void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int aOffset,
        int lda, double[] x, int xOffset, int incx, double beta, double[] y, int yOffset, int incy);

    void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int lda, float[] x,
        int incx, float beta, float[] y, int incy);

    void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx, float beta, float[] y, int yOffset, int incy);

    void dgemv(String trans, int m, int n, double alpha, double[] a, int lda, double[] x,
        int incx, double beta, double[] y, int incy);

    void dgemv(String trans, int m, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy);

    void sgemv(String trans, int m, int n, float alpha, float[] a, int lda, float[] x,
        int incx, float beta, float[] y, int incy);

    void sgemv(String trans, int m, int n, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy);

    void dger(int m, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda);

    void dger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y, int yOffset,
        int incy, double[] a, int aOffset, int lda);

    void sger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda);

    void sger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y, int yOffset,
        int incy, float[] a, int aOffset, int lda);

    void dsbmv(String uplo, int n, int k, double alpha, double[] a, int lda, double[] x, int incx,
        double beta, double[] y, int incy);

    void dsbmv(String uplo, int n, int k, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy);

    void ssbmv(String uplo, int n, int k, float alpha, float[] a, int lda, float[] x, int incx,
        float beta, float[] y, int incy);

    void ssbmv(String uplo, int n, int k, float alpha, float[] a, int aOffset, int lda, float[] x,
        int xOffset, int incx, float beta, float[] y, int yOffset, int incy);

    void dspmv(String uplo, int n, double alpha, double[] a, double[] x, int incx, double beta, double[] y, int incy);

    void dspmv(String uplo, int n, double alpha, double[] a, int aOffset, double[] x, int xOffset,
        int incx, double beta, double[] y, int yOffset, int incy);

    void sspmv(String uplo, int n, float alpha, float[] a, float[] x, int incx, float beta, float[] y, int incy);

    void sspmv(String uplo, int n, float alpha, float[] a, int aOffset, float[] x, int xOffset,
        int incx, float beta, float[] y, int yOffset, int incy);

    void dspr(String uplo, int n, double alpha, double[] x, int incx, double[] ap);

    void dspr(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] ap, int aOffset);

    void sspr(String uplo, int n, float alpha, float[] x, int incx, float[] ap);

    void sspr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] ap, int aOffset);

    void dspr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a);

    void dspr2(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] y,
        int yOffset, int incy, double[] a, int aOffset);

    void sspr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a);

    void sspr2(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] y,
        int yOffset, int incy, float[] a, int aOffset);

    void dsymv(String uplo, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta,
        double[] y, int incy);

    void dsymv(String uplo, int n, double alpha, double[] a, int aOffset, int lda, double[] x,
        int xOffset, int incx, double beta, double[] y, int yOffset, int incy);

    void ssymv(String uplo, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta,
        float[] y, int incy);

    void ssymv(String uplo, int n, float alpha, float[] a, int aOffset, int lda, float[] x, int xOffset,
        int incx, float beta, float[] y, int yOffset, int incy);

    void dsyr(String uplo, int n, double alpha, double[] x, int incx, double[] a, int lda);

    void dsyr(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] a, int aOffset, int lda);

    void ssyr(String uplo, int n, float alpha, float[] x, int incx, float[] a, int lda);

    void ssyr(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] a, int aOffset, int lda);

    void dsyr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda);

    void dsyr2(String uplo, int n, double alpha, double[] x, int xOffset, int incx, double[] y,
        int yOffset, int incy, double[] a, int aOffset, int lda);

    void ssyr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda);

    void ssyr2(String uplo, int n, float alpha, float[] x, int xOffset, int incx, float[] y,
        int yOffset, int incy, float[] a, int aOffset, int lda);

    void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx);

    void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx);

    void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx);

    void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx);

    void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx);

    void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx);

    void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx);

    void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx);

    void dtpmv(String uplo, String transa, String diag, int n, double[] a, double[] x, int incx);

    void dtpmv(String uplo, String transa, String diag, int n, double[] a, int aOffset, double[] x,
        int xOffset, int incx);

    void stpmv(String uplo, String transa, String diag, int n, float[] a, float[] x, int incx);

    void stpmv(String uplo, String transa, String diag, int n, float[] a, int aOffset, float[] x,
        int xOffset, int incx);

    void dtpsv(String uplo, String transa, String diag, int n, double[] a, double[] x, int incx);

    void dtpsv(String uplo, String transa, String diag, int n, double[] a, int aOffset, double[] x,
        int xOffset, int incx);

    void stpsv(String uplo, String transa, String diag, int n, float[] a, float[] x, int incx);

    void stpsv(String uplo, String transa, String diag, int n, float[] a, int aOffset, float[] x,
        int xOffset, int incx);

    void dtrmv(String uplo, String trans, String diag, int n, double[] a, int lda, double[] x, int incx);

    void dtrmv(String uplo, String trans, String diag, int n, double[] a, int aOffset, int lda,
        double[] x, int xOffset, int incx);

    void strmv(String uplo, String trans, String diag, int n, float[] a, int lda, float[] x, int incx);

    void strmv(String uplo, String trans, String diag, int n, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx);

    void dtrsv(String uplo, String transa, String diag, int n, double[] a, int lda, double[] x, int incx);

    void dtrsv(String uplo, String transa, String diag, int n, double[] a,
        int aOffset, int lda, double[] x, int xOffset, int incx);

    void strsv(String uplo, String transa, String diag, int n, float[] a, int lda, float[] x, int incx);

    void strsv(String uplo, String transa, String diag, int n, float[] a, int aOffset, int lda,
        float[] x, int xOffset, int incx);

    // BLAS 3
    void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc);

    void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int aOffset,
        int lda, double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc);

    void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a,
        int lda, float[] b, int ldb, float beta, float[] c, int ldc);

    void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int aOffset,
        int lda, float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc);

    void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc);

    void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc);

    void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc);

    void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc);

    void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int lda,
        double[] b, int ldb, double beta, double[] c, int ldc);

    void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb, double beta, double[] c, int cOffset, int ldc);

    void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int lda,
        float[] b, int ldb, float beta, float[] c, int ldc);

    void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb, float beta, float[] c, int cOffset, int ldc);

    void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int lda,
        double beta, double[] c, int ldc);

    void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int aOffset, int lda,
        double beta, double[] c, int cOffset, int ldc);

    void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int lda,
        float beta, float[] c, int ldc);

    void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int aOffset, int lda,
        float beta, float[] c, int cOffset, int ldc);

    void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb);

    void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb);

    void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int lda, float[] b, int ldb);

    void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int aOffset, int lda, float[] b, int bOffset, int ldb);

    void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb);

    void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha,
        double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb);

    void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int lda, float[] b, int ldb);

    void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a,
        int aOffset, int lda, float[] b, int bOffset, int ldb);
}
