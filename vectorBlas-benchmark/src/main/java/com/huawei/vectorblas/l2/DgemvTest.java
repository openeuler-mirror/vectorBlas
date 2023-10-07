/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

package com.huawei.vectorblas.l2;

import com.huawei.vectorblas.BenchmarkUtils;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.infra.Blackhole;

public class DgemvTest extends L2Benchmark {
    @Param({"N", "T"})
    public String trans;

    @Param({"500", "4000", "10000"})
    public int m;
    public int n;

    public double alpha;
    public double[] a;
    public double[] x;
    public double beta;
    public double[] y, yclone;

    @Setup(Level.Trial)
    public void setup() {
        n = m;
        alpha = BenchmarkUtils.randomDouble();
        a = BenchmarkUtils.randomDoubleArray(m * n);
        x = BenchmarkUtils.randomDoubleArray(trans.equals("T") ? m : n);
        beta = BenchmarkUtils.randomDouble();
        y = BenchmarkUtils.randomDoubleArray(trans.equals("T") ? n : m);
    }

    @Benchmark
    public void vecBlasDgemv(Blackhole bh) {
        vBlas.dgemv(trans, m, n, alpha, a, 0, m, x, 0, 1, beta, yclone = y.clone(), 0, 1);
        bh.consume(yclone);
    }
}
