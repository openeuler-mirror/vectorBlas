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

public class SgemvTest extends L2Benchmark {
    @Param({"N", "T"})
    public String trans;

    @Param({"500", "4000", "10000"})
    public int m;
    public int n;

    public float alpha;
    public float[] a;
    public float[] x;
    public float beta;
    public float[] y, yclone;

    @Setup(Level.Trial)
    public void setup() {
        n = m;
        alpha = BenchmarkUtils.randomFloat();
        a = BenchmarkUtils.randomFloatArray(m * n);
        x = BenchmarkUtils.randomFloatArray(trans.equals("T") ? m : n);
        beta = BenchmarkUtils.randomFloat();
        y = BenchmarkUtils.randomFloatArray(trans.equals("T") ? n : m);
    }

    @Benchmark
    public void vecBlasSgemv(Blackhole bh) {
        vBlas.sgemv(trans, m, n, alpha, a, 0, m, x, 0, 1, beta, yclone = y.clone(), 0, 1);
        bh.consume(yclone);
    }
}
