/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

package com.huawei.vectorblas.l1;

import com.huawei.vectorblas.BenchmarkUtils;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.infra.Blackhole;


public class SdotTest extends L1Benchmark {
    @Param({"100", "1000", "10000", "100000", "1000000", "10000000", "100000000"})
    public int n;
    public float[] x;
    public float[] y;

    @Setup(Level.Trial)
    public void setup() {
        x = BenchmarkUtils.randomFloatArray(n);
        y = BenchmarkUtils.randomFloatArray(n);
    }

    @Benchmark
    public void vecBlasSdot(Blackhole bh) {
        bh.consume(vBlas.sdot(n, x, 0, 1, y, 0, 1));
    }
}
