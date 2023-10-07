package com.huawei.vectorblas.l3;

import com.huawei.vectorblas.BenchmarkUtils;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.infra.Blackhole;

public class SgemmTest extends L3Benchmark {
    @Param({"N", "T"})
    public String transa;
    @Param({"N", "T"})
    public String transb;

    @Param({"1000", "2000", "3000"})
    public int m;
    public int n;
    public int k;

    public float alpha;
    public float[] a;
    public float[] b;
    public float beta;
    public float[] c, cclone;

    @Setup(Level.Trial)
    public void setup() {
        n = m;
        k = m;
        alpha = BenchmarkUtils.randomFloat();
        a = BenchmarkUtils.randomFloatArray(k * m);
        b = BenchmarkUtils.randomFloatArray(k * n);
        beta = BenchmarkUtils.randomFloat();
        c = BenchmarkUtils.randomFloatArray(m * n);
    }

    @Benchmark
    public void vecBlasSgemm(Blackhole bh) {
        vBlas.sgemm(transa, transb, m, n, k, alpha, a, 0, transa.equals("N") ? m : k, b, 0, transb.equals("N") ? k : n, beta, cclone = c.clone(), 0, m);
        bh.consume(cclone);
    }
}
