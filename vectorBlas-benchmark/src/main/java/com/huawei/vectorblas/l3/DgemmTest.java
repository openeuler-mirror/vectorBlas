package com.huawei.vectorblas.l3;

import com.huawei.vectorblas.BenchmarkUtils;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.infra.Blackhole;

public class DgemmTest extends L3Benchmark {
    @Param({"N", "T"})
    public String transa;
    @Param({"N", "T"})
    public String transb;

    public int m;
    public int n;
    @Param({"1000", "2000", "3000"})
    public int k;

    public double alpha;
    public double[] a;
    public double[] b;
    public double beta;
    public double[] c, cclone;

    @Setup(Level.Trial)
    public void setup() {
        m = k;
        n = k;
        alpha = BenchmarkUtils.randomDouble();
        a = BenchmarkUtils.randomDoubleArray(k * m);
        b = BenchmarkUtils.randomDoubleArray(k * n);
        beta = BenchmarkUtils.randomDouble();
        c = BenchmarkUtils.randomDoubleArray(m * n);
    }

    @Benchmark
    public void vecBlasDgemm(Blackhole bh) {
        vBlas.dgemm(transa, transb, m, n, k, alpha, a, 0, transa.equals("N") ? m : k, b, 0, transb.equals("N") ? k : n, beta, cclone = c.clone(), 0, m);
        bh.consume(cclone);
    }
}
