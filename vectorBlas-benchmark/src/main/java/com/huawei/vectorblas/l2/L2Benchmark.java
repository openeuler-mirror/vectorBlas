package com.huawei.vectorblas.l2;

import com.huawei.vectorblas.BlasBenchmark;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Warmup;

@Warmup(iterations = 3)
@Measurement(iterations = 6, time = 10)
public class L2Benchmark extends BlasBenchmark {
}
