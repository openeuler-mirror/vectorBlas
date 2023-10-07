package com.huawei.vectorblas.l1;

import com.huawei.vectorblas.BlasBenchmark;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Warmup;

@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 6, time = 3)
public class L1Benchmark extends BlasBenchmark {
}
