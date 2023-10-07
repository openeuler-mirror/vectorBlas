package com.huawei.vectorblas.l3;

import com.huawei.vectorblas.BlasBenchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Warmup;

@Warmup(iterations = 3, time = 10)
@Measurement(iterations = 3, time = 20)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class L3Benchmark extends BlasBenchmark {
}
