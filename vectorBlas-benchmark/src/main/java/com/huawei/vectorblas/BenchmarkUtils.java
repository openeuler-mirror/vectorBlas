package com.huawei.vectorblas;

import java.util.Random;

public class BenchmarkUtils {
    private static final Random rand = new Random(0);

    public static double randomDouble() {
        return rand.nextDouble();
    }

    public static double[] randomDoubleArray(int n) {
        double[] res = new double[n];
        for (int i = 0; i < n; i++) {
            res[i] = rand.nextDouble();
        }
        return res;
    }

    public static float randomFloat() {
        return rand.nextFloat();
    }

    public static float[] randomFloatArray(int n) {
        float[] res = new float[n];
        for (int i = 0; i < n; i++) {
            res[i] = rand.nextFloat();
        }
        return res;
    }
}
