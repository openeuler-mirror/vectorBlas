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

package com.huawei.vectorblas.utils;

import java.util.Random;

public class ArrayUtil {
    public static int loopBound(int length, int size) {
        return roundDown(length, size);
    }

    private static int roundDown(int length, int size) {
        if ((size & (size - 1)) == 0) {
            // Size is zero or a power of two, so we got this.
            return length & ~(size - 1);
        } else {
            return roundDownNPOT(length, size);
        }
    }

    private static int roundDownNPOT(int length, int size) {
        if (length >= 0) {
            return length - (length % size);
        } else {
            return length - Math.floorMod(length, Math.abs(size));
        }
    }

    private static final Random RANDOM = new Random(0);

    public static double randomDouble() {
        return RANDOM.nextDouble();
    }

    public static void randomDoubleArray(double[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = RANDOM.nextDouble() - 0.5d; // Produce double values between -0.5 and 0.5.
        }
    }

    public static float randomFloat() {
        return RANDOM.nextFloat();
    }

    public static void randomFloatArray(float[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = RANDOM.nextFloat() - 0.5f; // Produce float values between -0.5 and 0.5.
        }
    }
}
