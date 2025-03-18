package jcuda.core;

import jcuda.runtime.cudaDeviceProp;
import static jcuda.runtime.JCuda.*;

public class GPUInfo {
    public static void printGPUDetails() {
        int[] deviceCount = {0};
        cudaGetDeviceCount(deviceCount);
        System.out.println("Total GPU Count: " + deviceCount[0]);

        for (int i = 0; i < deviceCount[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            cudaGetDeviceProperties(prop, i);

            // GPU ismini byte dizisinden düzgün bir String'e çeviriyoruz
            String gpuName = new String(prop.name).trim();

            System.out.println("GPU " + i + " - " + gpuName);
            System.out.println("   Compute Capability: " + prop.major + "." + prop.minor);
            System.out.println("   Global Memory: " + prop.totalGlobalMem / (1024 * 1024) + " MB");
        }
    }
}
