package jcuda.core;

import jcuda.runtime.JCuda;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;

public class JCudaManager {
    private static boolean isCudaAvailable = false;

    static {
        try {
            JCuda.setExceptionsEnabled(true);
            int[] deviceCount = new int[1];
            cudaGetDeviceCount(deviceCount);
            isCudaAvailable = deviceCount[0] > 0;
        }catch (Exception e) {
            isCudaAvailable = false;
        }
    }

    public static boolean isGpuAvailable() {
        return isCudaAvailable;
    }

    public static void printCudaInfo() {
        if(isGpuAvailable()) {
            System.out.println("CUDA is supported, GPU will be used.");
        }
        else {
            System.out.println("CUDA is not supported, CPU will be used.");
        }
    }

}
