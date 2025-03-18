package jcuda.core;

import static jcuda.runtime.JCuda.*;
import jcuda.runtime.JCuda;

public class DeviceManager {
    public enum Device {
        CPU, GPU
    }
    private static int deviceCount = 0;
    private static int currentGpuIndex = 0;
    private static Device currentDevice = Device.GPU;

    static {
        try {
            JCuda.setExceptionsEnabled(true);
            int[] count = new int[1];
            cudaGetDeviceCount(count);
            deviceCount = count[0];

            if(deviceCount > 0) {
                currentDevice = Device.GPU;
                setDevice(0);
            }
        } catch (Exception e) {
            deviceCount = 0;
            currentDevice = Device.CPU;
        }
    }

    public static Device getDevice() {
        return currentDevice;
    }

    public static void setDevice(int gpuIndex) {
        if(deviceCount ==0 || gpuIndex >= deviceCount || gpuIndex < 0) {
            System.out.println("Invalid GPU index, CPU will be used");
            currentDevice = Device.CPU;
        }else {
            currentGpuIndex = gpuIndex;
            cudaSetDevice(gpuIndex);
            currentDevice=Device.GPU;
            System.out.println("The GPU currently in use: " + gpuIndex);
        }
    }

    public static boolean isUsingGPU() {
        return currentDevice == Device.GPU;
    }

    public static int getCurrentGpuIndex() {
        return currentGpuIndex;
    }

    public static int getDeviceCount() {
        return deviceCount;
    }
}
