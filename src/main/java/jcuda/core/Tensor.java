package jcuda.core;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.runtime.JCuda.*;

public class Tensor {
    private float[] data;
    private Pointer gpuData;
    private int size;
    private boolean onGPU;

    public Tensor(float[] data) {
        this.data = data;
        this.size = data.length;
        this.onGPU = false;
    }

    public void to(DeviceManager.Device device, int gpuIndex) {
        if(device == DeviceManager.Device.GPU && !onGPU) {
            DeviceManager.setDevice(gpuIndex);
            gpuData = new Pointer();
            cudaMalloc(gpuData, (long )size * Sizeof.FLOAT);
            cudaMemcpy(gpuData, Pointer.to(data), (long) size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            onGPU = true;
        } else if(device == DeviceManager.Device.CPU && onGPU) {
            cudaMemcpy(Pointer.to(data), gpuData, (long )size * Sizeof.FLOAT,cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cudaFree(gpuData);
            gpuData=null;
            onGPU = false;
        }
    }

    public void printTensor() {
        float[] values = data;
        System.out.print("[");
        for(float v: values) {
            System.out.print(v + " ");
        }
        System.out.println("]");
    }
}
