package jcuda.losses;

import jcuda.core.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class MSELossGradient implements LossGradient {

    /**
     * Computes the gradient for MSE: grad = 2*(predictions - targets)/N.
     * Uses JCuda kernel if tensors are on GPU.
     */
    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        int N = predData.length;
        float[] gradData = new float[N];
        Tensor gradTensor = new Tensor(gradData, predictions.getShape());

        if (predictions.getGpuData() != null && targets.getGpuData() != null) {
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(predictions.getGpuData()),
                    Pointer.to(targets.getGpuData()),
                    Pointer.to(gradTensor.getGpuData()),
                    Pointer.to(new int[]{N})
            );
            cuLaunchKernel(LossKernels.mseLossBackwardFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);
            cudaMemcpy(Pointer.to(gradData), gradTensor.getGpuData(), (long)N * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        } else {
            for (int i = 0; i < N; i++) {
                gradData[i] = 2 * (predData[i] - targetData[i]) / N;
            }
            gradTensor.setData(gradData);
        }
        return gradTensor;
    }
}
