package jcuda.losses;

import jcuda.core.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;

import static jcuda.losses.LossKernels.crossEntropyLossBackwardFunction;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CrossEntropyLossGradient implements LossGradient {

    /**
     * Computes the gradient for Cross-Entropy Loss:
     * grad = -(target/(prediction+epsilon))/N.
     * Uses JCuda kernel if available; otherwise, uses CPU fallback.
     */
    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        int N = predData.length;
        float epsilon = 1e-7f;
        float[] gradData = new float[N];
        Tensor gradTensor = new Tensor(gradData, predictions.getShape());

        if (predictions.getGpuData() != null && targets.getGpuData() != null) {
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(predictions.getGpuData()),
                    Pointer.to(targets.getGpuData()),
                    Pointer.to(gradTensor.getGpuData()),
                    Pointer.to(new int[]{N}),
                    Pointer.to(new float[]{epsilon})
            );
            cuLaunchKernel(crossEntropyLossBackwardFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);
            cudaMemcpy(Pointer.to(gradData), gradTensor.getGpuData(), (long)N * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        } else {
            for (int i = 0; i < N; i++) {
                gradData[i] = -targetData[i] / (predData[i] + epsilon) / N;
            }
            gradTensor.setData(gradData);
        }
        return gradTensor;
    }
}
