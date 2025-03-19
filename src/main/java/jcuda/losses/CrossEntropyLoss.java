package jcuda.losses;

import jcuda.core.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;

import static jcuda.losses.LossKernels.crossEntropyLossForwardFunction;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CrossEntropyLoss extends Loss {

    /**
     * Computes Cross-Entropy Loss:
     * loss = -(1/N) * sum(target * log(prediction + epsilon)).
     * If GPU is available, uses JCuda kernel; otherwise, uses CPU fallback.
     */
    @Override
    public float forward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        int N = predData.length;
        float epsilon = 1e-7f;

        if (predictions.getGpuData() != null && targets.getGpuData() != null) {
            Pointer d_out = new Pointer();
            cudaMalloc(d_out, N * Sizeof.FLOAT);

            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(predictions.getGpuData()),
                    Pointer.to(targets.getGpuData()),
                    Pointer.to(d_out),
                    Pointer.to(new int[]{N}),
                    Pointer.to(new float[]{epsilon})
            );
            cuLaunchKernel(crossEntropyLossForwardFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);

            float[] losses = new float[N];
            cudaMemcpy(Pointer.to(losses), d_out, (long)N * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(d_out);

            float lossSum = 0.0f;
            for (int i = 0; i < N; i++) {
                lossSum += losses[i];
            }
            return lossSum / N;
        } else {
            float lossSum = 0.0f;
            for (int i = 0; i < N; i++) {
                lossSum += (float)(-targetData[i] * Math.log(predData[i] + epsilon));
            }
            return lossSum / N;
        }
    }
}
