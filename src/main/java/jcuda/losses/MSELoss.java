package jcuda.losses;

import jcuda.core.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class MSELoss extends Loss {

    /**
     * Computes Mean Squared Error (MSE).
     * If predictions and targets are on GPU, uses JCuda kernel to compute squared differences,
     * copies the result to host, sums them, and returns the average.
     */
    @Override
    public float forward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        int N = predData.length;
        if (predictions.getGpuData() != null && targets.getGpuData() != null) {
            // Allocate device memory for squared differences.
            Pointer d_out = new Pointer();
            cudaMalloc(d_out, N * Sizeof.FLOAT);

            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(predictions.getGpuData()),
                    Pointer.to(targets.getGpuData()),
                    Pointer.to(d_out),
                    Pointer.to(new int[]{N})
            );
            cuLaunchKernel(LossKernels.mseLossForwardFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);

            // Copy result back to host and sum.
            float[] squaredErrors = new float[N];
            cudaMemcpy(Pointer.to(squaredErrors), d_out, (long)N * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(d_out);

            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += squaredErrors[i];
            }
            return sum / N;
        } else {
            // CPU fallback.
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                float diff = predData[i] - targetData[i];
                sum += diff * diff;
            }
            return sum / N;
        }
    }
}
