package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.core.Tensor;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class RMSPropOptimizer extends Optimizer {
    private final float beta;
    private final float epsilon;
    private Tensor[] cache;

    public RMSPropOptimizer(float learningRate, float beta, float epsilon) {
        super(learningRate);
        this.beta = beta;
        this.epsilon = epsilon;
    }

    /**
     * Updates parameters using RMSProp.
     * If tensors are on GPU, launches the JCuda kernel; otherwise, uses CPU fallback.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Number of parameters and gradients must match.");

        if (cache == null) {
            cache = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                cache[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            int N = parameters[i].getData().length;
            if (parameters[i].getGpuData() != null && gradients[i].getGpuData() != null &&
                    cache[i].getGpuData() != null) {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                Pointer kernelParams = Pointer.to(
                        Pointer.to(parameters[i].getGpuData()),
                        Pointer.to(gradients[i].getGpuData()),
                        Pointer.to(cache[i].getGpuData()),
                        Pointer.to(new float[]{learningRate}),
                        Pointer.to(new float[]{beta}),
                        Pointer.to(new float[]{epsilon}),
                        Pointer.to(new int[]{N})
                );
                cuLaunchKernel(OptimizerKernels.rmspropUpdateFunction,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, null,
                        kernelParams, null);
            } else {
                // CPU fallback
                float[] paramData = parameters[i].getData();
                float[] gradData = gradients[i].getData();
                float[] cacheData = cache[i].getData();
                for (int j = 0; j < N; j++) {
                    cacheData[j] = beta * cacheData[j] + (1 - beta) * gradData[j] * gradData[j];
                    paramData[j] -= learningRate * gradData[j] / ((float)Math.sqrt(cacheData[j]) + epsilon);
                }
                parameters[i].setData(paramData);
                cache[i].setData(cacheData);
            }
        }
    }
}
