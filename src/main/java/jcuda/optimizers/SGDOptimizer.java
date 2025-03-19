package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.core.Tensor;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.runtime.JCuda.*;

public class SGDOptimizer extends Optimizer {

    public SGDOptimizer(float learningRate) {
        super(learningRate);
    }

    /**
     * Updates parameters using vanilla Stochastic Gradient Descent.
     * If both parameter and gradient tensors have a valid GPU pointer (via getGpuData()),
     * the JCuda kernel is launched; otherwise, CPU fallback logic is used.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Number of parameters and gradients must match.");

        for (int i = 0; i < parameters.length; i++) {
            int N = parameters[i].getData().length;
            if (parameters[i].getGpuData() != null && gradients[i].getGpuData() != null) {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                Pointer kernelParams = Pointer.to(
                        Pointer.to(parameters[i].getGpuData()),
                        Pointer.to(gradients[i].getGpuData()),
                        Pointer.to(new float[]{learningRate}),
                        Pointer.to(new int[]{N})
                );
                cuLaunchKernel(OptimizerKernels.sgdUpdateFunction,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, null,
                        kernelParams, null);
            } else {
                // CPU fallback
                float[] paramData = parameters[i].getData();
                float[] gradData = gradients[i].getData();
                for (int j = 0; j < N; j++) {
                    paramData[j] -= learningRate * gradData[j];
                }
                parameters[i].setData(paramData);
            }
        }
    }
}
