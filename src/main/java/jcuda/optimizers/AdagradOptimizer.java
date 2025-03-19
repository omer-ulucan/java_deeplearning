package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.core.Tensor;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AdagradOptimizer extends Optimizer {
    private final float epsilon;
    private Tensor[] sumSquares;

    public AdagradOptimizer(float learningRate, float epsilon) {
        super(learningRate);
        this.epsilon = epsilon;
    }

    /**
     * Updates parameters using Adagrad.
     * If tensors are on the GPU, uses the JCuda kernel; otherwise, uses CPU fallback.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Number of parameters and gradients must match.");

        if (sumSquares == null) {
            sumSquares = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                sumSquares[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            int N = parameters[i].getData().length;
            if (parameters[i].getGpuData() != null && gradients[i].getGpuData() != null &&
                    sumSquares[i].getGpuData() != null) {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                Pointer kernelParams = Pointer.to(
                        Pointer.to(parameters[i].getGpuData()),
                        Pointer.to(gradients[i].getGpuData()),
                        Pointer.to(sumSquares[i].getGpuData()),
                        Pointer.to(new float[]{learningRate}),
                        Pointer.to(new float[]{epsilon}),
                        Pointer.to(new int[]{N})
                );
                cuLaunchKernel(OptimizerKernels.adagradUpdateFunction,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, null,
                        kernelParams, null);
            } else {
                // CPU fallback
                float[] paramData = parameters[i].getData();
                float[] gradData = gradients[i].getData();
                float[] sumSqData = sumSquares[i].getData();
                for (int j = 0; j < N; j++) {
                    sumSqData[j] += gradData[j] * gradData[j];
                    paramData[j] -= learningRate * gradData[j] / ((float)Math.sqrt(sumSqData[j]) + epsilon);
                }
                parameters[i].setData(paramData);
                sumSquares[i].setData(sumSqData);
            }
        }
    }
}
