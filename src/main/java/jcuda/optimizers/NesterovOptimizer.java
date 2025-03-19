package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.core.Tensor;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class NesterovOptimizer extends Optimizer {
    private final float momentum;
    private Tensor[] velocities;

    public NesterovOptimizer(float learningRate, float momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    /**
     * Updates parameters using Nesterov Accelerated Gradient.
     * If possible, uses the JCuda kernel; otherwise, uses CPU fallback.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Number of parameters and gradients must match.");

        if (velocities == null) {
            velocities = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                velocities[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            int N = parameters[i].getData().length;
            if (parameters[i].getGpuData() != null && gradients[i].getGpuData() != null &&
                    velocities[i].getGpuData() != null) {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                Pointer kernelParams = Pointer.to(
                        Pointer.to(parameters[i].getGpuData()),
                        Pointer.to(gradients[i].getGpuData()),
                        Pointer.to(velocities[i].getGpuData()),
                        Pointer.to(new float[]{learningRate}),
                        Pointer.to(new float[]{momentum}),
                        Pointer.to(new int[]{N})
                );
                cuLaunchKernel(OptimizerKernels.nesterovUpdateFunction,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, null,
                        kernelParams, null);
            } else {
                // CPU fallback
                float[] paramData = parameters[i].getData();
                float[] gradData = gradients[i].getData();
                float[] velocityData = velocities[i].getData();
                float[] prevVelocity = velocityData.clone();
                for (int j = 0; j < N; j++) {
                    velocityData[j] = momentum * velocityData[j] - learningRate * gradData[j];
                    paramData[j] += -momentum * prevVelocity[j] + (1 + momentum) * velocityData[j];
                }
                parameters[i].setData(paramData);
                velocities[i].setData(velocityData);
            }
        }
    }
}
