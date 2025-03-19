package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.core.Tensor;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AdamWOptimizer extends Optimizer {
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private Tensor[] m; // first moment estimates
    private Tensor[] v; // second moment estimates
    private int t;      // time step

    public AdamWOptimizer(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.t = 0;
    }

    /**
     * Updates parameters using the AdamW algorithm (Adam with decoupled weight decay).
     * If possible, uses the JCuda kernel; otherwise, falls back to CPU logic.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Number of parameters and gradients must match.");

        if (m == null || v == null) {
            m = new Tensor[parameters.length];
            v = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                m[i] = new Tensor(zeros, parameters[i].getShape());
                v[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }
        t++;
        for (int i = 0; i < parameters.length; i++) {
            int N = parameters[i].getData().length;
            if (parameters[i].getGpuData() != null && gradients[i].getGpuData() != null &&
                    m[i].getGpuData() != null && v[i].getGpuData() != null) {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                Pointer kernelParams = Pointer.to(
                        Pointer.to(parameters[i].getGpuData()),
                        Pointer.to(gradients[i].getGpuData()),
                        Pointer.to(m[i].getGpuData()),
                        Pointer.to(v[i].getGpuData()),
                        Pointer.to(new float[]{learningRate}),
                        Pointer.to(new float[]{beta1}),
                        Pointer.to(new float[]{beta2}),
                        Pointer.to(new float[]{epsilon}),
                        Pointer.to(new float[]{weightDecay}),
                        Pointer.to(new int[]{t}),
                        Pointer.to(new int[]{N})
                );
                cuLaunchKernel(OptimizerKernels.adamWUpdateFunction,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, null,
                        kernelParams, null);
            } else {
                // CPU fallback
                float[] paramData = parameters[i].getData();
                float[] gradData = gradients[i].getData();
                float[] mData = m[i].getData();
                float[] vData = v[i].getData();
                for (int j = 0; j < N; j++) {
                    mData[j] = beta1 * mData[j] + (1 - beta1) * gradData[j];
                    vData[j] = beta2 * vData[j] + (1 - beta2) * gradData[j] * gradData[j];
                    float mHat = mData[j] / (1 - (float)Math.pow(beta1, t));
                    float vHat = vData[j] / (1 - (float)Math.pow(beta2, t));
                    paramData[j] -= learningRate * (mHat / ((float)Math.sqrt(vHat) + epsilon) + weightDecay * paramData[j]);
                }
                parameters[i].setData(paramData);
                m[i].setData(mData);
                v[i].setData(vData);
            }
        }
    }
}
