package jcuda.optimizers;

import jcuda.core.Tensor;

public class AdamWOptimizer extends Optimizer {
    private float beta1;
    private float beta2;
    private float epsilon;
    private float weightDecay;
    private Tensor[] m; // First moment estimates
    private Tensor[] v; // Second moment estimates
    private int t;       // Time step

    /**
     * Constructs an AdamW optimizer.
     *
     * @param learningRate The learning rate.
     * @param beta1 The exponential decay rate for the first moment estimates.
     * @param beta2 The exponential decay rate for the second moment estimates.
     * @param epsilon A small constant for numerical stability.
     * @param weightDecay The weight decay (L2 regularization) factor.
     */
    public AdamWOptimizer(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.t = 0;
    }

    /**
     * Performs a parameter update using the AdamW optimization algorithm.
     * AdamW decouples weight decay from the gradient-based update.
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("The number of parameters must match the number of gradients.");
        }

        // Initialize moment estimates if this is the first update
        if (m == null || v == null) {
            m = new Tensor[parameters.length];
            v = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                m[i] = new Tensor(zeros, parameters[i].getShape());
                v[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        t++; // Increment time step

        for (int i = 0; i < parameters.length; i++) {
            float[] paramData = parameters[i].getData();
            float[] gradData = gradients[i].getData();
            float[] mData = m[i].getData();
            float[] vData = v[i].getData();

            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }

            for (int j = 0; j < paramData.length; j++) {
                mData[j] = beta1 * mData[j] + (1 - beta1) * gradData[j];
                vData[j] = beta2 * vData[j] + (1 - beta2) * gradData[j] * gradData[j];
                float mHat = mData[j] / (1 - (float)Math.pow(beta1, t));
                float vHat = vData[j] / (1 - (float)Math.pow(beta2, t));
                // Update parameter using decoupled weight decay
                paramData[j] -= learningRate * (mHat / ((float)Math.sqrt(vHat) + epsilon) + weightDecay * paramData[j]);
            }
            parameters[i].setData(paramData);
            m[i].setData(mData);
            v[i].setData(vData);
        }
    }
}
