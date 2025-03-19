package jcuda.optimizers;

import jcuda.core.Tensor;

public class RMSPropOptimizer extends Optimizer {
    private float beta;
    private float epsilon;
    private Tensor[] cache; // Exponential moving average of squared gradients

    public RMSPropOptimizer(float learningRate, float beta, float epsilon) {
        super(learningRate);
        this.beta = beta;
        this.epsilon = epsilon;
    }

    /**
     * Performs parameter update using RMSProp.
     * Update rule:
     *   cache = beta * cache + (1 - beta) * gradient^2
     *   parameter = parameter - learningRate * gradient / (sqrt(cache) + epsilon)
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("The number of parameters must match the number of gradients.");
        }
        if (cache == null) {
            cache = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                cache[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            float[] paramData = parameters[i].getData();
            float[] gradData = gradients[i].getData();
            float[] cacheData = cache[i].getData();

            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }

            for (int j = 0; j < paramData.length; j++) {
                cacheData[j] = beta * cacheData[j] + (1 - beta) * gradData[j] * gradData[j];
                paramData[j] -= learningRate * gradData[j] / ((float)Math.sqrt(cacheData[j]) + epsilon);
            }

            parameters[i].setData(paramData);
            cache[i].setData(cacheData);
        }
    }
}
