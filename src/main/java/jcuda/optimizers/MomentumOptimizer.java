package jcuda.optimizers;

import jcuda.core.Tensor;

public class MomentumOptimizer extends Optimizer {
    private float momentum;
    private Tensor[] velocities; // Stores the velocity for each parameter

    public MomentumOptimizer(float learningRate, float momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    /**
     * Performs parameter update using SGD with Momentum.
     * Update rule:
     *   velocity = momentum * velocity - learningRate * gradient
     *   parameter = parameter + velocity
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("The number of parameters must match the number of gradients.");
        }
        if (velocities == null) {
            velocities = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                velocities[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            float[] paramData = parameters[i].getData();
            float[] gradData = gradients[i].getData();
            float[] velocityData = velocities[i].getData();

            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }

            for (int j = 0; j < paramData.length; j++) {
                velocityData[j] = momentum * velocityData[j] - learningRate * gradData[j];
                paramData[j] += velocityData[j];
            }

            parameters[i].setData(paramData);
            velocities[i].setData(velocityData);
        }
    }
}
