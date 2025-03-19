package jcuda.optimizers;

import jcuda.core.Tensor;

public class NesterovOptimizer extends Optimizer {
    private float momentum;
    private Tensor[] velocities; // Stores the velocity for each parameter

    public NesterovOptimizer(float learningRate, float momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    /**
     * Performs parameter update using Nesterov Accelerated Gradient.
     * Update rule:
     *   prevVelocity = velocity (before update)
     *   velocity = momentum * velocity - learningRate * gradient
     *   parameter = parameter - momentum * prevVelocity + (1 + momentum) * velocity
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
            float[] prevVelocity = velocityData.clone();

            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }

            for (int j = 0; j < paramData.length; j++) {
                velocityData[j] = momentum * velocityData[j] - learningRate * gradData[j];
                paramData[j] += -momentum * prevVelocity[j] + (1 + momentum) * velocityData[j];
            }

            parameters[i].setData(paramData);
            velocities[i].setData(velocityData);
        }
    }
}
