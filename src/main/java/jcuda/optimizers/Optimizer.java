package jcuda.optimizers;

import jcuda.core.Tensor;

public abstract class Optimizer {
    protected float learningRate;

    public Optimizer(float learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Update model parameters based on computed gradients.
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    public abstract void update(Tensor[] parameters, Tensor[] gradients);
}
