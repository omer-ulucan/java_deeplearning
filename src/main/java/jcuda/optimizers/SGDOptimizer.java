package jcuda.optimizers;

import jcuda.core.Tensor;

public class SGDOptimizer extends Optimizer {

    public SGDOptimizer(float learningRate) {
        super(learningRate);
    }

    /**
     * Performs parameter update using vanilla Stochastic Gradient Descent (SGD).
     * For each parameter: param = param - learningRate * gradient.
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("The number of parameters must match the number of gradients.");
        }
        for (int i = 0; i < parameters.length; i++) {
            float[] paramData = parameters[i].getData();
            float[] gradData = gradients[i].getData();
            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }
            for (int j = 0; j < paramData.length; j++) {
                paramData[j] -= learningRate * gradData[j];
            }
            parameters[i].setData(paramData);
        }
    }
}
