package jcuda.optimizers;

import jcuda.core.Tensor;

public class AdagradOptimizer extends Optimizer {
    private float epsilon;
    private Tensor[] sumSquares; // Sum of squares of gradients for each parameter

    public AdagradOptimizer(float learningRate, float epsilon) {
        super(learningRate);
        this.epsilon = epsilon;
    }

    /**
     * Performs parameter update using Adagrad.
     * Update rule:
     *   sumSquares = sumSquares + gradient^2
     *   parameter = parameter - learningRate * gradient / (sqrt(sumSquares) + epsilon)
     *
     * @param parameters Array of model parameters.
     * @param gradients Array of gradients corresponding to each parameter.
     */
    @Override
    public void update(Tensor[] parameters, Tensor[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("The number of parameters must match the number of gradients.");
        }
        if (sumSquares == null) {
            sumSquares = new Tensor[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                float[] zeros = new float[parameters[i].getData().length];
                sumSquares[i] = new Tensor(zeros, parameters[i].getShape());
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            float[] paramData = parameters[i].getData();
            float[] gradData = gradients[i].getData();
            float[] sumSqData = sumSquares[i].getData();

            if (paramData.length != gradData.length) {
                throw new IllegalArgumentException("Parameter and gradient dimensions must match.");
            }

            for (int j = 0; j < paramData.length; j++) {
                sumSqData[j] += gradData[j] * gradData[j];
                paramData[j] -= learningRate * gradData[j] / ((float)Math.sqrt(sumSqData[j]) + epsilon);
            }

            parameters[i].setData(paramData);
            sumSquares[i].setData(sumSqData);
        }
    }
}
