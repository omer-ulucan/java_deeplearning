package jcuda.losses;

import jcuda.core.Tensor;

public class MSELossGradient implements LossGradient {

    /**
     * Computes the gradient for Mean Squared Error (MSE):
     * dL/dprediction = 2 * (prediction - target) / N
     */
    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        int n = predData.length;
        float[] gradData = new float[n];
        for (int i = 0; i < n; i++) {
            gradData[i] = 2 * (predData[i] - targetData[i]) / n;
        }
        return new Tensor(gradData, predictions.getShape());
    }
}
