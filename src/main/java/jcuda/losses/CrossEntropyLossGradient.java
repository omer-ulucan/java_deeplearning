package jcuda.losses;

import jcuda.core.Tensor;

public class CrossEntropyLossGradient implements LossGradient {

    /**
     * Computes the gradient for Cross-Entropy Loss:
     * dL/dprediction = -(target / (prediction + epsilon)) / N
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
        float epsilon = 1e-7f;
        for (int i = 0; i < n; i++) {
            gradData[i] = -targetData[i] / (predData[i] + epsilon) / n;
        }
        return new Tensor(gradData, predictions.getShape());
    }
}
