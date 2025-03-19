package jcuda.losses;

import jcuda.core.Tensor;

public class CrossEntropyLoss extends Loss {

    /**
     * Computes the Cross-Entropy Loss:
     * loss = -(1/N) * sum(target * log(prediction + epsilon))
     */
    @Override
    public float forward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        float loss = 0.0f;
        int n = predData.length;
        float epsilon = 1e-7f;
        for (int i = 0; i < n; i++) {
            loss -= targetData[i] * Math.log(predData[i] + epsilon);
        }
        return loss / n;
    }
}
