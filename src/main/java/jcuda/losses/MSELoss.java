package jcuda.losses;

import jcuda.core.Tensor;

public class MSELoss extends Loss {

    /**
     * Computes the Mean Squared Error (MSE):
     * loss = (1/N) * sum((prediction - target)^2)
     */
    @Override
    public float forward(Tensor predictions, Tensor targets) {
        float[] predData = predictions.getData();
        float[] targetData = targets.getData();
        if (predData.length != targetData.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of elements.");
        }
        float sum = 0.0f;
        int n = predData.length;
        for (int i = 0; i < n; i++) {
            float diff = predData[i] - targetData[i];
            sum += diff * diff;
        }
        return sum / n;
    }
}
