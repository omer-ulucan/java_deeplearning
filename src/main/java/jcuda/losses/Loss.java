package jcuda.losses;

import jcuda.core.Tensor;

public abstract class Loss {
    /**
     * Forward pass: Computes the loss value between the predictions and targets.
     *
     * @param predictions The model predictions.
     * @param targets The ground truth values.
     * @return The computed loss value.
     */
    public abstract float forward(Tensor predictions, Tensor targets);
}
