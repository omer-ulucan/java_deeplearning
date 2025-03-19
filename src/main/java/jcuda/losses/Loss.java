package jcuda.losses;

import jcuda.core.Tensor;

public abstract class Loss {
    /**
     * Forward pass: Computes the loss value between predictions and targets.
     * Uses JCuda kernels if data is on GPU; otherwise, uses CPU fallback.
     *
     * @param predictions The model predictions.
     * @param targets The ground truth values.
     * @return The computed loss value.
     */
    public abstract float forward(Tensor predictions, Tensor targets);
}
