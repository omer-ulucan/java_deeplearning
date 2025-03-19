package jcuda.losses;

import jcuda.core.Tensor;

public interface LossGradient {
    /**
     * Backward pass: Computes the gradient of the loss with respect to predictions.
     * Uses JCuda kernels if available; otherwise, uses CPU fallback.
     *
     * @param predictions The model predictions.
     * @param targets The ground truth values.
     * @return A Tensor containing the computed gradients.
     */
    Tensor backward(Tensor predictions, Tensor targets);
}
