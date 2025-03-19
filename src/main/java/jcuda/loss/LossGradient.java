package jcuda.loss;

import jcuda.core.Tensor;

public interface LossGradient {
    /**
     * Backward pass: Computes the gradient of the loss with respect to the predictions.
     *
     * @param predictions The model predictions.
     * @param targets The ground truth values.
     * @return A Tensor containing the gradient with respect to the predictions.
     */
    Tensor backward(Tensor predictions, Tensor targets);
}
