package minigptj.core;

/**
 * Rectified Linear Unit (ReLU) activation layer.
 *
 * Forward pass:
 *     f(x) = max(0, x)
 *
 * Backward pass:
 *     gradients pass through unchanged where x > 0
 *     gradients are blocked where x <= 0
 *
 * The input from the most recent forward pass is cached so the backward
 * pass can determine which neurons were active.
 */
public class ReLU {
    private Matrix lastInput;

    /**
     * Applies ReLU element-wise.
     *
     * @param x input matrix
     * @return activated output matrix
     */
    public Matrix forward(Matrix x) {
        this.lastInput = x;
        return x.apply(v -> Math.max(0.0, v));
    }

    /**
     * Backpropagates gradients through the ReLU activation.
     *
     * Gradients are passed through only for elements whose original
     * input value was greater than zero.
     *
     * @param dOut upstream gradient from the next layer
     * @return gradient with respect to the ReLU input
     */
    public Matrix backward(Matrix dOut) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward() before backward().");
        }

        Matrix dX = new Matrix(dOut.getRows(), dOut.getCols());

        for (int i = 0; i < dOut.getRows(); i++) {
            for (int j = 0; j < dOut.getCols(); j++) {
                // If the input was <= 0, ReLU output was 0 and gradient is blocked.
                // If input was > 0, gradient passes through unchanged.
                double grad = (lastInput.get(i, j) > 0.0) ? dOut.get(i, j) : 0.0;
                dX.set(i, j, grad);
            }
        }
        return dX;
    }
}
