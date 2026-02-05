package minigptj.core;

/**
 * ReLU activation layer.
 *
 * Forward:  max(0, x)
 * Backward: passes gradient through where x > 0, blocks it where x <= 0.
 *
 * We store the last input so backward() knows which values were "active".
 */
public class ReLU {
    private Matrix lastInput;

    /**
     * Applies ReLU element-wise and caches the input for backprop.
     */
    public Matrix forward(Matrix x) {
        this.lastInput = x;
        return x.apply(v -> Math.max(0.0, v));
    }

    /**
     * Backprop through ReLU.
     *
     * @param dOut upstream gradient (same shape as forward output)
     * @return gradient w.r.t the input of ReLU
     */
    public Matrix backward(Matrix dOut) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward() before backward().");
        }

        // Same shape as input/output
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
