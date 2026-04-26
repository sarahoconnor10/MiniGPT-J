package minigptj.optim;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.model.Embedding;
import java.util.HashMap;
import java.util.Map;

/**
 * Adam (Adaptive Moment Estimation) optimiser.
 *
 * Adam extends stochastic gradient descent by maintaining:
 * - a first moment estimate (moving average of gradients)
 * - a second moment estimate (moving average of squared gradients)
 *
 * These statistics are used to compute adaptive per-parameter learning rates,
 * which helps stabilise training when different parameters receive gradients of
 * very different magnitudes.
 *
 * Update rule:
 *
 *   m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
 *   v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
 *
 * After bias correction:
 *
 *   param = param - lr * mHat / (sqrt(vHat) + eps)
 */
public class Adam {
    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;

    // Optimisation timestep used for bias correction.
    private int t = 0;

    /**
     * First moment estimates (mean of gradients) for each parameter matrix.
     */
    private final Map<Matrix, double[][]> m1 = new HashMap<>();
    /**
     * Second moment estimates (mean of squared gradients) for each parameter matrix.
     */
    private final Map<Matrix, double[][]> m2 = new HashMap<>();

    /**
     * Creates an Adam optimiser using standard default hyperparameters.
     *
     * Defaults:
     *   beta1 = 0.9
     *   beta2 = 0.999
     *   eps   = 1e-8
     *
     * @param lr learning rate
     */
    public Adam(double lr) {
        this(lr, 0.9, 0.999, 1e-8);
    }

    /**
     * Creates an Adam optimiser with custom hyperparameters.
     *
     * @param lr learning rate
     * @param beta1 exponential decay rate for first moment estimates
     * @param beta2 exponential decay rate for second moment estimates
     * @param eps small constant added for numerical stability
     */
    public Adam(double lr, double beta1, double beta2, double eps) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    /**
     * Applies one optimisation step to a Linear layer.
     *
     * Both the weight matrix and bias vector are updated in-place.
     *
     * @param layer Linear layer to update
     */
    public void step(Linear layer) {
        update(layer.getWeights(), layer.getGradWeights());
        update(layer.getBias(),    layer.getGradBias());
    }

    /**
     * Applies one optimisation step to an Embedding layer.
     *
     * @param emb embedding layer to update
     */
    public void step(Embedding emb) {
        update(emb.getWeights(), emb.getGradWeights());
    }

    /**
     * Updates a parameter matrix using the Adam update rule.
     *
     * Moment statistics are stored separately for each parameter matrix.
     *
     * @param param parameter matrix to update
     * @param grad gradient matrix for the parameter
     */
    private void update(Matrix param, Matrix grad) {
        if (grad == null) throw new IllegalStateException("Grad is null; call backward() first.");

        // initialise moment buffers lazily on first use
        double[][] firstMoment  = m1.computeIfAbsent(param,
            k -> new double[param.getRows()][param.getCols()]);
        double[][] secondMoment = m2.computeIfAbsent(param,
            k -> new double[param.getRows()][param.getCols()]);

        // bias correction terms
        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);

        for (int i = 0; i < param.getRows(); i++) {
            for (int j = 0; j < param.getCols(); j++) {
                double g = grad.get(i, j);
                // update moving averages
                firstMoment[i][j]  = beta1 * firstMoment[i][j]  + (1 - beta1) * g;
                secondMoment[i][j] = beta2 * secondMoment[i][j] + (1 - beta2) * g * g;

                // bias-corrected estimates
                double mHat = firstMoment[i][j]  / bc1;
                double vHat = secondMoment[i][j] / bc2;

                // parameter update
                param.set(i, j, param.get(i, j) - lr * mHat / (Math.sqrt(vHat) + eps));
            }
        }
    }

    /**
     * Advances the optimiser timestep.
     *
     * This should be called once per training step before parameter updates
     * so that bias correction is computed correctly.
     */
    public void tick() {
        t++;
    }
}
