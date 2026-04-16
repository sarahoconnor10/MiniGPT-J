package minigptj.optim;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.model.Embedding;
import java.util.HashMap;
import java.util.Map;

public class Adam {
    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private int t = 0;

    private final Map<Matrix, double[][]> m1 = new HashMap<>();
    private final Map<Matrix, double[][]> m2 = new HashMap<>();

    public Adam(double lr) {
        this(lr, 0.9, 0.999, 1e-8);
    }

    public Adam(double lr, double beta1, double beta2, double eps) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    public void step(Linear layer) {
        update(layer.getWeights(), layer.getGradWeights());
        update(layer.getBias(),    layer.getGradBias());
    }

    public void step(Embedding emb) {
        update(emb.getWeights(), emb.getGradWeights());
    }

    private void update(Matrix param, Matrix grad) {
        if (grad == null) throw new IllegalStateException("Grad is null — call backward() first.");

        double[][] firstMoment  = m1.computeIfAbsent(param,
            k -> new double[param.getRows()][param.getCols()]);
        double[][] secondMoment = m2.computeIfAbsent(param,
            k -> new double[param.getRows()][param.getCols()]);

        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);

        for (int i = 0; i < param.getRows(); i++) {
            for (int j = 0; j < param.getCols(); j++) {
                double g = grad.get(i, j);
                firstMoment[i][j]  = beta1 * firstMoment[i][j]  + (1 - beta1) * g;
                secondMoment[i][j] = beta2 * secondMoment[i][j] + (1 - beta2) * g * g;
                double mHat = firstMoment[i][j]  / bc1;
                double vHat = secondMoment[i][j] / bc2;
                param.set(i, j, param.get(i, j) - lr * mHat / (Math.sqrt(vHat) + eps));
            }
        }
    }

    public void tick() { t++; }
}