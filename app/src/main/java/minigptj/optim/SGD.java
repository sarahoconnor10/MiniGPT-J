package minigptj.optim;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.model.Embedding;

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 *
 * This class is responsible for updating the parameters of a trainable layer
 * using gradients computed during backpropagation.
 *
 * SGD applies the update rule:
 *
 *   parameter = parameter - learningRate * gradient
 *
 * The optimizer does not compute gradients itself — it only applies them.
 * This separation keeps layer logic independent from training strategy.
 */
public class SGD {
    private final double learningRate;

    /**
     * Create a new SGD optimizer.
     *
     * @param learningRate the step size used when updating parameters
     */
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }
    /**
     * Apply one optimisation step to a Linear layer.
     *
     * This method assumes that:
     *  1) forward() has been called to cache inputs
     *  2) backward() has been called to compute gradients
     *
     * It then updates the layer's weights and bias in-place using SGD.
     *
     * @param layer the Linear layer whose parameters should be updated
     */
    public void step(Linear layer) {
        Matrix w = layer.getWeights();
        Matrix b = layer.getBias();

        Matrix dW = layer.getGradWeights();
        Matrix dB = layer.getGradBias();

        if (dW == null || dB == null) {
            throw new IllegalStateException("Gradients are null. Call forward() and backward() before step().");
        }

        // W = W - lr * dW
        for (int i = 0; i < w.getRows(); i++) {
            for (int j = 0; j < w.getCols(); j++) {
                double updated = w.get(i, j) - learningRate * dW.get(i, j);
                w.set(i, j, updated);
            }
        }

        // b = b - lr * db
        for (int j = 0; j < b.getCols(); j++) {
            double updated = b.get(0, j) - learningRate * dB.get(0, j);
            b.set(0, j, updated);
        }
    }


    /**
     * Apply one optimisation step to an Embedding layer.
     *
     * weights = weights - lr * gradWeights
     */
    public void step(Embedding emb) {
        Matrix w = emb.getWeights();
        Matrix dW = emb.getGradWeights();

        if (dW == null) {
            throw new IllegalStateException("Embedding gradients are null. Call forward() and backward() before step().");
        }

        for (int i = 0; i < w.getRows(); i++) {
            for (int j = 0; j < w.getCols(); j++) {
                w.set(i, j, w.get(i, j) - learningRate * dW.get(i, j));
            }
        }
    }

}
