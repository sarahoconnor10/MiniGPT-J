package minigptj.optim;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.model.Embedding;

/**
 * Stochastic Gradient Descent (SGD) optimiser.
 *
 * SGD updates model parameters using the gradient of the loss with respect
 * to each parameter. A fixed learning rate is applied uniformly across all
 * parameters.
 *
 * Update rule:
 *
 *   parameter = parameter - learningRate * gradient
 *
 * This optimiser does not compute gradients itself. Gradients are expected
 * to be produced during backpropagation by the corresponding layer.
 */
public class SGD {
    private final double learningRate;

    /**
     * Creates a new SGD optimiser.
     *
     * @param learningRate the step size used when updating parameters
     */
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }
    /**
     * Applies one optimisation step to a Linear layer.
     *
     * Both the weight matrix and bias vector are updated in-place using
     * gradients computed during backpropagation.
     *
     * This method assumes:
     * - forward() has already been called
     * - backward() has already been called
     *
     * @param layer Linear layer to update
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
     * The embedding matrix is updated in-place using accumulated embedding
     * gradients from the backward pass.
     *
     * @param emb embedding layer to update
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
