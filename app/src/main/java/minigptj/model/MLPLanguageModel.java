package minigptj.model;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.core.ReLU;

/**
 * Baseline multilayer perceptron language model.
 *
 * This model was used as an early next-token prediction baseline before the
 * transformer attention block was introduced. It takes a fixed-size input vector,
 * passes it through one hidden layer with ReLU activation, and outputs logits over
 * the vocabulary.
 *
 * Architecture:
 *   input -> Linear -> ReLU -> Linear -> logits
 */
public class MLPLanguageModel {
    private final Linear layer1;
    private final ReLU relu;
    private final Linear layer2;

    // cached forward activations (for backward flow)
    private Matrix lastHidden;

    /**
     * Creates a two-layer MLP language model.
     *
     * @param inputSize  number of input features
     * @param hiddenSize number of hidden units in the intermediate layer
     * @param outputSize number of output classes, usually the vocabulary size
     */
    public MLPLanguageModel(int inputSize, int hiddenSize, int outputSize) {
        this.layer1 = new Linear(inputSize, hiddenSize);
        this.relu = new ReLU();
        this.layer2 = new Linear(hiddenSize, outputSize);
    }

    /**
     * Performs a forward pass through the model.
     *
     * The returned values are logits, not probabilities. Softmax is applied
     * separately during loss calculation or prediction.
     *
     * @param x input matrix of shape batchSize x inputSize
     * @return logits of shape batchSize x outputSize
     */
    public Matrix forward(Matrix x) {
        Matrix h = layer1.forward(x);
        h = relu.forward(h);
        this.lastHidden = h;
        return layer2.forward(h);
    }

    /**
     * Performs backpropagation through the MLP.
     *
     * The gradient flows backwards through:
     *   output Linear -> ReLU -> input Linear
     *
     * @param dLogits upstream gradient of the loss with respect to the logits
     * @return gradient of the loss with respect to the model input
     */
    public Matrix backward(Matrix dLogits) {
        Matrix dH = layer2.backward(dLogits);
        dH = relu.backward(dH);
        Matrix dX = layer1.backward(dH);
        return dX;
    }

    /**
     * Returns the first Linear layer.
     *
     * @return input-to-hidden layer
     */
    public Linear getLayer1() { return layer1; }

    /**
     * Returns the second Linear layer.
     *
     * @return hidden-to-output layer
     */
    public Linear getLayer2() { return layer2; }
}
