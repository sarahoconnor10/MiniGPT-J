package minigptj.core;

/**
 * Simple two-layer multilayer perceptron (MLP).
 *
 * Architecture:
 *
 *     input -> Linear -> ReLU -> Linear -> logits
 *
 * This class was developed as an early neural network architecture
 * before the transformer-based language model was introduced.
 *
 * It provides a simpler baseline model for validating:
 * - matrix operations
 * - forward propagation
 * - activation behaviour
 * - output generation
 */
public class SimpleMLP {
    private final Linear layer1;
    private final Linear layer2;

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    /**
     * Constructs a simple MLP with one hidden layer.
     *
     * @param inputSize  number of input features
     * @param hiddenSize number of hidden units in the intermediate layer
     * @param outputSize number of output units
     */
    public SimpleMLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.layer1 = new Linear(inputSize, hiddenSize);
        this.layer2 = new Linear(hiddenSize, outputSize);
    }

    /**
     * Performs a forward pass through the MLP.
     *
     * @param input a matrix of shape (batchSize x inputSize)
     * @return the output logits matrix of shape (batchSize x outputSize)
     */
    public Matrix forward(Matrix input) {
        // 1. First linear layer
        Matrix h = layer1.forward(input);

        // 2. ReLU activation
        h = h.apply(Activation::relu);

        // 3. Second linear layer (logits, not yet softmaxed)
        Matrix logits = layer2.forward(h);
        return logits;
    }

    /**
     * Performs a full forward pass followed by row-wise softmax.
     *
     * Converts logits into probability distributions where each row
     * sums to 1.
     *
     * @param input matrix of shape batchSize x inputSize
     * @return probability matrix of shape batchSize x outputSize
     */
    public Matrix forwardWithSoftmax(Matrix input) {
        Matrix logits = forward(input);
        return logits.softmaxRows();
    }

    /**
     * Returns the expected input feature count.
     *
     * @return input size
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Returns the hidden layer size.
     *
     * @return hidden size
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * Returns the output feature count.
     *
     * @return output size
     */
    public int getOutputSize() {
        return outputSize;
    }
}
