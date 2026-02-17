package minigptj.core;

/**
 * A simple two-layer feed-forward neural network:
 *
 *   input -> Linear -> ReLU -> Linear -> (logits)
 *
 * This class currently supports forward-only computation.
 * Training (backpropagation) will be implemented separately.
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
     * @param outputSize number of output units (e.g. number of classes)
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
     * Performs a full forward pass and returns probabilities via softmax.
     *
     * @param input a Matrix of shape (batchSize x inputSize)
     * @return probabilities with shape (batchSize x outputSize),
     *         where each row sums to 1.
     */
    public Matrix forwardWithSoftmax(Matrix input) {
        Matrix logits = forward(input);
        return logits.softmaxRows();
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getOutputSize() {
        return outputSize;
    }
}
