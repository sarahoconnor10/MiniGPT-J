package minigptj.core;

import java.util.Random;

/**
 * Fully connected neural network layer.
 *
 * Computes:
 *     output = input * weights + bias
 *
 * This class stores the most recent input during the forward pass so that
 * gradients can be calculated manually during backpropagation.
 */
public class Linear {

    private final int inputSize;
    private final int outputSize;

    private final Matrix weights;   // shape: inputSize x outputSize
    private final Matrix bias;      // shape: 1 x outputSize

    private Matrix lastInput;
    private Matrix gradWeights;
    private Matrix gradBias;

    /**
     * Creates a linear layer with randomly initialised weights and zero bias.
     *
     * Weights are initialised using a Xavier-style standard deviation
     * to help keep activations stable during training.
     *
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public Linear(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        Random rand = new Random();
        weights = new Matrix(inputSize, outputSize);
        bias = new Matrix(1, outputSize);

        double std = Math.sqrt(2.0 / (inputSize + outputSize));

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.set(i, j, rand.nextGaussian() * std);
            }
        }
    }

    /**
     * Forward pass through the layer.
     *
     * @param input matrix of shape batchSize x inputSize
     * @return output matrix of shape batchSize x outputSize
     */
    public Matrix forward(Matrix input) {
        this.lastInput = input;

        Matrix out = input.dot(weights);  // (batchSize x outputSize)

        int rows = out.getRows();
        int cols = out.getCols();

        // Add the same bias row to each row in the batch.
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = out.get(i, j) + bias.get(0, j);
                out.set(i, j, value);
            }
        }

        return out;
    }

    /**
     * Backward pass through the layer.
     *
     * Calculates gradients for weights and bias, and returns the gradient
     * with respect to the input so earlier layers can continue backpropagation.
     *
     * @param dOut upstream gradient of shape batchSize x outputSize
     * @return gradient with respect to input, shape batchSize x inputSize
     */
    public Matrix backward(Matrix dOut) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward() before backward().");
        }

        // 1) dX = dOut * W^T
        Matrix dX = dOut.dot(weights.transpose());

        // 2) dW = X^T * dOut
        this.gradWeights = lastInput.transpose().dot(dOut);

        // 3) db = sum over batch rows
        this.gradBias = new Matrix(1, outputSize);
        for (int j = 0; j < outputSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < dOut.getRows(); i++) {
                sum += dOut.get(i, j);
            }
            gradBias.set(0, j, sum);
        }

        return dX;
    }

    /**
     * Returns the trainable weight matrix.
     *
     * @return weights with shape inputSize x outputSize
     */
    public Matrix getWeights() {
        return weights;
    }

    /**
     * Returns the trainable bias row.
     *
     * @return bias with shape 1 x outputSize
     */
    public Matrix getBias() {
        return bias;
    }

    /**
     * Returns the most recent weight gradients.
     *
     * @return gradient matrix for the weights
     */
    public Matrix getGradWeights() {
        return gradWeights;
    }

    /**
     * Returns the most recent bias gradients.
     *
     * @return gradient row for the bias
     */
    public Matrix getGradBias() {
        return gradBias;
    }
}
