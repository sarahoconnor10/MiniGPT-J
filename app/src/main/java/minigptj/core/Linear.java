package minigptj.core;

import java.util.Random;

public class Linear {

    private final int inputSize;
    private final int outputSize;
    private final Matrix weights;
    private final Matrix bias;
    private Matrix lastInput;
    private Matrix gradWeights;
    private Matrix gradBias;
    
    public Linear(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        Random rand = new Random();

        weights = new Matrix(inputSize, outputSize);

        // random small initialisation
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.set(i, j, rand.nextGaussian() * 0.01);
            }
        }

        // bias is a 1 x outputSize matrix
        bias = new Matrix(1, outputSize);
    }

    public Matrix forward(Matrix input) {
        this.lastInput = input;
        // input shape: (batchSize x inputSize)
        Matrix out = input.dot(weights);  // (batchSize x outputSize)

        int rows = out.getRows();
        int cols = out.getCols();

        // Manually add the bias row to each row of the output
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = out.get(i, j) + bias.get(0, j);
                out.set(i, j, value);
            }
        }

        return out;
    }

    public Matrix backward(Matrix dOut) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward() before backward().");
        }
        // dOut: (batchSize x outputSize)

        // 1) dX = dOut * W^T
        Matrix dX = dOut.dot(weights.transpose()); // (batchSize x inputSize)

        // 2) dW = X^T * dOut
        this.gradWeights = lastInput.transpose().dot(dOut); // (inputSize x outputSize)

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

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBias() {
        return bias;
    }

    public Matrix getGradWeights() {
        return gradWeights;
     }

    public Matrix getGradBias() {
        return gradBias;
    }
    
}
