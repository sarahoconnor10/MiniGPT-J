package minigptj.core;

import java.util.Random;

public class Linear {

    private final int inputSize;
    private final int outputSize;
    private final Matrix weights;
    private final Matrix bias;

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
        // input shape: (batchSize x inputSize)
        Matrix out = input.dot(weights);   // (batchSize x outputSize)
        return out.add(bias);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBias() {
        return bias;
    }

    


}
