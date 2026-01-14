package minigptj;

import minigptj.core.Linear;
import minigptj.core.LossFunctions;
import minigptj.core.Matrix;
import minigptj.optim.SGD;

public class TrainLinearDemo {
    public static void main(String[] args) {
        // Train a single Linear layer to learn a simple rule:
        // y = 2*x1 - 1*x2

        Linear layer = new Linear(2, 1);
        SGD optim = new SGD(0.1);

        // Small toy dataset (batch of 4)
        Matrix X = new Matrix(new double[][] {
            {1.0, 0.0},  // y = 2
            {0.0, 1.0},  // y = -1
            {1.0, 1.0},  // y = 1
            {2.0, 1.0}   // y = 3
        });

        Matrix yTrue = new Matrix(new double[][] {
            {2.0},
            {-1.0},
            {1.0},
            {3.0}
        });

        int epochs = 50;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // 1) forward: prediction
            Matrix yPred = layer.forward(X);

            // 2) loss: how wrong are we?
            double loss = LossFunctions.meanSquaredError(yPred, yTrue);

            // 3) gradient of MSE wrt predictions:
            // MSE = average((pred - true)^2)
            // dMSE/dPred = (2/N) * (pred - true)
            Matrix dOut = mseGradient(yPred, yTrue);

            // 4) backward: compute gradients for weights/bias
            layer.backward(dOut);

            // 5) step: update weights/bias using SGD
            optim.step(layer);

            // Print every 10 epochs so output isn't spammy
            if (epoch == 1 || epoch % 10 == 0) {
                System.out.printf("Epoch %d | loss = %.6f%n", epoch, loss);
            }
        }

        // Final parameters (optional)
        System.out.println("\nLearned weights + bias:");
        System.out.println("W[0,0] = " + layer.getWeights().get(0, 0));
        System.out.println("W[1,0] = " + layer.getWeights().get(1, 0));
        System.out.println("b[0,0] = " + layer.getBias().get(0, 0));
    }

    private static Matrix mseGradient(Matrix predicted, Matrix actual) {
        int rows = predicted.getRows();
        int cols = predicted.getCols();

        Matrix grad = new Matrix(rows, cols);

        // N = total number of elements (batchSize * outputSize)
        double N = rows * cols;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = predicted.get(i, j) - actual.get(i, j);
                grad.set(i, j, (2.0 / N) * diff);
            }
        }
        return grad;
    }
}
