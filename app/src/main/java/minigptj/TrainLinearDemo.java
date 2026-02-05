package minigptj;

import minigptj.core.Linear;
import minigptj.core.LossFunctions;
import minigptj.core.Matrix;
import minigptj.optim.SGD;

/**
 * Demonstrates end-to-end training of a single Linear layer:
 * - forward pass
 * - MSE loss
 * - backpropagation
 * - SGD optimisation
 *
 * The target function is y = 2*x1 - 1*x2.
 * Loss should decrease over epochs and learned weights should converge.
 */
public class TrainLinearDemo {
    public static void main(String[] args) {
        /*
         * PROBLEM SETUP:
         *
         * We want the model to learn a known rule:
         *     y = 2*x1 - 1*x2
         *
         * Because we know the correct relationship,
         * we can verify that learning is working
         * by checking whether the weights converge
         * to approximately 2 and -1.
         */

        // Create a single Linear layer with:
        //  - 2 inputs
        //  - 1 output
        Linear layer = new Linear(2, 1);

        // Create an SGD optimiser with learning rate 0.1
        // This controls how big each learning step is
        SGD optim = new SGD(0.1);

        /*
         * TOY DATASET:
         *
         * Each row is one training example with two inputs.
         * This is a small batch of four examples,
         * kept simple for demonstration.
         */
        Matrix X = new Matrix(new double[][] {
            {1.0, 0.0},  // y = 2
            {0.0, 1.0},  // y = -1
            {1.0, 1.0},  // y = 1
            {2.0, 1.0}   // y = 3
        });

        /*
         * These are the correct target outputs
         * computed using y = 2*x1 - x2.
         */
        Matrix yTrue = new Matrix(new double[][] {
            {2.0},
            {-1.0},
            {1.0},
            {3.0}
        });

        // Number of training iterations
        int epochs = 50;

        /*
         * TRAINING LOOP:
         *
         * Each epoch performs one full learning step:
         *  - forward pass
         *  - loss calculation
         *  - backpropagation
         *  - weight update
         */
        for (int epoch = 1; epoch <= epochs; epoch++) {
            /*
             * 1) FORWARD PASS
             *
             * The model uses its current weights and bias
             * to produce predictions.
             *
             * At the beginning, these predictions are poor
             * because the weights are random.
             */
            Matrix yPred = layer.forward(X);

            /*
             * 2) LOSS COMPUTATION
             *
             * Mean Squared Error measures how far
             * the predictions are from the true values.
             *
             * This gives us a single number representing
             * how wrong the model currently is.
             */
            double loss = LossFunctions.meanSquaredError(yPred, yTrue);

            /*
             * 3) LOSS GRADIENT
             *
             * We compute how the loss changes
             * with respect to the predictions.
             *
             * This gradient is the learning signal
             * that will be propagated backward.
             */
            Matrix dOut = mseGradient(yPred, yTrue);

            /*
             * 4) BACKPROPAGATION
             *
             * The error signal is sent backward
             * through the Linear layer.
             *
             * This computes gradients for:
             *  - the weights
             *  - the bias
             */
            layer.backward(dOut);

            /*
             * 5) OPTIMISATION STEP
             *
             * Stochastic Gradient Descent updates
             * the weights and bias by nudging them
             * in the direction that reduces the loss.
             *
             * This is where learning actually happens.
             */
            optim.step(layer);

            /*
             * Print progress occasionally so we can
             * observe that the loss is decreasing.
             */
            if (epoch == 1 || epoch % 10 == 0) {
                System.out.printf("Epoch %d | loss = %.6f%n", epoch, loss);
            }
        }

        /*
         * FINAL RESULT:
         *
         * After training, we print the learned
         * weights and bias.
         *
         * These should be close to:
         *  - weight for x1 ≈ 2
         *  - weight for x2 ≈ -1
         *  - bias ≈ 0
         *
         * This confirms the training pipeline works.
         */
        System.out.println("\nLearned weights + bias:");
        System.out.println("W[0,0] = " + layer.getWeights().get(0, 0));
        System.out.println("W[1,0] = " + layer.getWeights().get(1, 0));
        System.out.println("b[0,0] = " + layer.getBias().get(0, 0));
    }

    /*
     * Computes the gradient of Mean Squared Error
     * with respect to the predictions.
     *
     * MSE = average((predicted - actual)^2)
     *
     * This function provides the error signal
     * used for backpropagation.
     */
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
