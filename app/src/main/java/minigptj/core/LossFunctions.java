package minigptj.core;

/**
 * Utility class containing loss functions and loss gradients used during training.
 *
 * Loss functions measure how far the model's predictions are from the expected
 * targets. The corresponding gradients provide the starting point for
 * backpropagation.
 */
public final class LossFunctions {
    /**
     * Prevent instantiation of utility class.
     */
    private LossFunctions() {
    }

    /**
     * Computes the Mean Squared Error (MSE) between a predicted matrix
     * and an actual target matrix.
     *
     * MSE = average((predicted - actual)^2)
     *
     * @param predicted the model's output values
     * @param actual the ground-truth target values
     * @return the mean squared error as a double
     * @throws IllegalArgumentException if the shapes of the matrices do not match
     */
    public static double meanSquaredError(Matrix predicted, Matrix actual) {
        if (predicted.getRows() != actual.getRows() || predicted.getCols() != actual.getCols()) {
            throw new IllegalArgumentException("Shapes must match for MSE");
        }

        double sum = 0.0;
        int rows = predicted.getRows();
        int cols = predicted.getCols();
        int n = rows * cols;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = predicted.get(i, j) - actual.get(i, j);
                sum += diff * diff;
            }
        }

        return sum / n;
    }

    /**
     * Computes categorical cross-entropy loss between predicted probabilities
     * and one-hot encoded target labels.
     *
     * Each row represents one training example. For each row, only the
     * probability assigned to the correct class contributes to the loss.
     *
     * @param predicted a matrix of predicted probabilities (e.g., from softmax)
     * @param actual a matrix of one-hot encoded true labels
     * @return the average cross-entropy loss over all rows
     * @throws IllegalArgumentException if the shapes do not match
     */
    public static double crossEntropy(Matrix predicted, Matrix actual) {
        if (predicted.getRows() != actual.getRows() || predicted.getCols() != actual.getCols()) {
            throw new IllegalArgumentException("Shapes must match for cross-entropy");
        }

        double epsilon = 1e-12; // prevent log(0)
        double sumLoss = 0.0;
        int rows = predicted.getRows();
        int cols = predicted.getCols();

        for (int i = 0; i < rows; i++) {
            double rowLoss = 0.0;

            for (int j = 0; j < cols; j++) {
                double yTrue = actual.get(i, j);

                if (yTrue == 1.0) {
                    double yPred = predicted.get(i, j);
                    yPred = Math.max(epsilon, Math.min(1.0 - epsilon, yPred));
                    rowLoss += -Math.log(yPred);
                }
            }

            sumLoss += rowLoss;
        }

        return sumLoss / rows;
    }

    /**
     * Converts integer class labels into a one-hot encoded matrix.
     *
     * Example:
     * labels = [2, 0], numClasses = 3
     * output =
     * [0, 0, 1]
     * [1, 0, 0]
     *
     * @param labels class labels, each in the range [0, numClasses)
     * @param numClasses total number of classes
     * @return one-hot matrix of shape labels.length x numClasses
     */
    public static Matrix oneHotFromLabels(int[] labels, int numClasses) {
        if (labels == null) throw new IllegalArgumentException("labels cannot be null");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");

        Matrix y = new Matrix(labels.length, numClasses);
        for (int i = 0; i < labels.length; i++) {
            int c = labels[i];
            if (c < 0 || c >= numClasses) {
                throw new IllegalArgumentException("label out of range: " + c);
            }
            y.set(i, c, 1.0);
        }
        return y;
    }

    /**
     * Computes the gradient of softmax combined with cross-entropy loss.
     *
     * If:
     *     probs = softmax(logits)
     *
     * Then:
     *     dLogits = (probs - yTrue) / batchSize
     *
     * This simplification is standard for classification and language modelling.
     *
     * @param probs predicted probabilities, shape batchSize x numClasses
     * @param yTrue one-hot encoded targets, shape batchSize x numClasses
     * @return gradient with respect to logits
     */
    public static Matrix softmaxCrossEntropyGrad(Matrix probs, Matrix yTrue) {
        if (probs.getRows() != yTrue.getRows() || probs.getCols() != yTrue.getCols()) {
            throw new IllegalArgumentException("Shapes must match for softmaxCrossEntropyGrad");
        }

        int rows = probs.getRows();
        int cols = probs.getCols();
        Matrix dLogits = new Matrix(rows, cols);

        double invBatch = 1.0 / rows;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double grad = (probs.get(i, j) - yTrue.get(i, j)) * invBatch;
                dLogits.set(i, j, grad);
            }
        }
        return dLogits;
    }
}
