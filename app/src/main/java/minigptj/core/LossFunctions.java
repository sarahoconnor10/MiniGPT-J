package minigptj.core;

public class LossFunctions {
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
     * Computes the categorical cross-entropy loss between predicted probabilities
     * and actual one-hot encoded class labels.
     * Each row represents one training example.
     * 
     * Cross-entropy for each row is:
     *   -log(predicted_probability_of_correct_class)
     *
     * For numerical stability, probabilities are clamped between [epsilon, 1 - epsilon].
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
}
