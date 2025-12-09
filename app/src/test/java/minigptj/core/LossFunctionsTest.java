package minigptj.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class LossFunctionsTest {
    @Test
    void testMeanSquaredError() {
        Matrix predicted = new Matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });

        Matrix actual = new Matrix(new double[][]{
            {1.0, 3.0},
            {2.0, 4.0}
        });

        // Differences:
        // (1-1)^2 = 0
        // (2-3)^2 = 1
        // (3-2)^2 = 1
        // (4-4)^2 = 0
        // Sum = 2, number of elements = 4 → MSE = 0.5
        double mse = LossFunctions.meanSquaredError(predicted, actual);

        assertEquals(0.5, mse, 1e-9);
    }

    @Test
    void testCrossEntropySimple() {
        Matrix predicted = new Matrix(new double[][]{
            {0.7, 0.2, 0.1}
        });

        // True class is index 0 → one-hot: [1, 0, 0]
        Matrix actual = new Matrix(new double[][]{
            {1.0, 0.0, 0.0}
        });

        double loss = LossFunctions.crossEntropy(predicted, actual);

        // Cross-entropy = -log(0.7)
        double expected = -Math.log(0.7);

        assertEquals(expected, loss, 1e-9);
    }

    @Test
    void testCrossEntropyDifferentClass() {
        Matrix predicted = new Matrix(new double[][]{
            {0.1, 0.2, 0.7}
        });

        // True class is index 2 → one-hot: [0, 0, 1]
        Matrix actual = new Matrix(new double[][]{
            {0.0, 0.0, 1.0}
        });

        double loss = LossFunctions.crossEntropy(predicted, actual);

        // Cross-entropy = -log(0.7)
        double expected = -Math.log(0.7);

        assertEquals(expected, loss, 1e-9);
    }
}
