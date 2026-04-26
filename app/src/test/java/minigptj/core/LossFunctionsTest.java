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

    @Test
    void testOneHotFromLabelsShape() {
        int[] labels = {0, 1, 2};
        Matrix m = LossFunctions.oneHotFromLabels(labels, 4);
        assertEquals(3, m.getRows());
        assertEquals(4, m.getCols());
    }

    @Test
    void testOneHotFromLabelsCorrectPositions() {
        int[] labels = {0, 2};
        Matrix m = LossFunctions.oneHotFromLabels(labels, 3);
        assertEquals(1.0, m.get(0, 0), 1e-9);
        assertEquals(0.0, m.get(0, 1), 1e-9);
        assertEquals(1.0, m.get(1, 2), 1e-9);
        assertEquals(0.0, m.get(1, 0), 1e-9);
    }

    @Test
    void testOneHotFromLabelsOutOfRangeThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> LossFunctions.oneHotFromLabels(new int[]{5}, 3));
    }

    @Test
    void testSoftmaxCrossEntropyGradShape() {
        Matrix probs = new Matrix(new double[][]{
            {0.7, 0.2, 0.1},
            {0.1, 0.8, 0.1}
        });
        Matrix yTrue = new Matrix(new double[][]{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}
        });
        Matrix grad = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);
        assertEquals(2, grad.getRows());
        assertEquals(3, grad.getCols());
    }

    @Test
    void testSoftmaxCrossEntropyGradValues() {
        Matrix probs = new Matrix(new double[][]{{0.7, 0.2, 0.1}});
        Matrix yTrue = new Matrix(new double[][]{{1.0, 0.0, 0.0}});
        Matrix grad = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);
        assertEquals(-0.3, grad.get(0, 0), 1e-9);
        assertEquals(0.2, grad.get(0, 1), 1e-9);
    }

    @Test
    void testSoftmaxCrossEntropyGradScaledByBatchSize() {
        Matrix probs = new Matrix(new double[][]{
            {0.7, 0.3},
            {0.4, 0.6}
        });
        Matrix yTrue = new Matrix(new double[][]{
            {1.0, 0.0},
            {1.0, 0.0}
        });
        Matrix grad = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);
        assertEquals((0.7 - 1.0) / 2.0, grad.get(0, 0), 1e-9);
        assertEquals((0.4 - 1.0) / 2.0, grad.get(1, 0), 1e-9);
    }
}