package minigptj.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class SimpleMLPTest {
    @Test
    void testForwardOutputShape() {
        int inputSize = 4;
        int hiddenSize = 5;
        int outputSize = 3;

        SimpleMLP mlp = new SimpleMLP(inputSize, hiddenSize, outputSize);

        Matrix input = new Matrix(new double[][]{
            {1.0, 0.5, -1.0, 2.0},
            {0.0, 1.0, 1.0, -0.5}
        });

        Matrix logits = mlp.forward(input);

        // We passed 2 examples (rows), expect 2 x outputSize
        assertEquals(2, logits.getRows());
        assertEquals(outputSize, logits.getCols());
    }

    @Test
    void testForwardWithSoftmaxSumsToOne() {
        SimpleMLP mlp = new SimpleMLP(4, 5, 3);

        Matrix input = new Matrix(new double[][]{
            {1.0, 0.5, -1.0, 2.0}
        });

        Matrix probs = mlp.forwardWithSoftmax(input);

        assertEquals(1, probs.getRows());
        assertEquals(3, probs.getCols());

        double sum = probs.get(0,0) + probs.get(0,1) + probs.get(0,2);
        assertEquals(1.0, sum, 1e-9);
    }
}
