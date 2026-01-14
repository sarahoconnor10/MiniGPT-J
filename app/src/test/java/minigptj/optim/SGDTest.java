package minigptj.optim;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import minigptj.core.Linear;
import minigptj.core.Matrix;

public class SGDTest {
    @Test
    void testSGDUpdatesWeightsAndBias() {
        Linear layer = new Linear(2, 2);

        // Set deterministic weights and zero bias
        Matrix w = layer.getWeights();
        w.set(0, 0, 1.0); w.set(0, 1, 2.0);
        w.set(1, 0, 3.0); w.set(1, 1, 4.0);

        Matrix b = layer.getBias();
        b.set(0, 0, 0.0);
        b.set(0, 1, 0.0);

        // Forward just to set lastInput
        Matrix x = new Matrix(new double[][] {
            {1.0, 0.0},
            {0.0, 1.0}
        });
        layer.forward(x);

        // Choose a simple dOut so grads are predictable
        Matrix dOut = new Matrix(new double[][] {
            {1.0, 2.0},
            {3.0, 4.0}
        });
        layer.backward(dOut);

        // With X = identity, dW = dOut, db = [4, 6]
        SGD optim = new SGD(0.1);
        optim.step(layer);

        // Expected:
        // W_new = W_old - 0.1 * dW
        assertEquals(1.0 - 0.1 * 1.0, w.get(0, 0), 1e-9);
        assertEquals(2.0 - 0.1 * 2.0, w.get(0, 1), 1e-9);
        assertEquals(3.0 - 0.1 * 3.0, w.get(1, 0), 1e-9);
        assertEquals(4.0 - 0.1 * 4.0, w.get(1, 1), 1e-9);

        // b_new = b_old - 0.1 * db
        assertEquals(0.0 - 0.1 * 4.0, b.get(0, 0), 1e-9);
        assertEquals(0.0 - 0.1 * 6.0, b.get(0, 1), 1e-9);
    }
}
