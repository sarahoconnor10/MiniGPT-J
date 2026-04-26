package minigptj.optim;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import minigptj.core.Linear;
import minigptj.core.Matrix;

public class AdamTest {

    @Test
    void testAdamUpdatesWeights() {
        Linear layer = new Linear(2, 2);

        Matrix w = layer.getWeights();
        w.set(0, 0, 1.0); w.set(0, 1, 1.0);
        w.set(1, 0, 1.0); w.set(1, 1, 1.0);

        Matrix x = new Matrix(new double[][]{{1.0, 0.0}, {0.0, 1.0}});
        layer.forward(x);

        Matrix dOut = new Matrix(new double[][]{{1.0, 1.0}, {1.0, 1.0}});
        layer.backward(dOut);

        Adam adam = new Adam(0.001);
        adam.tick();
        adam.step(layer);

        // weights should have changed
        assertNotEquals(1.0, w.get(0, 0), 1e-9);
        assertNotEquals(1.0, w.get(1, 1), 1e-9);
    }

    @Test
    void testAdamUpdatesBias() {
        Linear layer = new Linear(2, 2);

        Matrix b = layer.getBias();
        b.set(0, 0, 0.0);
        b.set(0, 1, 0.0);

        Matrix x = new Matrix(new double[][]{{1.0, 1.0}});
        layer.forward(x);

        Matrix dOut = new Matrix(new double[][]{{1.0, 1.0}});
        layer.backward(dOut);

        Adam adam = new Adam(0.001);
        adam.tick();
        adam.step(layer);

        assertNotEquals(0.0, b.get(0, 0), 1e-9);
        assertNotEquals(0.0, b.get(0, 1), 1e-9);
    }

    @Test
    void testAdamRequiresBackwardFirst() {
        Linear layer = new Linear(2, 2);
        Adam adam = new Adam(0.001);
        adam.tick();
        assertThrows(IllegalStateException.class, () -> adam.step(layer));
    }

    @Test
    void testAdamStepReducesLossDirection() {
        // After one Adam step, gradient should point away from original param
        Linear layer = new Linear(1, 1);
        layer.getWeights().set(0, 0, 1.0);
        layer.getBias().set(0, 0, 0.0);

        Matrix x = new Matrix(new double[][]{{1.0}});
        layer.forward(x);

        // positive gradient means weight should decrease
        Matrix dOut = new Matrix(new double[][]{{1.0}});
        layer.backward(dOut);

        double before = layer.getWeights().get(0, 0);
        Adam adam = new Adam(0.001);
        adam.tick();
        adam.step(layer);
        double after = layer.getWeights().get(0, 0);

        assertTrue(after < before);
    }
}