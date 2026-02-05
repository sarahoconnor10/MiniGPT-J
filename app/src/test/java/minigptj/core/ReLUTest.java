package minigptj.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class ReLUTest {
    @Test
    void testReLUBackwardMasksNegativesAndZero() {
        ReLU relu = new ReLU();

        Matrix x = new Matrix(new double[][]{
            {-1.0,  2.0},
            { 0.0,  3.0}
        });

        relu.forward(x);

        Matrix dOut = new Matrix(new double[][]{
            {10.0, 10.0},
            {10.0, 10.0}
        });

        Matrix dX = relu.backward(dOut);

        // x <= 0 -> gradient blocked
        assertEquals(0.0,  dX.get(0, 0), 1e-9);
        assertEquals(10.0, dX.get(0, 1), 1e-9);
        assertEquals(0.0,  dX.get(1, 0), 1e-9);
        assertEquals(10.0, dX.get(1, 1), 1e-9);
    }

    @Test
    void testReLUBackwardRequiresForward() {
        ReLU relu = new ReLU();
        Matrix dOut = new Matrix(new double[][]{{1.0}});
        assertThrows(IllegalStateException.class, () -> relu.backward(dOut));
    }

}
