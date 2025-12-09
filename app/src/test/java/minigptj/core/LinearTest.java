package minigptj.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class LinearTest {
    @Test
    void testLinearOutputShape() {
        Linear layer = new Linear(3, 2);

        // One input row with 3 columns
        Matrix input = new Matrix(new double[][] {
            {1.0, 2.0, 3.0}
        });

        Matrix out = layer.forward(input);

        // Shape should be (1 x 2)
        assertEquals(1, out.getRows());
        assertEquals(2, out.getCols());
    }

    @Test
    void testLinearDeterministicForward() {
        // Create a layer with known weights
        Linear layer = new Linear(3, 2);

        // Overwrite weights with known values for testing
        Matrix w = layer.getWeights();
        w.set(0, 0, 1.0);  
        w.set(0, 1, 2.0);
        w.set(1, 0, 3.0);  
        w.set(1, 1, 4.0);
        w.set(2, 0, 5.0);  
        w.set(2, 1, 6.0);

        // Overwrite bias with zeros
        Matrix b = layer.getBias();
        b.set(0, 0, 0.0);
        b.set(0, 1, 0.0);

        Matrix input = new Matrix(new double[][] {
            {1.0, 1.0, 1.0}
        });

        Matrix out = layer.forward(input);

        // Expected:
        // [1*1 + 1*3 + 1*5 = 9]
        // [1*2 + 1*4 + 1*6 = 12]
        assertEquals(9.0,  out.get(0, 0), 1e-9);
        assertEquals(12.0, out.get(0, 1), 1e-9);
    }

    @Test
    void testBiasAddition() {
        Linear layer = new Linear(2, 2);

        // Set weights to zero
        Matrix w = layer.getWeights();
        w.set(0,0,0); 
        w.set(0,1,0);
        w.set(1,0,0); 
        w.set(1,1,0);

        // Set bias to known values
        Matrix b = layer.getBias();
        b.set(0,0,5.0);
        b.set(0,1,7.0);

        Matrix input = new Matrix(new double[][]{
            {10.0, 20.0}
        });

        Matrix out = layer.forward(input);

        // Expect output = bias (because weights = 0)
        assertEquals(5.0, out.get(0, 0), 1e-9);
        assertEquals(7.0, out.get(0, 1), 1e-9);
    }
}
