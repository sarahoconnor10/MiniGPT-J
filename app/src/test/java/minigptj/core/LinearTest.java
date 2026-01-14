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
    
    @Test
    void testBackwardComputesGradientsCorrectly() {
        Linear layer = new Linear(2, 2);
        
        // Set deterministic weights W = [[1,2],[3,4]]
        Matrix w = layer.getWeights();
        w.set(0, 0, 1.0); w.set(0, 1, 2.0);
        w.set(1, 0, 3.0); w.set(1, 1, 4.0);
        
        // Bias doesn't affect gradients of W/db for this layer's backward,
        // but set to 0 for cleanliness.
        Matrix b = layer.getBias();
        b.set(0, 0, 0.0);
        b.set(0, 1, 0.0);
        
        // Batch size 2: X shape (2x2)
        Matrix x = new Matrix(new double[][] {
            {1.0, 0.0},
            {0.0, 1.0}
        });
        
        // Must call forward to cache lastInput
        layer.forward(x);
        
        // Upstream gradient dOut shape (2x2)
        Matrix dOut = new Matrix(new double[][] {
            {1.0, 2.0},
            {3.0, 4.0}
        });
        
        Matrix dX = layer.backward(dOut);
        
        // ---- Expected dX = dOut * W^T ----
        // W^T = [[1,3],[2,4]]
        // Row0: [1,2] * W^T => [1*1+2*2=5, 1*3+2*4=11]
        // Row1: [3,4] * W^T => [3*1+4*2=11, 3*3+4*4=25]
        assertEquals(5.0,  dX.get(0, 0), 1e-9);
        assertEquals(11.0, dX.get(0, 1), 1e-9);
        assertEquals(11.0, dX.get(1, 0), 1e-9);
        assertEquals(25.0, dX.get(1, 1), 1e-9);
        
        // ---- Expected dW = X^T * dOut ----
        // X is identity so X^T * dOut = dOut
        Matrix dW = layer.getGradWeights();
        assertEquals(1.0, dW.get(0, 0), 1e-9);
        assertEquals(2.0, dW.get(0, 1), 1e-9);
        assertEquals(3.0, dW.get(1, 0), 1e-9);
        assertEquals(4.0, dW.get(1, 1), 1e-9);
        
        // ---- Expected db = sum rows of dOut ----
        // col0: 1 + 3 = 4
        // col1: 2 + 4 = 6
        Matrix db = layer.getGradBias();
        assertEquals(4.0, db.get(0, 0), 1e-9);
        assertEquals(6.0, db.get(0, 1), 1e-9);
    }
    
    @Test
    void testBackwardGradientShapes() {
        Linear layer = new Linear(3, 2);

        Matrix x = new Matrix(new double[][] {
            {1, 2, 3},
            {4, 5, 6}
        }); // (2x3)

        layer.forward(x);

        Matrix dOut = new Matrix(new double[][] {
            {1, 1},
            {1, 1}
        }); // (2x2)

        Matrix dX = layer.backward(dOut);

        // dX should be (2x3)
        assertEquals(2, dX.getRows());
        assertEquals(3, dX.getCols());

        // dW should be (3x2)
        Matrix dW = layer.getGradWeights();
        assertEquals(3, dW.getRows());
        assertEquals(2, dW.getCols());

        // db should be (1x2)
        Matrix db = layer.getGradBias();
        assertEquals(1, db.getRows());
        assertEquals(2, db.getCols());
    }


    @Test
    void testBackwardRequiresForwardCall() {
        Linear layer = new Linear(2, 2);

        Matrix dOut = new Matrix(new double[][] {
            {1, 1}
        });

        assertThrows(IllegalStateException.class, () -> layer.backward(dOut));
    }

}
    