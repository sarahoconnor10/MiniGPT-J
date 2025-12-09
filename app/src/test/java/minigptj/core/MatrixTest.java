package minigptj.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class MatrixTest {
     @Test
     public void testAddition() {
        Matrix a = new Matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix b = new Matrix(new double[][]{
            {10, 20},
            {30, 40}
        });

        Matrix result = a.add(b);

        assertEquals(11, result.get(0, 0), 1e-9);
        assertEquals(22, result.get(0, 1), 1e-9);
        assertEquals(33, result.get(1, 0), 1e-9);
        assertEquals(44, result.get(1, 1), 1e-9);
    }

    @Test
    void testScalarMultiply() {
        Matrix a = new Matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix scaled = a.multiply(2.0);

        assertEquals(2, scaled.get(0, 0), 1e-9);
        assertEquals(4, scaled.get(0, 1), 1e-9);
        assertEquals(6, scaled.get(1, 0), 1e-9);
        assertEquals(8, scaled.get(1, 1), 1e-9);
    }

    @Test
    void testDotProduct() {
        Matrix a = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix b = new Matrix(new double[][]{
            {7, 8},
            {9, 10},
            {11, 12}
        });

        Matrix c = a.dot(b);

        assertEquals(58, c.get(0, 0), 1e-9);
        assertEquals(64, c.get(0, 1), 1e-9);
        assertEquals(139, c.get(1, 0), 1e-9);
        assertEquals(154, c.get(1, 1), 1e-9);
    }

    @Test
    void testTranspose() {
        Matrix a = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix t = a.transpose();

        assertEquals(1, t.get(0, 0), 1e-9);
        assertEquals(4, t.get(0, 1), 1e-9);

        assertEquals(2, t.get(1, 0), 1e-9);
        assertEquals(5, t.get(1, 1), 1e-9);

        assertEquals(3, t.get(2, 0), 1e-9);
        assertEquals(6, t.get(2, 1), 1e-9);

    }

    @Test
    void testSoftmaxRows() {
        Matrix logits = new Matrix(new double[][]{
            {2.0, 1.0, 0.1}
        });

        Matrix probs = logits.softmaxRows();

        double p0 = probs.get(0,0);
        double p1 = probs.get(0,1);
        double p2 = probs.get(0,2);

        assertEquals(1.0, p0 + p1 + p2, 1e-9);

        assertTrue(p0 > p1);
        assertTrue(p0 > p2);
    }

}
