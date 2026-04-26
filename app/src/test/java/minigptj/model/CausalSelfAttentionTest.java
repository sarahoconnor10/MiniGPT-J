package minigptj.model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import minigptj.core.Matrix;

public class CausalSelfAttentionTest {

    @Test
    void testForwardOutputShape() {
        int dModel = 8;
        int seqLen = 4;
        CausalSelfAttention attn = new CausalSelfAttention(dModel, seqLen);

        // batchSize=2, so input is (2*4) x 8
        Matrix x = randomMatrix(8, dModel);
        Matrix out = attn.forward(x);

        assertEquals(8, out.getRows());
        assertEquals(dModel, out.getCols());
    }

    @Test
    void testForwardSingleBatch() {
        int dModel = 4;
        int seqLen = 3;
        CausalSelfAttention attn = new CausalSelfAttention(dModel, seqLen);

        Matrix x = randomMatrix(3, dModel);
        Matrix out = attn.forward(x);

        assertEquals(3, out.getRows());
        assertEquals(4, out.getCols());
    }

    @Test
    void testForwardWrongColsThrows() {
        CausalSelfAttention attn = new CausalSelfAttention(8, 4);
        Matrix x = randomMatrix(4, 6); // wrong cols
        assertThrows(IllegalArgumentException.class, () -> attn.forward(x));
    }

    @Test
    void testForwardRowsNotDivisibleBySeqLenThrows() {
        CausalSelfAttention attn = new CausalSelfAttention(8, 4);
        Matrix x = randomMatrix(5, 8); // 5 not divisible by seqLen=4
        assertThrows(IllegalArgumentException.class, () -> attn.forward(x));
    }

    @Test
    void testBackwardOutputShape() {
        int dModel = 8;
        int seqLen = 4;
        CausalSelfAttention attn = new CausalSelfAttention(dModel, seqLen);

        Matrix x = randomMatrix(8, dModel);
        attn.forward(x);

        Matrix dOut = randomMatrix(8, dModel);
        Matrix dX = attn.backward(dOut);

        assertEquals(8, dX.getRows());
        assertEquals(dModel, dX.getCols());
    }

    @Test
    void testBackwardRequiresForward() {
        CausalSelfAttention attn = new CausalSelfAttention(8, 4);
        Matrix dOut = randomMatrix(8, 8);
        assertThrows(IllegalStateException.class, () -> attn.backward(dOut));
    }

    @Test
    void testCausalMaskingZerosFutureAttention() {
        // With causal masking, attention weights for future positions must be 0
        // We verify this indirectly: output at position 0 should not be affected
        // by changing values at position 1+
        int dModel = 4;
        int seqLen = 3;
        CausalSelfAttention attn1 = new CausalSelfAttention(dModel, seqLen);
        CausalSelfAttention attn2 = new CausalSelfAttention(dModel, seqLen);

        // Copy weights from attn1 to attn2 so they're identical
        copyWeights(attn1, attn2);

        Matrix x1 = new Matrix(new double[][]{
            {1.0, 0.0, 0.0, 0.0},  // position 0
            {0.0, 1.0, 0.0, 0.0},  // position 1
            {0.0, 0.0, 1.0, 0.0}   // position 2
        });

        // x2 is identical at position 0 but different at positions 1 and 2
        Matrix x2 = new Matrix(new double[][]{
            {1.0, 0.0, 0.0, 0.0},  // same position 0
            {9.9, 9.9, 9.9, 9.9},  // different position 1
            {9.9, 9.9, 9.9, 9.9}   // different position 2
        });

        Matrix out1 = attn1.forward(x1);
        Matrix out2 = attn2.forward(x2);

        // Position 0 output should be identical — it can only attend to itself
        for (int j = 0; j < dModel; j++) {
            assertEquals(out1.get(0, j), out2.get(0, j), 1e-6,
                "Position 0 output should not depend on future positions");
        }
    }

    // helpers

    private Matrix randomMatrix(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, rng.nextGaussian() * 0.1);
        return m;
    }

    private void copyWeights(CausalSelfAttention src, CausalSelfAttention dst) {
        copyMatrix(src.getWq().getWeights(), dst.getWq().getWeights());
        copyMatrix(src.getWk().getWeights(), dst.getWk().getWeights());
        copyMatrix(src.getWv().getWeights(), dst.getWv().getWeights());
        copyMatrix(src.getWo().getWeights(), dst.getWo().getWeights());
        copyMatrix(src.getWq().getBias(), dst.getWq().getBias());
        copyMatrix(src.getWk().getBias(), dst.getWk().getBias());
        copyMatrix(src.getWv().getBias(), dst.getWv().getBias());
        copyMatrix(src.getWo().getBias(), dst.getWo().getBias());
    }

    private void copyMatrix(Matrix src, Matrix dst) {
        for (int i = 0; i < src.getRows(); i++)
            for (int j = 0; j < src.getCols(); j++)
                dst.set(i, j, src.get(i, j));
    }
}