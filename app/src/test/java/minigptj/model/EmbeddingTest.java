package minigptj.model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import minigptj.core.Matrix;

public class EmbeddingTest {

    @Test
    void testForwardSeqShape() {
        Embedding emb = new Embedding(10, 4);
        int[][] ids = {{2, 3, 4}, {1, 5, 6}};
        Matrix out = emb.forwardSeq(ids);
        // batchSize * contextLen x dModel = 6 x 4
        assertEquals(6, out.getRows());
        assertEquals(4, out.getCols());
    }

    @Test
    void testForwardSeqLooksUpCorrectRow() {
        Embedding emb = new Embedding(10, 4);
        // set a known embedding for token 3
        for (int j = 0; j < 4; j++) {
            emb.getWeights().set(3, j, 99.0);
        }
        int[][] ids = {{3}};
        Matrix out = emb.forwardSeq(ids);
        for (int j = 0; j < 4; j++) {
            assertEquals(99.0, out.get(0, j), 1e-9);
        }
    }

    @Test
    void testBackwardSeqShape() {
        Embedding emb = new Embedding(10, 4);
        int[][] ids = {{2, 3}, {4, 5}};
        emb.forwardSeq(ids);

        Matrix dOut = new Matrix(4, 4); // batchSize*contextLen x dModel
        emb.backwardSeq(dOut);

        Matrix grad = emb.getGradWeights();
        assertEquals(10, grad.getRows());
        assertEquals(4, grad.getCols());
    }

    @Test
    void testBackwardSeqAccumulatesGrads() {
        Embedding emb = new Embedding(5, 3);
        // same token appearing twice in batch
        int[][] ids = {{2}, {2}};
        emb.forwardSeq(ids);

        Matrix dOut = new Matrix(new double[][]{
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0}
        });
        emb.backwardSeq(dOut);

        // grad for token 2 should be sum = 2.0 per dimension
        for (int j = 0; j < 3; j++) {
            assertEquals(2.0, emb.getGradWeights().get(2, j), 1e-9);
        }
    }

    @Test
    void testBackwardSeqRequiresForward() {
        Embedding emb = new Embedding(5, 3);
        Matrix dOut = new Matrix(1, 3);
        assertThrows(IllegalStateException.class, () -> emb.backwardSeq(dOut));
    }
}