package minigptj.data;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import minigptj.core.Matrix;

public class OneHotTest {

    @Test
    void testEncode1DShape() {
        int[] ids = {0, 1, 2};
        Matrix m = OneHot.encode1D(ids, 5);
        assertEquals(3, m.getRows());
        assertEquals(5, m.getCols());
    }

    @Test
    void testEncode1DCorrectPosition() {
        int[] ids = {2};
        Matrix m = OneHot.encode1D(ids, 5);
        assertEquals(1.0, m.get(0, 2), 1e-9);
        assertEquals(0.0, m.get(0, 0), 1e-9);
        assertEquals(0.0, m.get(0, 4), 1e-9);
    }

    @Test
    void testEncodeContextLastUsesLastToken() {
        int[][] ctx = {{0, 1, 3}};
        Matrix m = OneHot.encodeContextLast(ctx, 5);
        // should one-hot encode token 3 only
        assertEquals(1, m.getRows());
        assertEquals(1.0, m.get(0, 3), 1e-9);
        assertEquals(0.0, m.get(0, 1), 1e-9);
    }

    @Test
    void testEncodeContextConcatShape() {
        int[][] ctx = {{0, 1, 2}, {1, 2, 3}};
        Matrix m = OneHot.encodeContextConcat(ctx, 5);
        assertEquals(2, m.getRows());
        assertEquals(15, m.getCols()); // contextLen * vocabSize = 3 * 5
    }
}