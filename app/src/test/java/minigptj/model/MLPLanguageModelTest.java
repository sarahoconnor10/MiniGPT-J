package minigptj.model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import minigptj.core.Matrix;

public class MLPLanguageModelTest {

    @Test
    void testForwardOutputShape() {
        MLPLanguageModel model = new MLPLanguageModel(4, 8, 5);
        Matrix x = new Matrix(new double[][]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0}
        });
        Matrix out = model.forward(x);
        assertEquals(2, out.getRows());
        assertEquals(5, out.getCols());
    }

    @Test
    void testForwardSingleExample() {
        MLPLanguageModel model = new MLPLanguageModel(3, 6, 4);
        Matrix x = new Matrix(new double[][]{{1.0, 0.5, -1.0}});
        Matrix out = model.forward(x);
        assertEquals(1, out.getRows());
        assertEquals(4, out.getCols());
    }

    @Test
    void testBackwardOutputShape() {
        MLPLanguageModel model = new MLPLanguageModel(4, 8, 5);
        Matrix x = new Matrix(new double[][]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0}
        });
        model.forward(x);
        Matrix dLogits = new Matrix(new double[][]{
            {0.1, 0.2, 0.3, 0.2, 0.2},
            {0.2, 0.1, 0.1, 0.3, 0.3}
        });
        Matrix dX = model.backward(dLogits);
        assertEquals(2, dX.getRows());
        assertEquals(4, dX.getCols());
    }

    @Test
    void testBackwardRequiresForward() {
        MLPLanguageModel model = new MLPLanguageModel(4, 8, 5);
        Matrix dLogits = new Matrix(2, 5);
        assertThrows(IllegalStateException.class, () -> model.backward(dLogits));
    }
}