package minigptj.data;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.Random;

public class TextDatasetTest {

    @Test
    void testSize() {
        int[] tokens = {2, 3, 4, 5, 6};
        TextDataset ds = new TextDataset(tokens, 3);
        // size = tokens.length - 1
        assertEquals(4, ds.size());
    }

    @Test
    void testGetTargetIsNextToken() {
        int[] tokens = {2, 3, 4, 5};
        TextDataset ds = new TextDataset(tokens, 2);
        assertEquals(3, ds.getTarget(0));
        assertEquals(4, ds.getTarget(1));
        assertEquals(5, ds.getTarget(2));
    }

    @Test
    void testContextLengthIsCorrect() {
        int[] tokens = {2, 3, 4, 5, 6};
        TextDataset ds = new TextDataset(tokens, 3);
        assertEquals(3, ds.getContext(0).length);
    }

    @Test
    void testLeftPaddingApplied() {
        int[] tokens = {2, 3, 4, 5};
        TextDataset ds = new TextDataset(tokens, 4);
        int[] ctx = ds.getContext(0);
        // context for index 0: needs 4 tokens ending at tokens[0]=2
        // so 3 PAD tokens + tokens[0]
        assertEquals(CharTokenizer.PAD_ID, ctx[0]);
        assertEquals(CharTokenizer.PAD_ID, ctx[1]);
        assertEquals(CharTokenizer.PAD_ID, ctx[2]);
        assertEquals(2, ctx[3]);
    }

    @Test
    void testSampleBatchShape() {
        int[] tokens = {2, 3, 4, 5, 6, 7, 8};
        TextDataset ds = new TextDataset(tokens, 3);
        TextDataset.Batch batch = ds.sampleBatch(4, new Random(42));
        assertEquals(4, batch.x.length);
        assertEquals(3, batch.x[0].length);
        assertEquals(4, batch.y.length);
    }
}