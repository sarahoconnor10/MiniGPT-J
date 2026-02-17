package minigptj.data;

import java.util.Random;

/**
 * TextDataset turns a token stream into (context -> next token) training examples.
 *
 * Language modelling task:
 *   Given previous tokens (context), predict the next token (target).
 */
public class TextDataset {
    private final int[] tokens;
    private final int contextLen;
    private final int padId;

    /**
     * @param tokens     full tokenised text (e.g., output of tokenizer.encode)
     * @param contextLen number of previous tokens used as input context
     * @param padId      token id used for left padding (usually CharTokenizer.PAD_ID)
    */
    public TextDataset(int[] tokens, int contextLen, int padId) {
        if (tokens == null) throw new IllegalArgumentException("tokens cannot be null");
        if (tokens.length < 2) throw new IllegalArgumentException("need at least 2 tokens to form training pairs");
        if (contextLen < 1) throw new IllegalArgumentException("contextLen must be >= 1");

        this.tokens = tokens;
        this.contextLen = contextLen;
        this.padId = padId;
    }

    /** Constructor using CharTokenizer.PAD_ID. */
    public TextDataset(int[] tokens, int contextLen) {
        this(tokens, contextLen, CharTokenizer.PAD_ID);
    }

    /**
     * Number of training examples.
     * We predict tokens[i+1] from context ending at tokens[i].
     */
    public int size() {
        return tokens.length - 1;
    }

    /**
     * Returns the context window (length = contextLen) for a given example index.
     *
     * Example index i corresponds to predicting tokens[i+1].
     * The context ends at tokens[i] and looks back contextLen tokens.
     * Missing left positions are padded with padId.
     */
    public int[] getContext(int index) {
        if (index < 0 || index >= size()) {
            throw new IllegalArgumentException("index out of range: " + index);
        }

        int[] x = new int[contextLen];

        // Context ends at position 'index' (predicting index+1)
        int end = index;
        int start = end - contextLen + 1;

        for (int j = 0; j < contextLen; j++) {
            int srcIndex = start + j;
            x[j] = (srcIndex < 0) ? padId : tokens[srcIndex];
        }

        return x;
    }

    /**
     * Returns the target token id (the "next token") for a given example index.
     * Example index i targets tokens[i+1].
     */
    public int getTarget(int index) {
        if (index < 0 || index >= size()) {
            throw new IllegalArgumentException("index out of range: " + index);
        }
        return tokens[index + 1];
    }

    /**
     * Sample a random batch (with replacement).
     * Useful for SGD training loops.
     */
    public Batch sampleBatch(int batchSize, Random rng) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be >= 1");
        if (rng == null) throw new IllegalArgumentException("rng cannot be null");

        int[][] xBatch = new int[batchSize][contextLen];
        int[] yBatch = new int[batchSize];

        for (int i = 0; i < batchSize; i++) {
            int idx = rng.nextInt(size());
            xBatch[i] = getContext(idx);
            yBatch[i] = getTarget(idx);
        }

        return new Batch(xBatch, yBatch);
    }

    /** Simple batch container. */
    public static class Batch {
        public final int[][] x; // shape: [batchSize][contextLen]
        public final int[] y;   // shape: [batchSize]

        public Batch(int[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }
}
