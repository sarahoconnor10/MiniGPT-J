package minigptj.data;

import java.util.Random;

/**
 * Converts a token stream into training examples for language modelling.
 *
 * Each example follows the task:
 *
 *     given a fixed-length context window, predict the next token
 *
 * For example, with text "hello" and context length 3:
 *
 *     context: [PAD, PAD, h] -> target: e
 *     context: [PAD, h, e]   -> target: l
 *     context: [h, e, l]     -> target: l
 *     context: [e, l, l]     -> target: o
 */
public class TextDataset {
    private final int[] tokens;
    private final int contextLen;
    private final int padId;

    /**
     * Creates a dataset from a tokenised text sequence.
     *
     * @param tokens full tokenised text
     * @param contextLen number of previous tokens used as context
     * @param padId token ID used for left padding
     */
    public TextDataset(int[] tokens, int contextLen, int padId) {
        if (tokens == null) throw new IllegalArgumentException("tokens cannot be null");
        if (tokens.length < 2) throw new IllegalArgumentException("need at least 2 tokens to form training pairs");
        if (contextLen < 1) throw new IllegalArgumentException("contextLen must be >= 1");

        this.tokens = tokens;
        this.contextLen = contextLen;
        this.padId = padId;
    }

    /**
     * Creates a dataset using CharTokenizer.PAD_ID as the padding token.
     *
     * @param tokens full tokenised text
     * @param contextLen number of previous tokens used as context
     */
    public TextDataset(int[] tokens, int contextLen) {
        this(tokens, contextLen, CharTokenizer.PAD_ID);
    }

    /**
     * Returns the number of available training examples.
     *
     * Example index i predicts tokens[i + 1] from a context ending at tokens[i].
     *
     * @return number of context-target pairs
     */
    public int size() {
        return tokens.length - 1;
    }

    /**
     * Returns the fixed-length context window for a training example.
     *
     * If there are not enough previous tokens at the start of the sequence,
     * the missing positions are left-padded using padId.
     *
     * @param index training example index
     * @return context token IDs of length contextLen
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
     * Returns the target token for a training example.
     *
     * The target is the next token after the context window.
     *
     * @param index training example index
     * @return target token ID
     */
    public int getTarget(int index) {
        if (index < 0 || index >= size()) {
            throw new IllegalArgumentException("index out of range: " + index);
        }
        return tokens[index + 1];
    }

    /**
     * Samples a random batch of context-target pairs with replacement.
     *
     * @param batchSize number of examples in the batch
     * @param rng random number generator
     * @return sampled batch
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

    /**
     * Immutable container for a batch of training examples.
     */
    public static class Batch {
        /** Context windows, shape: batchSize x contextLen. */
        public final int[][] x;

        /** Target token IDs, shape: batchSize. */
        public final int[] y;

        /**
         * Creates a batch container.
         *
         * @param x context windows
         * @param y target token IDs
         */
        public Batch(int[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }
}
