package minigptj.data;

import minigptj.core.Matrix;

/**
 * Utility class for converting token IDs into one-hot encoded matrices.
 *
 * One-hot encoding represents each token as a sparse vector where:
 * - the token's index is set to 1
 * - all other positions are 0
 *
 * These encodings were used in earlier stages of the project before
 * embedding layers were introduced.
 */
public final class OneHot {
    /**
     * Prevent instantiation of utility class.
     */
    private OneHot() {
    }

    /**
     * Converts a 1D array of token IDs into a one-hot encoded matrix.
     *
     * Input shape:
     *     [batchSize]
     *
     * Output shape:
     *     [batchSize x vocabSize]
     *
     * @param ids token IDs
     * @param vocabSize vocabulary size
     * @return one-hot encoded matrix
     */
    public static Matrix encode1D(int[] ids, int vocabSize) {
        if (ids == null) throw new IllegalArgumentException("ids cannot be null");
        if (vocabSize < 2) throw new IllegalArgumentException("vocabSize must be >= 2");

        Matrix m = new Matrix(ids.length, vocabSize);
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            if (id < 0 || id >= vocabSize) {
                throw new IllegalArgumentException("token id out of range: " + id);
            }
            m.set(i, id, 1.0);
        }
        return m;
    }

    /**
     * Encodes only the final token from each context window.
     *
     * Input shape:
     *     [batchSize][contextLen]
     *
     * Output shape:
     *     [batchSize x vocabSize]
     *
     * This method was useful in earlier experiments where only the most
     * recent token in a context window was used for prediction.
     *
     * @param contextIds batched context windows
     * @param vocabSize vocabulary size
     * @return one-hot encoded matrix of final context tokens
     */
    public static Matrix encodeContextLast(int[][] contextIds, int vocabSize) {
        if (contextIds == null) throw new IllegalArgumentException("contextIds cannot be null");
        if (contextIds.length == 0) throw new IllegalArgumentException("contextIds must not be empty");

        int batchSize = contextIds.length;
        int contextLen = contextIds[0].length;
        if (contextLen < 1) throw new IllegalArgumentException("contextLen must be >= 1");

        int[] lastIds = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            if (contextIds[i].length != contextLen) {
                throw new IllegalArgumentException("ragged contextIds: all rows must have same length");
            }
            lastIds[i] = contextIds[i][contextLen - 1];
        }
        return encode1D(lastIds, vocabSize);
    }

    /**
     * Concatenates one-hot encodings for an entire context window.
     *
     * Input shape:
     *     [batchSize][contextLen]
     *
     * Output shape:
     *     [batchSize x (contextLen * vocabSize)]
     *
     * Each token position occupies its own block of vocabSize columns.
     *
     * Example:
     *     column = (position * vocabSize) + tokenId
     *
     * @param contextIds batched context windows
     * @param vocabSize vocabulary size
     * @return concatenated one-hot matrix
     */
    public static Matrix encodeContextConcat(int[][] contextIds, int vocabSize) {
        if (contextIds == null) throw new IllegalArgumentException("contextIds cannot be null");
        if (contextIds.length == 0) throw new IllegalArgumentException("contextIds must not be empty");
        if (vocabSize < 2) throw new IllegalArgumentException("vocabSize must be >= 2");

        int batchSize = contextIds.length;
        int contextLen = contextIds[0].length;

        Matrix m = new Matrix(batchSize, contextLen * vocabSize);

        for (int i = 0; i < batchSize; i++) {
            if (contextIds[i].length != contextLen) {
                throw new IllegalArgumentException("ragged contextIds: all rows must have same length");
            }

            for (int t = 0; t < contextLen; t++) {
                int id = contextIds[i][t];
                if (id < 0 || id >= vocabSize) {
                    throw new IllegalArgumentException("token id out of range: " + id);
                }

                int col = t * vocabSize + id; // shift block per position
                m.set(i, col, 1.0);
            }
        }

        return m;
    }
}