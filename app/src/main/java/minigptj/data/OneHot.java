package minigptj.data;

import minigptj.core.Matrix;

/**
 * Utility to convert token ids into one-hot encoded matrices.
 */
public class OneHot {

    /** ids: [batchSize] -> Matrix [batchSize, vocabSize] */
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
     * contextIds: [batchSize][contextLen]
     * Uses ONLY the last token in each context window -> Matrix [batchSize, vocabSize]
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
     * contextIds: [batchSize][contextLen]
     * Concatenates one-hot vectors for the entire context window.
     *
     * Output shape: [batchSize, contextLen * vocabSize]
     *
     * For each position t in the context:
     *   we place a 1.0 at column (t * vocabSize + tokenId)
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