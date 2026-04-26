package minigptj.model;

import java.util.Random;

import minigptj.core.Matrix;

/**
 * Learned token embedding layer.
 *
 * Maps discrete token IDs into dense vectors of length dModel. This allows the
 * model to learn useful continuous representations of characters rather than
 * relying only on sparse one-hot encodings.
 *
 * The class supports two output layouts:
 * - flattened layout: batchSize x (contextLen * dModel)
 * - sequence layout: (batchSize * contextLen) x dModel
 */
public class Embedding {
    private final int vocabSize;
    private final int dModel;

    private final Matrix weights;     // (vocabSize x dModel)
    private Matrix gradWeights;       // (vocabSize x dModel)

    // cache last input IDs for backprop
    private int[][] lastIds;

    /**
     * Creates an embedding layer with randomly initialised weights.
     *
     * @param vocabSize number of tokens in the vocabulary
     * @param dModel size of each learned embedding vector
     */
    public Embedding(int vocabSize, int dModel) {
        if (vocabSize < 2) throw new IllegalArgumentException("vocabSize must be >= 2");
        if (dModel < 1) throw new IllegalArgumentException("dModel must be >= 1");

        this.vocabSize = vocabSize;
        this.dModel = dModel;

        this.weights = new Matrix(vocabSize, dModel);
        this.gradWeights = new Matrix(vocabSize, dModel);

        // small random init
        Random rand = new Random();
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < dModel; j++) {
                weights.set(i, j, rand.nextGaussian() * 0.01);
            }
        }
    }

    /**
     * Performs a forward pass using flattened output layout.
     *
     * Input shape:
     *     batchSize x contextLen
     *
     * Output shape:
     *     batchSize x (contextLen * dModel)
     *
     * @param ids token IDs for each context window
     * @return flattened embedding matrix
     */
    public Matrix forward(int[][] ids) {
        if (ids == null) throw new IllegalArgumentException("ids cannot be null");
        if (ids.length == 0) throw new IllegalArgumentException("ids must not be empty");

        int batchSize = ids.length;
        int contextLen = ids[0].length;
        if (contextLen < 1) throw new IllegalArgumentException("contextLen must be >= 1");

        // validate rectangular + cache
        for (int i = 0; i < batchSize; i++) {
            if (ids[i].length != contextLen) {
                throw new IllegalArgumentException("ragged ids: all rows must have same length");
            }
        }
        this.lastIds = ids;

        Matrix out = new Matrix(batchSize, contextLen * dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int tokenId = ids[b][t];
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException("token id out of range: " + tokenId);
                }

                int baseCol = t * dModel;
                for (int j = 0; j < dModel; j++) {
                    out.set(b, baseCol + j, weights.get(tokenId, j));
                }
            }
        }

        return out;
    }

    /**
     * Performs a forward pass using sequence output layout.
     *
     * Input shape:
     *     batchSize x contextLen
     *
     * Output shape:
     *     (batchSize * contextLen) x dModel
     *
     * Row mapping:
     *     row = batchIndex * contextLen + tokenPosition
     *
     * This layout is used by the causal self-attention layer.
     *
     * @param ids token IDs for each context window
     * @return sequence-form embedding matrix
     */
    public Matrix forwardSeq(int[][] ids) {
        if (ids == null) throw new IllegalArgumentException("ids cannot be null");
        if (ids.length == 0) throw new IllegalArgumentException("ids must not be empty");

        int batchSize = ids.length;
        int contextLen = ids[0].length;
        for (int i = 0; i < batchSize; i++) {
            if (ids[i].length != contextLen) throw new IllegalArgumentException("ragged ids");
        }
        this.lastIds = ids;

        Matrix out = new Matrix(batchSize * contextLen, dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int tokenId = ids[b][t];
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException("token id out of range: " + tokenId);
                }
                int row = b * contextLen + t;
                for (int j = 0; j < dModel; j++) {
                    out.set(row, j, weights.get(tokenId, j));
                }
            }
        }
        return out;
    }


    /**
     * Backward pass for the flattened forward layout.
     *
     * Gradients are accumulated into the embedding rows used during the most
     * recent forward pass. If the same token appears multiple times, its
     * gradients are summed.
     *
     * @param dOut upstream gradient of shape batchSize x (contextLen * dModel)
     */
    public void backward(Matrix dOut) {
        if (lastIds == null) throw new IllegalStateException("Must call forward() before backward().");

        int batchSize = lastIds.length;
        int contextLen = lastIds[0].length;

        if (dOut.getRows() != batchSize || dOut.getCols() != contextLen * dModel) {
            throw new IllegalArgumentException("dOut has wrong shape");
        }

        // reset gradWeights to 0
        this.gradWeights = new Matrix(vocabSize, dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int tokenId = lastIds[b][t];
                int baseCol = t * dModel;

                for (int j = 0; j < dModel; j++) {
                    double g = dOut.get(b, baseCol + j);
                    // accumulate because the same token can appear multiple times
                    gradWeights.set(tokenId, j, gradWeights.get(tokenId, j) + g);
                }
            }
        }
    }
    /**
     * Backward pass for the sequence forward layout.
     *
     * This is used when embeddings are passed into the transformer attention
     * block in sequence form.
     *
     * @param dOut upstream gradient of shape (batchSize * contextLen) x dModel
     */
    public void backwardSeq(Matrix dOut) {
        if (lastIds == null) throw new IllegalStateException("Must call forwardSeq() before backwardSeq().");

        int batchSize = lastIds.length;
        int contextLen = lastIds[0].length;

        if (dOut.getRows() != batchSize * contextLen || dOut.getCols() != dModel) {
            throw new IllegalArgumentException("dOut has wrong shape for backwardSeq()");
        }

        // reset grads
        this.gradWeights = new Matrix(vocabSize, dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int tokenId = lastIds[b][t];
                int row = b * contextLen + t;

                for (int j = 0; j < dModel; j++) {
                    double g = dOut.get(row, j);
                    gradWeights.set(tokenId, j, gradWeights.get(tokenId, j) + g);
                }
            }
        }
    }

    /**
     * Returns the trainable embedding matrix.
     *
     * @return embedding weights with shape vocabSize x dModel
     */
    public Matrix getWeights() {
        return weights;
    }

    /**
     * Returns the most recent embedding gradients.
     *
     * @return gradient matrix for the embedding weights
     */
    public Matrix getGradWeights() {
        return gradWeights;
    }

    /**
     * Returns the number of tokens in the vocabulary.
     *
     * @return vocabulary size
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Returns the embedding dimension.
     *
     * @return size of each embedding vector
     */
    public int getDModel() {
        return dModel;
    }
}
