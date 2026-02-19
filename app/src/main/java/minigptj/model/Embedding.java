package minigptj.model;

import java.util.Random;

import minigptj.core.Matrix;

/**
 * Learned token embedding layer.
 *
 * Maps token IDs to dense vectors of length dModel.
 * We return a 2D Matrix shaped (batchSize x (contextLen * dModel)
 *
 */
public class Embedding {
    private final int vocabSize;
    private final int dModel;

    private final Matrix weights;     // (vocabSize x dModel)
    private Matrix gradWeights;       // (vocabSize x dModel)

    // cache last input IDs for backprop
    private int[][] lastIds;

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
     * Forward pass: lookup embeddings and flatten per token position.
     *
     * @param ids shape: (batchSize x contextLen)
     * @return output shape: (batchSize x (contextLen * dModel))
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
     * Backward pass: accumulate gradients into embedding rows used in forward().
     *
     * @param dOut shape: (batchSize x (contextLen * dModel))
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

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getGradWeights() {
        return gradWeights;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int getDModel() {
        return dModel;
    }
}