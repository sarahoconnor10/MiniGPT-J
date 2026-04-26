package minigptj.model;

import minigptj.core.Linear;
import minigptj.core.Matrix;

/**
 * Implements single-head causal self-attention.
 *
 * This is the core transformer component used in MiniGPT-J. It allows each token
 * to attend to earlier tokens in the same sequence while preventing access to
 * future tokens using a causal mask.
 *
 * Input shape:
 *     (batchSize * seqLen) x dModel
 *
 * Output shape:
 *     (batchSize * seqLen) x dModel
 *
 * Row mapping:
 *     row = batchIndex * seqLen + tokenPosition
 *
 * Forward pass:
 *     Q = XWq
 *     K = XWk
 *     V = XWv
 *     scores = QK^T / sqrt(dModel)
 *     attn = causalMaskedSoftmax(scores)
 *     context = attn * V
 *     output = context * Wo
 */
public class CausalSelfAttention {
    private final int dModel;
    private final int seqLen;

    private final Linear Wq;
    private final Linear Wk;
    private final Linear Wv;
    private final Linear Wo;

    private boolean debugPrinted = true;
    private boolean backwardDebugPrinted = true;

    // caches for backward
    private Matrix lastX;      // (B*T, dModel)
    private Matrix Q;          // (B*T, dModel)
    private Matrix K;          // (B*T, dModel)
    private Matrix V;          // (B*T, dModel)
    private Matrix scores;     // (B*T, T)
    private Matrix attn;       // (B*T, T)
    private Matrix context;    // (B*T, dModel)

    /**
     * Creates a single-head causal self-attention layer.
     *
     * @param dModel embedding dimension
     * @param seqLen fixed sequence length used by the attention mask
     */
    public CausalSelfAttention(int dModel, int seqLen) {
        if (dModel < 1) throw new IllegalArgumentException("dModel must be >= 1");
        if (seqLen < 1) throw new IllegalArgumentException("seqLen must be >= 1");

        this.dModel = dModel;
        this.seqLen = seqLen;

        this.Wq = new Linear(dModel, dModel);
        this.Wk = new Linear(dModel, dModel);
        this.Wv = new Linear(dModel, dModel);
        this.Wo = new Linear(dModel, dModel);

        /*
         * Scale query/key projections slightly at initialisation.
         *
         * During development, Wq and Wk were found to receive near-zero gradients
         * when the attention scores started too uniformly. Scaling helped stabilise
         * early training and supported gradient flow through the attention softmax.
         */
        scaleWeights(Wq.getWeights(), 0.1);
        scaleWeights(Wk.getWeights(), 0.1);
    }

    /**
     * Forward pass through causal self-attention.
     *
     * @param X input matrix of shape (batchSize * seqLen) x dModel
     * @return output matrix of shape (batchSize * seqLen) x dModel
     */
    public Matrix forward(Matrix X) {
        if (X.getCols() != dModel) {
            throw new IllegalArgumentException("X cols must equal dModel");
        }
        if (X.getRows() % seqLen != 0) {
            throw new IllegalArgumentException("X rows must be divisible by seqLen");
        }

        this.lastX = X;

        Q = Wq.forward(X);
        K = Wk.forward(X);
        V = Wv.forward(X);

        scores = computeScores(Q, K);      // (B*T, T)
        attn = maskedSoftmax(scores);      // (B*T, T)
        context = computeContext(attn, V); // (B*T, dModel)

        if (!debugPrinted) {
            System.out.println("Q range: " + min(Q) + " to " + max(Q));
            System.out.println("K range: " + min(K) + " to " + max(K));
            System.out.println("V range: " + min(V) + " to " + max(V));
            System.out.println("scores range: " + min(scores) + " to " + max(scores));
            System.out.println("attn range: " + min(attn) + " to " + max(attn));

            int lastRow = seqLen - 1;
            System.out.println("attn row sum (last row, batch 0): " + rowSum(attn, lastRow));
            System.out.println("attn row (last row, batch 0):");
            printRow(attn, lastRow);

            debugPrinted = true;
        }

        return Wo.forward(context);
    }

    /**
     * Backward pass through causal self-attention.
     *
     * Manually propagates gradients through:
     * - output projection
     * - context/value weighted sum
     * - masked softmax
     * - scaled dot-product scores
     * - query, key, and value projections
     *
     * @param dOut upstream gradient of shape (batchSize * seqLen) x dModel
     * @return gradient with respect to input X
     */
    public Matrix backward(Matrix dOut) {
        if (lastX == null) throw new IllegalStateException("Must call forward() before backward().");
        if (dOut.getRows() != lastX.getRows() || dOut.getCols() != dModel) {
            throw new IllegalArgumentException("dOut has wrong shape");
        }

        int BT = lastX.getRows();
        int B = BT / seqLen;


        Matrix dContext = Wo.backward(dOut); // (B*T, dModel)

        /*
         * context[tq] = sum(attn[tq, tk] * V[tk])
         *
         * Therefore:
         *     dAttn[tq, tk] = dContext[tq] dot V[tk]
         *     dV[tk] += attn[tq, tk] * dContext[tq]
         */
        Matrix dAttn = new Matrix(BT, seqLen);
        Matrix dV = new Matrix(BT, dModel);

        for (int b = 0; b < B; b++) {
            int base = b * seqLen;

            for (int tq = 0; tq < seqLen; tq++) {
                int qRow = base + tq;

                for (int tk = 0; tk < seqLen; tk++) {
                    int vRow = base + tk;

                    double dot = 0.0;
                    for (int i = 0; i < dModel; i++) {
                        dot += dContext.get(qRow, i) * V.get(vRow, i);
                    }
                    dAttn.set(qRow, tk, dAttn.get(qRow, tk) + dot);
                }

                for (int tk = 0; tk < seqLen; tk++) {
                    double a = attn.get(qRow, tk);
                    int vRow = base + tk;

                    for (int i = 0; i < dModel; i++) {
                        double cur = dV.get(vRow, i);
                        dV.set(vRow, i, cur + a * dContext.get(qRow, i));
                    }
                }
            }
        }

        // Backprop through the masked softmax operation.
        Matrix dScores = maskedSoftmaxBackward(attn, dAttn); // (B*T, T)

        /*
         * scores[tq, tk] = Q[tq] dot K[tk] / sqrt(dModel)
         *
         * Therefore:
         *     dQ[tq] += dScore[tq, tk] * K[tk] / sqrt(dModel)
         *     dK[tk] += dScore[tq, tk] * Q[tq] / sqrt(dModel)
         */
        double scale = 1.0 / Math.sqrt(dModel);

        Matrix dQ = new Matrix(BT, dModel);
        Matrix dK = new Matrix(BT, dModel);

        for (int b = 0; b < B; b++) {
            int base = b * seqLen;

            for (int tq = 0; tq < seqLen; tq++) {
                int qRow = base + tq;

                for (int tk = 0; tk < seqLen; tk++) {
                    // masked positions contribute nothing
                    if (tk > tq) continue;

                    int kRow = base + tk;

                    double g = dScores.get(qRow, tk) * scale;

                    for (int i = 0; i < dModel; i++) {
                        // dQ[qRow,i] += g * K[kRow,i]
                        dQ.set(qRow, i, dQ.get(qRow, i) + g * K.get(kRow, i));
                        // dK[kRow,i] += g * Q[qRow,i]
                        dK.set(kRow, i, dK.get(kRow, i) + g * Q.get(qRow, i));
                    }
                }
            }
        }

        if (!backwardDebugPrinted) {
            System.out.println("dContext L2 = " + l2(dContext));
            System.out.println("dAttn L2    = " + l2(dAttn));
            System.out.println("dScores L2  = " + l2(dScores));
            System.out.println("dQ L2       = " + l2(dQ));
            System.out.println("dK L2       = " + l2(dK));
            System.out.println("dV L2       = " + l2(dV));
            backwardDebugPrinted = true;
        }

        // Backprop through projection linears
        Matrix dXq = Wq.backward(dQ);
        Matrix dXk = Wk.backward(dK);
        Matrix dXv = Wv.backward(dV);

        // Sum gradients to input X
        Matrix dX = new Matrix(BT, dModel);
        for (int r = 0; r < BT; r++) {
            for (int c = 0; c < dModel; c++) {
                dX.set(r, c, dXq.get(r, c) + dXk.get(r, c) + dXv.get(r, c));
            }
        }

        return dX;
    }

    /**
     * Computes scaled dot-product attention scores for each batch sequence.
     *
     * @param Q query matrix of shape (batchSize * seqLen) x dModel
     * @param K key matrix of shape (batchSize * seqLen) x dModel
     * @return score matrix of shape (batchSize * seqLen) x seqLen
     */
    private Matrix computeScores(Matrix Q, Matrix K) {
        int BT = Q.getRows();
        int B = BT / seqLen;

        Matrix s = new Matrix(BT, seqLen);
        double scale = 1.0 / Math.sqrt(dModel);

        for (int b = 0; b < B; b++) {
            int base = b * seqLen;

            for (int tq = 0; tq < seqLen; tq++) {
                int qRow = base + tq;

                for (int tk = 0; tk < seqLen; tk++) {
                    int kRow = base + tk;

                    double dot = 0.0;
                    for (int i = 0; i < dModel; i++) {
                        dot += Q.get(qRow, i) * K.get(kRow, i);
                    }
                    s.set(qRow, tk, dot * scale);
                }
            }
        }

        return s;
    }

    /**
     * Applies row-wise softmax with a causal mask.
     *
     * For token position tq, only keys tk <= tq are visible.
     * Future positions are assigned probability 0.
     *
     * @param scores attention score matrix of shape (batchSize * seqLen) x seqLen
     * @return masked attention probability matrix of the same shape
     */
    private Matrix maskedSoftmax(Matrix scores) {
        int BT = scores.getRows();
        Matrix probs = new Matrix(BT, seqLen);

        for (int row = 0; row < BT; row++) {
            int tq = row % seqLen;

            // max over allowed keys (0..tq)
            double max = Double.NEGATIVE_INFINITY;
            for (int tk = 0; tk <= tq; tk++) {
                double v = scores.get(row, tk);
                if (v > max) max = v;
            }

            double sumExp = 0.0;
            for (int tk = 0; tk < seqLen; tk++) {
                double e;
                if (tk > tq) {
                    e = 0.0;
                } else {
                    e = Math.exp(scores.get(row, tk) - max);
                }
                probs.set(row, tk, e);
                sumExp += e;
            }

            for (int tk = 0; tk < seqLen; tk++) {
                probs.set(row, tk, probs.get(row, tk) / sumExp);
            }
        }

        return probs;
    }

    /**
     * Computes context vectors as weighted sums of value vectors.
     *
     * For each query position tq:
     *     context[tq] = sum over tk of attn[tq, tk] * V[tk]
     *
     * @param attn attention probability matrix of shape (batchSize * seqLen) x seqLen
     * @param V value matrix of shape (batchSize * seqLen) x dModel
     * @return context matrix of shape (batchSize * seqLen) x dModel
     */
    private Matrix computeContext(Matrix attn, Matrix V) {
        int BT = V.getRows();
        int B = BT / seqLen;

        Matrix out = new Matrix(BT, dModel);

        for (int b = 0; b < B; b++) {
            int base = b * seqLen;

            for (int tq = 0; tq < seqLen; tq++) {
                int outRow = base + tq;

                for (int i = 0; i < dModel; i++) {
                    double sum = 0.0;
                    for (int tk = 0; tk < seqLen; tk++) {
                        double a = attn.get(outRow, tk);
                        int vRow = base + tk;
                        sum += a * V.get(vRow, i);
                    }
                    out.set(outRow, i, sum);
                }
            }
        }

        return out;
    }

    /**
     * Computes the backward pass for row-wise softmax with causal masking.
     *
     * For each unmasked position:
     *     dScores_j = attn_j * (dAttn_j - sum_k(dAttn_k * attn_k))
     *
     * Masked future positions receive zero gradient.
     *
     * @param attn attention probabilities from the forward pass
     * @param dAttn upstream gradient with respect to attention probabilities
     * @return gradient with respect to the pre-softmax attention scores
     */
    private Matrix maskedSoftmaxBackward(Matrix attn, Matrix dAttn) {
        int BT = attn.getRows();
        Matrix dScores = new Matrix(BT, seqLen);

        for (int row = 0; row < BT; row++) {
            int tq = row % seqLen;

            double dot = 0.0;
            for (int tk = 0; tk <= tq; tk++) {
                dot += dAttn.get(row, tk) * attn.get(row, tk);
            }

            for (int tk = 0; tk < seqLen; tk++) {
                if (tk > tq) {
                    dScores.set(row, tk, 0.0);
                } else {
                    double a = attn.get(row, tk);
                    double g = a * (dAttn.get(row, tk) - dot);
                    dScores.set(row, tk, g);
                }
            }
        }

        return dScores;
    }

    /**
     * Returns the query projection layer.
     *
     * @return query Linear layer
     */
    public Linear getWq() { return Wq; }

    /**
     * Returns the key projection layer.
     *
     * @return key Linear layer
     */
    public Linear getWk() { return Wk; }

    /**
     * Returns the value projection layer.
     *
     * @return value Linear layer
     */
    public Linear getWv() { return Wv; }

    /**
     * Returns the output projection layer.
     *
     * @return output Linear layer
     */
    public Linear getWo() { return Wo; }

    /**
     * Finds the minimum value in a matrix.
     *
     * @param m matrix to inspect
     * @return minimum value
     */
    private static double min(Matrix m) {
        double min = m.get(0, 0);

        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                min = Math.min(min, m.get(i, j));
            }
        }

        return min;
    }

    /**
     * Finds the maximum value in a matrix.
     *
     * @param m matrix to inspect
     * @return maximum value
     */
    private static double max(Matrix m) {
        double max = m.get(0, 0);

        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                max = Math.max(max, m.get(i, j));
            }
        }

        return max;
    }

    /**
     * Computes the sum of a single matrix row.
     *
     * @param m matrix to inspect
     * @param row row index
     * @return sum of values in the row
     */
    private static double rowSum(Matrix m, int row) {
        double sum = 0.0;
        for (int j = 0; j < m.getCols(); j++) {
            sum += m.get(row, j);
        }
        return sum;
    }

    /**
     * Prints one row of a matrix for debugging.
     *
     * @param m matrix to inspect
     * @param row row index to print
     */
    private static void printRow(Matrix m, int row) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int j = 0; j < m.getCols(); j++) {
            sb.append(String.format("%.4f", m.get(row, j)));
            if (j < m.getCols() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        System.out.println(sb.toString());
    }

    /**
     * Computes the L2 norm of all values in a matrix.
     *
     * @param m matrix to inspect
     * @return Euclidean norm of the matrix values
     */
    private static double l2(Matrix m) {
        double sum = 0.0;
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double v = m.get(i, j);
                sum += v * v;
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * Scales every value in a weight matrix in-place.
     *
     * Used to reduce the initial magnitude of query/key projection weights,
     * which helped stabilise early attention training.
     *
     * @param w matrix to scale
     * @param scale scalar multiplier
     */
    private static void scaleWeights(Matrix w, double scale) {
        for (int i = 0; i < w.getRows(); i++) {
            for (int j = 0; j < w.getCols(); j++) {
                w.set(i, j, w.get(i, j) * scale);
            }
        }
    }
}
