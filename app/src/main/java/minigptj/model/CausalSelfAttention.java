package minigptj.model;

import minigptj.core.Linear;
import minigptj.core.Matrix;

/**
 * Minimal single-head causal self-attention:
 * - Uses Matrix get/set + constructors.
 * - Uses existing Linear layers for projections (forward/backward).
 *
 * Shapes:
 *   Input X:  (B*T, dModel)   where row = b*T + t
 *   Output Y: (B*T, dModel)
 *
 * Internals (per batch block):
 *   Q = XWq, K = XWk, V = XWv
 *   scores[tq, tk] = (Q[tq] · K[tk]) / sqrt(dModel)
 *   causal mask: tk > tq masked
 *   attn = softmax(scores)
 *   context[tq] = Σ_k attn[tq,k] * V[k]
 *   Y = context * Wo
 */
public class CausalSelfAttention {
    private final int dModel;
    private final int seqLen;

    private final Linear Wq;
    private final Linear Wk;
    private final Linear Wv;
    private final Linear Wo;

    private boolean debugPrinted = false;
    private boolean backwardDebugPrinted = false;

    // caches for backward
    private Matrix lastX;      // (B*T, dModel)
    private Matrix Q;          // (B*T, dModel)
    private Matrix K;          // (B*T, dModel)
    private Matrix V;          // (B*T, dModel)
    private Matrix scores;     // (B*T, T)
    private Matrix attn;       // (B*T, T)
    private Matrix context;    // (B*T, dModel)

    public CausalSelfAttention(int dModel, int seqLen) {
        if (dModel < 1) throw new IllegalArgumentException("dModel must be >= 1");
        if (seqLen < 1) throw new IllegalArgumentException("seqLen must be >= 1");

        this.dModel = dModel;
        this.seqLen = seqLen;

        this.Wq = new Linear(dModel, dModel);
        this.Wk = new Linear(dModel, dModel);
        this.Wv = new Linear(dModel, dModel);
        this.Wo = new Linear(dModel, dModel);

        scaleWeights(Wq.getWeights(), 0.1);
        scaleWeights(Wk.getWeights(), 0.1);
    }

    /**
     * @param X (B*T, dModel)
     * @return  (B*T, dModel)
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
     * Backprop through attention.
     *
     * @param dOut (B*T, dModel)
     * @return dX  (B*T, dModel)
     */
    public Matrix backward(Matrix dOut) {
        if (lastX == null) throw new IllegalStateException("Must call forward() before backward().");
        if (dOut.getRows() != lastX.getRows() || dOut.getCols() != dModel) {
            throw new IllegalArgumentException("dOut has wrong shape");
        }

        int BT = lastX.getRows();
        int B = BT / seqLen;

        // 1) Through Wo
        Matrix dContext = Wo.backward(dOut); // (B*T, dModel)

        // 2) context[tq] = Σ_k attn[tq,k] * V[k]
        //    => dAttn[tq,k] = dContext[tq] · V[k]
        //    => dV[k] += attn[tq,k] * dContext[tq]
        Matrix dAttn = new Matrix(BT, seqLen);
        Matrix dV = new Matrix(BT, dModel);

        for (int b = 0; b < B; b++) {
            int base = b * seqLen;

            for (int tq = 0; tq < seqLen; tq++) {
                int qRow = base + tq;

                // dAttn row
                for (int tk = 0; tk < seqLen; tk++) {
                    int vRow = base + tk;

                    double dot = 0.0;
                    for (int i = 0; i < dModel; i++) {
                        dot += dContext.get(qRow, i) * V.get(vRow, i);
                    }
                    dAttn.set(qRow, tk, dAttn.get(qRow, tk) + dot);
                }

                // dV accumulation
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

        // 3) attn = softmax(scores) with causal mask
        Matrix dScores = maskedSoftmaxBackward(attn, dAttn); // (B*T, T)

        // 4) scores[tq,tk] = (Q[tq]·K[tk]) / sqrt(dModel)
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

        // 5) Backprop through projection linears
        Matrix dXq = Wq.backward(dQ);
        Matrix dXk = Wk.backward(dK);
        Matrix dXv = Wv.backward(dV);

        // 6) Sum gradients to input X
        Matrix dX = new Matrix(BT, dModel);
        for (int r = 0; r < BT; r++) {
            for (int c = 0; c < dModel; c++) {
                dX.set(r, c, dXq.get(r, c) + dXk.get(r, c) + dXv.get(r, c));
            }
        }

        return dX;
    }

    // scores[qRow, tk] = dot(Q[qRow], K[kRow]) / sqrt(dModel)
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

    // attn row-wise softmax with causal mask (tk > tq masked)
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
                    e = 0.0; // masked
                } else {
                    e = Math.exp(scores.get(row, tk) - max);
                }
                probs.set(row, tk, e);
                sumExp += e;
            }

            // normalise 
            for (int tk = 0; tk < seqLen; tk++) {
                probs.set(row, tk, probs.get(row, tk) / sumExp);
            }
        }

        return probs;
    }

    // context[tq] = Σ_k attn[tq,k] * V[k]
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
     * Backward for row-wise softmax with causal mask.
     *
     * For each row:
     *   dScores_j = a_j * (dAttn_j - Σ_k dAttn_k * a_k)
     * and for masked positions (tk > tq), dScores = 0.
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

    // Expose Linear layers so SGD.step(Linear) can update them.
    public Linear getWq() { return Wq; }
    public Linear getWk() { return Wk; }
    public Linear getWv() { return Wv; }
    public Linear getWo() { return Wo; }


    private static double min(Matrix m) {
        double min = m.get(0, 0);

        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                min = Math.min(min, m.get(i, j));
            }
        }

        return min;
    }

    private static double max(Matrix m) {
        double max = m.get(0, 0);

        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                max = Math.max(max, m.get(i, j));
            }
        }

        return max;
    }

    private static double rowSum(Matrix m, int row) {
        double sum = 0.0;
        for (int j = 0; j < m.getCols(); j++) {
            sum += m.get(row, j);
        }
        return sum;
    }

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

    private static void scaleWeights(Matrix w, double scale) {
    for (int i = 0; i < w.getRows(); i++)
        for (int j = 0; j < w.getCols(); j++)
            w.set(i, j, w.get(i, j) * scale);
    }
}