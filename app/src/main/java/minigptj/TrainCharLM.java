package minigptj;

import java.util.Random;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.core.ReLU;
import minigptj.data.CharTokenizer;
import minigptj.data.TextDataset;
import minigptj.model.CausalSelfAttention;
import minigptj.model.Embedding;
import minigptj.optim.SGD;

public class TrainCharLM {

    public static void main(String[] args) {
        String text =
            "hello world\n" +
            "hello there\n" +
            "how are you\n" +
            "how is the world\n";
        
        text = text.repeat(5);

        CharTokenizer tok = CharTokenizer.fromText(text);
        int[] tokens = tok.encode(text);
        int vocabSize = tok.vocabSize();

        int contextLen = 10;
        int dModel = 32;
        
        TextDataset ds = new TextDataset(tokens, contextLen);
        SequenceBatch fullBatch = buildFullSequenceBatch(ds, contextLen);
        int batchSize = fullBatch.x.length;
        
        int steps = 4000;
        double learningRate = 0.01;

        Embedding emb = new Embedding(vocabSize, dModel);
        CausalSelfAttention attn = new CausalSelfAttention(dModel, contextLen);
        Linear outProj = new Linear(dModel, vocabSize);
        SGD opt = new SGD(learningRate);

        Linear ffn1 = new Linear(dModel, dModel * 4);
        ReLU ffnAct = new ReLU();
        Linear ffn2 = new Linear(dModel * 4, dModel);

        Matrix pos = initPositionalEmbeddings(contextLen, dModel, new Random(123));

        for (int step = 1; step <= steps; step++) {
            SequenceBatch batch = fullBatch;

            Matrix xSeq = emb.forwardSeq(batch.x);
            addPositionalEmbeddings(xSeq, pos, batchSize, contextLen, dModel);

            if (step == 1) {
                System.out.println("xSeq = " + xSeq.getRows() + " x " + xSeq.getCols());
                System.out.println("expected = " + (batchSize * contextLen) + " x " + dModel);
            }

            Matrix attnOnly = attn.forward(xSeq);
            Matrix attnOutSeq = attnOnly.add(xSeq); 

            Matrix ffnHidden = ffn1.forward(attnOutSeq);
            ffnHidden = ffnAct.forward(ffnHidden);
            Matrix ffnOut = ffn2.forward(ffnHidden);

            Matrix blockOut = ffnOut.add(attnOutSeq); 

            // Full-sequence supervision:
            // blockOut shape = (batchSize * contextLen) x dModel
            // logits shape   = (batchSize * contextLen) x vocabSize
            Matrix logits = outProj.forward(blockOut);
            Matrix probs = logits.softmaxRows();

            int[] flatTargets = flattenTargets(batch.ySeq);
            double loss = maskedCrossEntropy(probs, flatTargets, CharTokenizer.PAD_ID);
            Matrix dLogits = maskedSoftmaxCrossEntropyGrad(probs, flatTargets, CharTokenizer.PAD_ID);

            Matrix dBlockOut = outProj.backward(dLogits);

            Matrix dFfnOut = dBlockOut;
            Matrix dAttnOutSeq = dBlockOut;

            Matrix dHidden = ffn2.backward(dFfnOut);
            dHidden = ffnAct.backward(dHidden);
            Matrix dFfnInput = ffn1.backward(dHidden);

            dAttnOutSeq = dAttnOutSeq.add(dFfnInput);
            
            // blockOut = ffnOut + attnOutSeq
            // attnOutSeq = attn.forward(xSeq) + xSeq
            Matrix dXSeq = attn.backward(dAttnOutSeq).add(dAttnOutSeq);

            emb.backwardSeq(dXSeq);

            Matrix gradPos = accumulatePosGradients(dXSeq, batchSize, contextLen, dModel);

            if (step == 1) {
                System.out.println("grad norms step1:");
                System.out.println("  outProj dW L2 = " + l2(outProj.getGradWeights()));
                System.out.println("  outProj db L2 = " + l2(outProj.getGradBias()));
                System.out.println("  Wq dW L2      = " + l2(attn.getWq().getGradWeights()));
                System.out.println("  Wk dW L2      = " + l2(attn.getWk().getGradWeights()));
                System.out.println("  Wv dW L2      = " + l2(attn.getWv().getGradWeights()));
                System.out.println("  Wo dW L2      = " + l2(attn.getWo().getGradWeights()));
                System.out.println("  emb dW L2     = " + l2(emb.getGradWeights()));
                System.out.println("  pos dW L2     = " + l2(gradPos));
            }

            opt.step(emb);
            opt.step(attn.getWq());
            opt.step(attn.getWk());
            opt.step(attn.getWv());
            opt.step(attn.getWo());
            opt.step(outProj);
            opt.step(ffn1);
            opt.step(ffn2);

            updatePositionalEmbeddings(pos, gradPos, learningRate);

            if (step % 100 == 0) {
                System.out.printf("step %d | loss %.4f%n", step, loss);

                String sample = generate(
                    tok,
                    emb,
                    attn,
                    ffn1,
                    ffnAct,
                    ffn2,
                    outProj,
                    pos,
                    contextLen,
                    dModel,
                    "h",
                    60
                );

                System.out.println("sample: " + sample.replace("\n", "\\n"));
                System.out.println();
            }
        }
    }

    private static String generate(CharTokenizer tok,
                                   Embedding emb,
                                   CausalSelfAttention attn,
                                   Linear ffn1,
                                   ReLU ffnAct,
                                   Linear ffn2,
                                   Linear outProj,
                                   Matrix pos,
                                   int contextLen,
                                   int dModel,
                                   String prompt,
                                   int maxNewChars) {

        StringBuilder out = new StringBuilder(prompt);

        for (int i = 0; i < maxNewChars; i++) {
            int[] ctx = new int[contextLen];

            int start = out.length() - contextLen;
            for (int j = 0; j < contextLen; j++) {
                int charIndex = start + j;
                if (charIndex < 0) {
                    ctx[j] = CharTokenizer.PAD_ID;
                } else {
                    ctx[j] = tok.charToId(out.charAt(charIndex));
                }
            }

            int[][] ctxBatch = new int[][] { ctx };
            Matrix xSeq = emb.forwardSeq(ctxBatch);
            addPositionalEmbeddings(xSeq, pos, 1, contextLen, dModel);

            Matrix attnOnly = attn.forward(xSeq);
            Matrix attnOutSeq = attnOnly.add(xSeq);

            Matrix ffnHidden = ffn1.forward(attnOutSeq);
            ffnHidden = ffnAct.forward(ffnHidden);
            Matrix blockOut = ffn2.forward(ffnHidden).add(attnOutSeq);
            
            Matrix last = takeLastToken(blockOut, 1, contextLen, dModel);

            Matrix logits = outProj.forward(last);
            Matrix probs = logits.softmaxRows();

            int nextId = argmaxRow(probs, 0);
            Character nextChar = tok.idToChar(nextId);

            if (nextChar == null) {
                break;
            }

            out.append(nextChar);

            if (nextChar == '\n') {
                break;
            }
        }

        return out.toString();
    }

    private static Matrix initPositionalEmbeddings(int contextLen, int dModel, Random rng) {
        Matrix pos = new Matrix(contextLen, dModel);

        for (int t = 0; t < contextLen; t++) {
            for (int j = 0; j < dModel; j++) {
                pos.set(t, j, rng.nextGaussian() * 0.01);
            }
        }

        return pos;
    }

    private static void addPositionalEmbeddings(Matrix xSeq, Matrix pos, int batchSize, int contextLen, int dModel) {
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int row = b * contextLen + t;

                for (int j = 0; j < dModel; j++) {
                    xSeq.set(row, j, xSeq.get(row, j) + pos.get(t, j));
                }
            }
        }
    }

    private static Matrix accumulatePosGradients(Matrix dXSeq, int batchSize, int contextLen, int dModel) {
        Matrix gradPos = new Matrix(contextLen, dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int row = b * contextLen + t;

                for (int j = 0; j < dModel; j++) {
                    gradPos.set(t, j, gradPos.get(t, j) + dXSeq.get(row, j));
                }
            }
        }

        return gradPos;
    }

    private static void updatePositionalEmbeddings(Matrix pos, Matrix gradPos, double learningRate) {
        for (int t = 0; t < pos.getRows(); t++) {
            for (int j = 0; j < pos.getCols(); j++) {
                pos.set(t, j, pos.get(t, j) - learningRate * gradPos.get(t, j));
            }
        }
    }

    private static Matrix takeLastToken(Matrix seq, int batchSize, int contextLen, int dModel) {
        // select the final token representation for each sequence
        Matrix last = new Matrix(batchSize, dModel);

        for (int b = 0; b < batchSize; b++) {
            int row = b * contextLen + (contextLen - 1);

            for (int j = 0; j < dModel; j++) {
                last.set(b, j, seq.get(row, j));
            }
        }

        return last;
    }

    private static int argmaxRow(Matrix m, int row) {
        int bestIdx = 0;
        double bestVal = m.get(row, 0);

        for (int j = 1; j < m.getCols(); j++) {
            double v = m.get(row, j);

            if (v > bestVal) {
                bestVal = v;
                bestIdx = j;
            }
        }

        return bestIdx;
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

    private static int[] flattenTargets(int[][] ySeq) {
        int batchSize = ySeq.length;
        int contextLen = ySeq[0].length;
        int[] flat = new int[batchSize * contextLen];

        int idx = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                flat[idx++] = ySeq[b][t];
            }
        }

        return flat;
    }

    private static double maskedCrossEntropy(Matrix probs, int[] targets, int padId) {
        double eps = 1e-12;
        double loss = 0.0;
        int count = 0;

        for (int i = 0; i < targets.length; i++) {
            int target = targets[i];

            if (target == padId) {
                continue;
            }

            double p = probs.get(i, target);
            loss += -Math.log(Math.max(p, eps));
            count++;
        }

        return count == 0 ? 0.0 : loss / count;
    }

    private static Matrix maskedSoftmaxCrossEntropyGrad(Matrix probs, int[] targets, int padId) {
        Matrix grad = new Matrix(probs.getRows(), probs.getCols());

        int count = 0;
        for (int target : targets) {
            if (target != padId) {
                count++;
            }
        }

        if (count == 0) {
            return grad;
        }

        for (int i = 0; i < targets.length; i++) {
            int target = targets[i];

            if (target == padId) {
                continue;
            }

            for (int j = 0; j < probs.getCols(); j++) {
                grad.set(i, j, probs.get(i, j));
            }

            grad.set(i, target, grad.get(i, target) - 1.0);
        }

        double scale = 1.0 / count;

        for (int i = 0; i < grad.getRows(); i++) {
            for (int j = 0; j < grad.getCols(); j++) {
                grad.set(i, j, grad.get(i, j) * scale);
            }
        }

        return grad;
    }

    private static SequenceBatch buildFullSequenceBatch(TextDataset ds, int contextLen) {
        int size = ds.size();
        int[][] x = new int[size][];
        int[][] ySeq = new int[size][contextLen];

        for (int i = 0; i < size; i++) {
            int[] ctx = ds.getContext(i);
            x[i] = ctx;

            // Shift the context left to create next-token targets for each position
            for (int t = 0; t < contextLen - 1; t++) {
                ySeq[i][t] = ctx[t + 1];
            }

            // Final target is the dataset's usual next token
            ySeq[i][contextLen - 1] = ds.getTarget(i);
        }

        return new SequenceBatch(x, ySeq);
    }

    private static class SequenceBatch {
        int[][] x;
        int[][] ySeq;

        SequenceBatch(int[][] x, int[][] ySeq) {
            this.x = x;
            this.ySeq = ySeq;
        }
    }
}