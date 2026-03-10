package minigptj;

import java.util.Random;

import minigptj.core.Linear;
import minigptj.core.LossFunctions;
import minigptj.core.Matrix;
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
            "hello sarah\n" +
            "how are you\n" +
            "how is the world\n";
        
        text = text.repeat(20);

        CharTokenizer tok = CharTokenizer.fromText(text);
        int[] tokens = tok.encode(text);
        int vocabSize = tok.vocabSize();

        int contextLen = 3;
        int dModel = 32;
        
        TextDataset ds = new TextDataset(tokens, contextLen);
        TextDataset.Batch fullBatch = buildFullBatch(ds);
        int batchSize = fullBatch.x.length;
        
        int steps = 6000;
        double learningRate = 0.03;

        Embedding emb = new Embedding(vocabSize, dModel);
        CausalSelfAttention attn = new CausalSelfAttention(dModel, contextLen);
        Linear outProj = new Linear(dModel, vocabSize);
        SGD opt = new SGD(learningRate);

        Matrix pos = initPositionalEmbeddings(contextLen, dModel, new Random(123));
        Random rng = new Random(42);

        for (int step = 1; step <= steps; step++) {
            TextDataset.Batch batch = fullBatch;
            Matrix xSeq = emb.forwardSeq(batch.x);
            addPositionalEmbeddings(xSeq, pos, batchSize, contextLen, dModel);

            if (step == 1) {
                System.out.println("xSeq = " + xSeq.getRows() + " x " + xSeq.getCols());
                System.out.println("expected = " + (batchSize * contextLen) + " x " + dModel);
            }

            Matrix attnOnly = attn.forward(xSeq);
            Matrix attnOutSeq = attnOnly.add(xSeq);

            Matrix last = takeLastToken(attnOutSeq, batchSize, contextLen, dModel);

            Matrix logits = outProj.forward(last);
            Matrix probs = logits.softmaxRows();

            Matrix yTrue = LossFunctions.oneHotFromLabels(batch.y, vocabSize);
            double loss = LossFunctions.crossEntropy(probs, yTrue);
            Matrix dLogits = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);

            Matrix dLast = outProj.backward(dLogits);
            Matrix dAttnOutSeq = scatterLastToken(dLast, batchSize, contextLen, dModel);

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

            updatePositionalEmbeddings(pos, gradPos, learningRate);

            if (step % 100 == 0) {
                System.out.printf("step %d | loss %.4f%n", step, loss);

                String sample = generate(
                    tok,
                    emb,
                    attn,
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
            
            Matrix last = takeLastToken(attnOutSeq, 1, contextLen, dModel);

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

    private static Matrix scatterLastToken(Matrix dLast, int batchSize, int contextLen, int dModel) {
        // place gradients on the final timestep of each sequence
        Matrix dSeq = new Matrix(batchSize * contextLen, dModel);

        for (int b = 0; b < batchSize; b++) {
            int row = b * contextLen + (contextLen - 1);

            for (int j = 0; j < dModel; j++) {
                dSeq.set(row, j, dLast.get(b, j));
            }
        }

        return dSeq;
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

    private static TextDataset.Batch buildFullBatch(TextDataset ds) {
        int size = ds.size();
        int[][] x = new int[size][];
        int[] y = new int[size];

        for (int i = 0; i < size; i++) {
            x[i] = ds.getContext(i);
            y[i] = ds.getTarget(i);
        }

        return new TextDataset.Batch(x, y);
    }
}