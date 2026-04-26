package minigptj;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.Scanner;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.core.ReLU;
import minigptj.data.CharTokenizer;
import minigptj.data.TextDataset;
import minigptj.model.CausalSelfAttention;
import minigptj.model.Embedding;
import minigptj.optim.Adam;

/**
 * Demo — interactive MiniGPT-J demonstration.
 *
 * Usage (from project root via Gradle):
 *
 *   Generate only (load pretrained model, prompt interactively):
 *     ./gradlew run --args="demo"
 *
 *   Train a few more steps first, then prompt interactively:
 *     ./gradlew run --args="demo --train 300"
 *
 * The model is loaded from model.bin in the project root.
 * model.bin is produced by running the main training (TrainCharLM)
 * after ModelIO.save() has been added to the end of training.
 */
public class Demo {

    // Must match the hyperparameters used during training
    static final int CONTEXT_LEN = 32;
    static final int D_MODEL     = 96;
    static final String MODEL_PATH = "model.bin";
    static final String DATA_PATH  = "app/src/main/java/minigptj/data/grimm_samples.txt";

    public static void main(String[] args) throws Exception {

        // --- parse args ---
        int extraSteps = 0;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--train") && i + 1 < args.length) {
                extraSteps = Integer.parseInt(args[i + 1]);
            }
        }

        // --- load training text and build tokenizer ---
        String text = Files.readString(Path.of(DATA_PATH));
        CharTokenizer tok = CharTokenizer.fromText(text);
        int vocabSize = tok.vocabSize();

        // --- build model (same architecture as training) ---
        Embedding emb       = new Embedding(vocabSize, D_MODEL);
        CausalSelfAttention attn = new CausalSelfAttention(D_MODEL, CONTEXT_LEN);
        Linear ffn1         = new Linear(D_MODEL, D_MODEL * 4);
        ReLU ffnAct         = new ReLU();
        Linear ffn2         = new Linear(D_MODEL * 4, D_MODEL);
        Linear outProj      = new Linear(D_MODEL, vocabSize);
        Matrix pos          = new Matrix(CONTEXT_LEN, D_MODEL);

        // --- load pretrained weights ---
        System.out.println("Loading model from " + MODEL_PATH + " ...");
        ModelIO.load(MODEL_PATH, emb, attn, ffn1, ffn2, outProj, pos);
        System.out.println("Done.\n");

        // --- optional: continue training for a few steps to show live learning ---
        if (extraSteps > 0) {
            System.out.println("Continuing training for " + extraSteps + " steps...");
            System.out.println("(Watch the loss decrease as the model keeps learning)\n");

            int[] tokens = tok.encode(text);
            TextDataset ds = new TextDataset(tokens, CONTEXT_LEN);
            Adam opt = new Adam(0.001);
            Random batchRng = new Random(42);

            for (int step = 1; step <= extraSteps; step++) {
                // --- forward pass ---
                int[][] x      = sampleX(ds, 64, batchRng);
                int[][] ySeq   = buildYSeq(ds, x, CONTEXT_LEN);
                Matrix xSeq    = emb.forwardSeq(x);
                addPos(xSeq, pos, x.length, CONTEXT_LEN, D_MODEL);

                Matrix attnOut  = attn.forward(xSeq).add(xSeq);
                Matrix ffnHid   = ffnAct.forward(ffn1.forward(attnOut));
                Matrix blockOut = ffn2.forward(ffnHid).add(attnOut);
                Matrix logits   = outProj.forward(blockOut);
                Matrix probs    = logits.softmaxRows();

                int[] flatY = flatten(ySeq);
                double loss = maskedCE(probs, flatY, CharTokenizer.PAD_ID);
                Matrix dLogits = maskedCEGrad(probs, flatY, CharTokenizer.PAD_ID);

                // --- backward pass ---
                Matrix dBlock  = outProj.backward(dLogits);
                Matrix dHid    = ffn2.backward(dBlock);
                dHid = ffnAct.backward(dHid);
                Matrix dAttn   = ffn1.backward(dHid);
                dAttn = dAttn.add(dBlock);
                Matrix dXSeq   = attn.backward(dAttn).add(dAttn);
                emb.backwardSeq(dXSeq);

                Matrix gradPos = accumPosGrad(dXSeq, x.length, CONTEXT_LEN, D_MODEL);

                // --- optimiser step ---
                opt.tick();
                opt.step(emb);
                opt.step(attn.getWq());
                opt.step(attn.getWk());
                opt.step(attn.getWv());
                opt.step(attn.getWo());
                opt.step(outProj);
                opt.step(ffn1);
                opt.step(ffn2);
                updatePos(pos, gradPos, 0.02);

                if (step % 50 == 0 || step == 1) {
                    System.out.printf("  step %d / %d  |  loss %.4f%n", step, extraSteps, loss);
                }
            }
            System.out.println("\nTraining complete. Entering generation mode.\n");
        }

        // --- interactive generation loop ---
        printBanner();
        Scanner scanner = new Scanner(System.in);
        Random genRng = new Random();

        while (true) {
            System.out.print("Prompt (or 'quit'): ");
            String prompt = scanner.nextLine().trim();

            if (prompt.equalsIgnoreCase("quit") || prompt.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            if (prompt.isEmpty()) {
                prompt = "The ";
            }

            System.out.print("\nGenerated: ");
            String output = generate(tok, emb, attn, ffn1, ffnAct, ffn2, outProj,
                                     pos, CONTEXT_LEN, D_MODEL, prompt, 200, 1.0, genRng);
            System.out.println(output);
            System.out.println();
        }

        scanner.close();
    }

    // -------------------------------------------------------------------------
    // Generation
    // -------------------------------------------------------------------------

    private static String generate(CharTokenizer tok,
                                   Embedding emb,
                                   CausalSelfAttention attn,
                                   Linear ffn1, ReLU ffnAct, Linear ffn2,
                                   Linear outProj, Matrix pos,
                                   int contextLen, int dModel,
                                   String prompt, int maxNewChars,
                                   double temperature, Random rng) {

        StringBuilder out = new StringBuilder(prompt);

        for (int i = 0; i < maxNewChars; i++) {
            int[] ctx = new int[contextLen];
            int start = out.length() - contextLen;

            for (int j = 0; j < contextLen; j++) {
                int charIndex = start + j;
                ctx[j] = charIndex < 0
                    ? CharTokenizer.PAD_ID
                    : tok.charToId(out.charAt(charIndex));
            }

            int[][] ctxBatch = new int[][] { ctx };
            Matrix xSeq = emb.forwardSeq(ctxBatch);
            addPos(xSeq, pos, 1, contextLen, dModel);

            Matrix attnOut  = attn.forward(xSeq).add(xSeq);
            Matrix ffnHid   = ffnAct.forward(ffn1.forward(attnOut));
            Matrix blockOut = ffn2.forward(ffnHid).add(attnOut);
            Matrix last     = takeLastToken(blockOut, 1, contextLen, dModel);
            Matrix probs    = softmaxTemp(outProj.forward(last), temperature);

            int nextId = sampleRow(probs, 0, rng);
            Character nextChar = tok.idToChar(nextId);
            if (nextChar == null) break;

            out.append(nextChar);
        }

        return out.toString();
    }

    // -------------------------------------------------------------------------
    // Training helpers (mirrors TrainCharLM)
    // -------------------------------------------------------------------------

    private static int[][] sampleX(TextDataset ds, int batchSize, Random rng) {
        int[][] x = new int[batchSize][];
        for (int b = 0; b < batchSize; b++) {
            x[b] = ds.getContext(rng.nextInt(ds.size()));
        }
        return x;
    }

    private static int[][] buildYSeq(TextDataset ds, int[][] x, int contextLen) {
        int batchSize = x.length;
        int[][] ySeq = new int[batchSize][contextLen];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen - 1; t++) {
                ySeq[b][t] = x[b][t + 1];
            }
            // last target: find this context in the dataset (approximate with PAD)
            ySeq[b][contextLen - 1] = CharTokenizer.PAD_ID;
        }
        return ySeq;
    }

    private static int[] flatten(int[][] ySeq) {
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

    private static double maskedCE(Matrix probs, int[] targets, int padId) {
        double eps = 1e-12, loss = 0.0;
        int count = 0;
        for (int i = 0; i < targets.length; i++) {
            if (targets[i] == padId) continue;
            loss += -Math.log(Math.max(probs.get(i, targets[i]), eps));
            count++;
        }
        return count == 0 ? 0.0 : loss / count;
    }

    private static Matrix maskedCEGrad(Matrix probs, int[] targets, int padId) {
        Matrix grad = new Matrix(probs.getRows(), probs.getCols());
        int count = 0;
        for (int t : targets) if (t != padId) count++;
        if (count == 0) return grad;
        double scale = 1.0 / count;
        for (int i = 0; i < targets.length; i++) {
            if (targets[i] == padId) continue;
            for (int j = 0; j < probs.getCols(); j++) {
                grad.set(i, j, probs.get(i, j) * scale);
            }
            grad.set(i, targets[i], grad.get(i, targets[i]) - scale);
        }
        return grad;
    }

    // -------------------------------------------------------------------------
    // Positional embedding helpers
    // -------------------------------------------------------------------------

    private static void addPos(Matrix xSeq, Matrix pos, int batchSize, int contextLen, int dModel) {
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < contextLen; t++) {
                int row = b * contextLen + t;
                for (int j = 0; j < dModel; j++) {
                    xSeq.set(row, j, xSeq.get(row, j) + pos.get(t, j));
                }
            }
        }
    }

    private static Matrix accumPosGrad(Matrix dXSeq, int batchSize, int contextLen, int dModel) {
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

    private static void updatePos(Matrix pos, Matrix gradPos, double lr) {
        for (int t = 0; t < pos.getRows(); t++) {
            for (int j = 0; j < pos.getCols(); j++) {
                pos.set(t, j, pos.get(t, j) - lr * gradPos.get(t, j));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sampling helpers
    // -------------------------------------------------------------------------

    private static Matrix takeLastToken(Matrix seq, int batchSize, int contextLen, int dModel) {
        Matrix last = new Matrix(batchSize, dModel);
        for (int b = 0; b < batchSize; b++) {
            int row = b * contextLen + (contextLen - 1);
            for (int j = 0; j < dModel; j++) {
                last.set(b, j, seq.get(row, j));
            }
        }
        return last;
    }

    private static Matrix softmaxTemp(Matrix logits, double temperature) {
        Matrix probs = new Matrix(logits.getRows(), logits.getCols());
        for (int i = 0; i < logits.getRows(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < logits.getCols(); j++) {
                double v = logits.get(i, j) / temperature;
                if (v > max) max = v;
            }
            double sumExp = 0.0;
            for (int j = 0; j < logits.getCols(); j++) {
                double e = Math.exp((logits.get(i, j) / temperature) - max);
                probs.set(i, j, e);
                sumExp += e;
            }
            for (int j = 0; j < logits.getCols(); j++) {
                probs.set(i, j, probs.get(i, j) / sumExp);
            }
        }
        return probs;
    }

    private static int sampleRow(Matrix probs, int row, Random rng) {
        double r = rng.nextDouble(), cumulative = 0.0;
        for (int j = 0; j < probs.getCols(); j++) {
            cumulative += probs.get(row, j);
            if (r <= cumulative) return j;
        }
        return probs.getCols() - 1;
    }

    // -------------------------------------------------------------------------
    // Banner
    // -------------------------------------------------------------------------

    private static void printBanner() {
        System.out.println("╔══════════════════════════════════════╗");
        System.out.println("║         MiniGPT-J  —  Demo           ║");
        System.out.println("║  Character-level transformer in Java  ║");
        System.out.println("╚══════════════════════════════════════╝");
        System.out.println();
        System.out.println("Type a prompt and press Enter to generate text.");
        System.out.println("Type 'quit' to exit.\n");
    }
}