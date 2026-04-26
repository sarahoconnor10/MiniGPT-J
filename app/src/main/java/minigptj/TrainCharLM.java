package minigptj;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.core.ReLU;
import minigptj.data.CharTokenizer;
import minigptj.data.TextDataset;
import minigptj.model.CausalSelfAttention;
import minigptj.model.Embedding;
import minigptj.optim.Adam;
import minigptj.ModelIO;

/**
 * End-to-end training pipeline for MiniGPT-J.
 *
 * This class coordinates the full language model training process:
 *
 * - loading and tokenising the training corpus
 * - constructing training batches
 * - embedding tokens and positions
 * - running the transformer block forward pass
 * - computing cross-entropy loss
 * - performing backpropagation
 * - updating parameters using the Adam optimiser
 * - generating text samples during training
 *
 * The model is trained autoregressively using a next-character prediction
 * objective on a character-level text corpus.
 */
public class TrainCharLM {

    /**
     * Trains the character-level language model and periodically prints samples.
     *
     * @param args command-line arguments, unused
     * @throws Exception if the training corpus cannot be read
     */
    public static void main(String[] args) throws Exception {
        // Load training corpus and build tokenizer
        String text = Files.readString(Path.of(System.getProperty("user.dir"), "app/src/main/java/minigptj/data/grimm_samples.txt"));        Random sampleRng = new Random(42);
        Random batchRng = new Random(123);

        CharTokenizer tok = CharTokenizer.fromText(text);
        int[] tokens = tok.encode(text);
        int vocabSize = tok.vocabSize();

        // Configure training hyperparameters
        int contextLen = 32;
        int dModel = 96;

        TextDataset ds = new TextDataset(tokens, contextLen);

        int batchSize = 64;
        int steps = 5000;
        double learningRate = 0.02;

        // Initialise model components
        Embedding emb = new Embedding(vocabSize, dModel);
        CausalSelfAttention attn = new CausalSelfAttention(dModel, contextLen);
        Linear outProj = new Linear(dModel, vocabSize);

        Adam opt = new Adam(0.001);

        // Position-wise feed-forward network used after attention.
        Linear ffn1 = new Linear(dModel, dModel * 4);
        ReLU ffnAct = new ReLU();
        Linear ffn2 = new Linear(dModel * 4, dModel);

        // Learned positional embeddings are updated manually because they are
        // stored as a raw Matrix rather than inside a layer class.
        Matrix pos = initPositionalEmbeddings(contextLen, dModel, new Random(123));

        // -- Training loop --
        for (int step = 1; step <= steps; step++) {
            // Sample a mini-batch of context windows and target sequences.
            SequenceBatch batch = sampleBatch(ds, contextLen, batchSize, batchRng);

            // Convert token IDs into dense embedding vectors.
            Matrix xSeq = emb.forwardSeq(batch.x);

            // Add learned positional embeddings so the model can represent order.
            addPositionalEmbeddings(xSeq, pos, batchSize, contextLen, dModel);

            if (step == 1) {
                System.out.println("xSeq = " + xSeq.getRows() + " x " + xSeq.getCols());
                System.out.println("expected = " + (batchSize * contextLen) + " x " + dModel);
            }

            // -- Forward pass through transformer-style block --

            // Causal self-attention allows each token to attend only to previous
            // tokens and itself.
            Matrix attnOnly = attn.forward(xSeq);

            // Residual connection around attention.
            Matrix attnOutSeq = attnOnly.add(xSeq);

            // Position-wise feed-forward network.
            Matrix ffnHidden = ffn1.forward(attnOutSeq);
            ffnHidden = ffnAct.forward(ffnHidden);
            Matrix ffnOut = ffn2.forward(ffnHidden);

            // Residual connection around feed-forward network.
            Matrix blockOut = ffnOut.add(attnOutSeq);

            // Project transformer outputs into vocabulary logits.
            Matrix logits = outProj.forward(blockOut);
            Matrix probs = logits.softmaxRows();

            // -- Loss calculation --

            // Full-sequence supervision:
            // logits shape = (batchSize * contextLen) x vocabSize
            // targets are flattened to align with the flattened sequence rows.
            int[] flatTargets = flattenTargets(batch.ySeq);

            double loss = maskedCrossEntropy(probs, flatTargets, CharTokenizer.PAD_ID);
            Matrix dLogits = maskedSoftmaxCrossEntropyGrad(probs, flatTargets, CharTokenizer.PAD_ID);

            // -- Backward pass --

            Matrix dBlockOut = outProj.backward(dLogits);

            // blockOut = ffnOut + attnOutSeq
            // The upstream gradient flows to both branches of the residual add.
            Matrix dFfnOut = dBlockOut;
            Matrix dAttnOutSeq = dBlockOut;

            // Backprop through feed-forward network.
            Matrix dHidden = ffn2.backward(dFfnOut);
            dHidden = ffnAct.backward(dHidden);
            Matrix dFfnInput = ffn1.backward(dHidden);

            // Add gradient from FFN input path into the attention output path.
            dAttnOutSeq = dAttnOutSeq.add(dFfnInput);

            // attnOutSeq = attn.forward(xSeq) + xSeq
            // Gradient flows through both attention and residual branch.
            Matrix dXSeq = attn.backward(dAttnOutSeq).add(dAttnOutSeq);

            // Backprop into token embeddings.
            emb.backwardSeq(dXSeq);

            // Positional embeddings are updated manually.
            Matrix gradPos = accumulatePosGradients(dXSeq, batchSize, contextLen, dModel);

            // Print diagnostic gradient norms on the first step.
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

            // -- Parameter updates --
            opt.tick();

            opt.step(emb);
            opt.step(attn.getWq());
            opt.step(attn.getWk());
            opt.step(attn.getWv());
            opt.step(attn.getWo());
            opt.step(outProj);
            opt.step(ffn1);
            opt.step(ffn2);

            updatePositionalEmbeddings(pos, gradPos, learningRate);

            // -- Progress logging and text generation --
            if (step % 200 == 0) {
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
                    "The ",
                    80,
                    1.1,
                    sampleRng
                );

                System.out.println("sample: " + sample.replace("\n", "\\n"));
                System.out.println();
            }
        }
        ModelIO.save("model.bin", emb, attn, ffn1, ffn2, outProj, pos);
        System.out.println("Training complete. Model saved to model.bin");
    }

    /**
     * Generates text autoregressively from a prompt.
     *
     * At each step, the most recent contextLen characters are encoded and passed
     * through the model. The model predicts a probability distribution over the
     * next character, one character is sampled, and that character is appended
     * to the output.
     *
     * @param tok tokenizer used to convert between characters and token IDs
     * @param emb token embedding layer
     * @param attn causal self-attention layer
     * @param ffn1 first feed-forward layer
     * @param ffnAct ReLU activation for the feed-forward network
     * @param ffn2 second feed-forward layer
     * @param outProj output projection layer
     * @param pos learned positional embeddings
     * @param contextLen fixed context window length
     * @param dModel embedding dimension
     * @param prompt initial text prompt
     * @param maxNewChars maximum number of characters to generate
     * @param temperature sampling temperature controlling randomness
     * @param rng random generator used for sampling
     * @return generated text including the original prompt
     */
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
                                   int maxNewChars,
                                   double temperature,
                                   Random rng) {

        StringBuilder out = new StringBuilder(prompt);

        for (int i = 0; i < maxNewChars; i++) {
            int[] ctx = new int[contextLen];

            // Build context from the most recent characters.
            // If the prompt is shorter than contextLen, left-pad with PAD tokens.
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

            // Only the final token position is used to predict the next character.
            Matrix last = takeLastToken(blockOut, 1, contextLen, dModel);

            Matrix logits = outProj.forward(last);
            Matrix probs = softmaxWithTemperature(logits, temperature);

            int nextId = sampleRow(probs, 0, rng);

            Character nextChar = tok.idToChar(nextId);

            // Stop if the model predicts a special token.
            if (nextChar == null) {
                break;
            }

            out.append(nextChar);

            // Stop early at a newline to keep samples readable.
            if (nextChar == '\n') {
                break;
            }
        }

        return out.toString();
    }

    /**
     * Initialises learned positional embeddings with small random values.
     *
     * @param contextLen number of positions in each context window
     * @param dModel embedding dimension
     * @param rng random generator
     * @return positional embedding matrix of shape contextLen x dModel
     */
    private static Matrix initPositionalEmbeddings(int contextLen, int dModel, Random rng) {
        Matrix pos = new Matrix(contextLen, dModel);

        for (int t = 0; t < contextLen; t++) {
            for (int j = 0; j < dModel; j++) {
                pos.set(t, j, rng.nextGaussian() * 0.01);
            }
        }

        return pos;
    }

    /**
     * Adds learned positional embeddings to token embeddings in-place.
     *
     * Self-attention alone has no built-in understanding of token order, so
     * positional embeddings provide a learned representation of each position
     * in the context window.
     *
     * @param xSeq sequence embeddings of shape (batchSize * contextLen) x dModel
     * @param pos positional embeddings of shape contextLen x dModel
     * @param batchSize number of sequences in the batch
     * @param contextLen number of tokens per sequence
     * @param dModel embedding dimension
     */
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

    /**
     * Accumulates gradients for positional embeddings.
     *
     * Since the same positional embedding row is reused across every batch item,
     * gradients for each position are summed across the batch.
     *
     * @param dXSeq gradient with respect to sequence input embeddings
     * @param batchSize number of sequences in the batch
     * @param contextLen number of tokens per sequence
     * @param dModel embedding dimension
     * @return gradient matrix for positional embeddings
     */
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

    /**
     * Updates positional embeddings using a simple SGD-style update.
     *
     * Positional embeddings are stored directly as a Matrix, so they are updated
     * manually rather than through the Adam optimiser.
     *
     * @param pos positional embedding matrix
     * @param gradPos gradients for positional embeddings
     * @param learningRate learning rate used for the update
     */
    private static void updatePositionalEmbeddings(Matrix pos, Matrix gradPos, double learningRate) {
        for (int t = 0; t < pos.getRows(); t++) {
            for (int j = 0; j < pos.getCols(); j++) {
                pos.set(t, j, pos.get(t, j) - learningRate * gradPos.get(t, j));
            }
        }
    }

    /**
     * Extracts the final token representation from each sequence.
     *
     * This is used during generation, where only the final context position is
     * needed to predict the next character.
     *
     * @param seq sequence matrix of shape (batchSize * contextLen) x dModel
     * @param batchSize number of sequences in the batch
     * @param contextLen number of tokens per sequence
     * @param dModel embedding dimension
     * @return matrix containing only the last token representation per batch item
     */
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

    /**
     * Returns the index of the largest value in a row.
     *
     * This is useful for deterministic decoding, although the final training
     * pipeline uses probabilistic sampling instead.
     *
     * @param m matrix to search
     * @param row row index
     * @return column index containing the maximum value
     */
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

    /**
     * Computes the L2 norm of all values in a matrix.
     *
     * This is used for diagnostic gradient logging during the first training
     * step.
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
     * Flattens a batch of target sequences into a single vector.
     *
     * This aligns the target IDs with the flattened model output layout:
     * (batchSize * contextLen) x vocabSize.
     *
     * @param ySeq target sequences of shape batchSize x contextLen
     * @return flattened target IDs of length batchSize * contextLen
     */
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

    /**
     * Computes cross-entropy loss while ignoring padding targets.
     *
     * Padding positions are excluded so that left-padding at the start of
     * sequences does not affect the training signal.
     *
     * @param probs predicted probabilities from softmax
     * @param targets flattened target token IDs
     * @param padId token ID used for padding
     * @return average cross-entropy loss over non-padding targets
     */
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

    /**
     * Computes the gradient of softmax cross-entropy loss while ignoring padding.
     *
     * For non-padding positions, the gradient is:
     *   probs - oneHot(target)
     *
     * The result is divided by the number of non-padding targets so the gradient
     * scale is independent of how many valid positions are in the batch.
     *
     * @param probs predicted probabilities from softmax
     * @param targets flattened target token IDs
     * @param padId token ID used for padding
     * @return gradient with respect to logits
     */
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

    /**
     * Applies temperature-scaled softmax to logits.
     *
     * Lower temperatures make the distribution sharper and more deterministic.
     * Higher temperatures make the distribution flatter and more random.
     *
     * @param logits raw model outputs
     * @param temperature positive sampling temperature
     * @return probability distribution over the vocabulary
     */
    private static Matrix softmaxWithTemperature(Matrix logits, double temperature) {
        if (temperature <= 0.0) {
            throw new IllegalArgumentException("temperature must be > 0");
        }

        Matrix probs = new Matrix(logits.getRows(), logits.getCols());

        for (int i = 0; i < logits.getRows(); i++) {
            double max = Double.NEGATIVE_INFINITY;

            for (int j = 0; j < logits.getCols(); j++) {
                double v = logits.get(i, j) / temperature;
                if (v > max) {
                    max = v;
                }
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

    /**
     * Samples a token index from a probability distribution.
     *
     * This allows generated text to vary between runs, rather than always taking
     * the highest-probability token.
     *
     * @param probs probability matrix
     * @param row row to sample from
     * @param rng random generator
     * @return sampled token ID
     */
    private static int sampleRow(Matrix probs, int row, Random rng) {
        double r = rng.nextDouble();
        double cumulative = 0.0;

        for (int j = 0; j < probs.getCols(); j++) {
            cumulative += probs.get(row, j);
            if (r <= cumulative) {
                return j;
            }
        }

        return probs.getCols() - 1;
    }

    /**
     * Builds a full dataset batch containing every example.
     *
     * This helper is useful for evaluation or debugging, but the main training
     * loop uses random mini-batches instead.
     *
     * @param ds dataset to read from
     * @param contextLen context window length
     * @return full sequence batch
     */
    private static SequenceBatch buildFullSequenceBatch(TextDataset ds, int contextLen) {
        int size = ds.size();
        int[][] x = new int[size][];
        int[][] ySeq = new int[size][contextLen];

        for (int i = 0; i < size; i++) {
            int[] ctx = ds.getContext(i);
            x[i] = ctx;

            for (int t = 0; t < contextLen - 1; t++) {
                ySeq[i][t] = ctx[t + 1];
            }

            ySeq[i][contextLen - 1] = ds.getTarget(i);
        }

        return new SequenceBatch(x, ySeq);
    }

    /**
     * Samples a random mini-batch for full-sequence supervision.
     *
     * Each input row is a context window. Each target row is the same sequence
     * shifted one position to the left, with the final target supplied by the
     * dataset's next-token label.
     *
     * @param ds source text dataset
     * @param contextLen context window length
     * @param batchSize number of examples to sample
     * @param rng random generator
     * @return sampled sequence batch
     */
    private static SequenceBatch sampleBatch(TextDataset ds, int contextLen, int batchSize, Random rng) {
        int[][] x = new int[batchSize][];
        int[][] ySeq = new int[batchSize][contextLen];

        for (int b = 0; b < batchSize; b++) {
            int idx = rng.nextInt(ds.size());

            int[] ctx = ds.getContext(idx);
            x[b] = ctx;

            for (int t = 0; t < contextLen - 1; t++) {
                ySeq[b][t] = ctx[t + 1];
            }

            ySeq[b][contextLen - 1] = ds.getTarget(idx);
        }

        return new SequenceBatch(x, ySeq);
    }

    /**
     * Simple container for sequence training batches.
     *
     * x contains input context windows.
     * ySeq contains a target token for each position in each context window.
     */
    private static class SequenceBatch {
        int[][] x;
        int[][] ySeq;

        SequenceBatch(int[][] x, int[][] ySeq) {
            this.x = x;
            this.ySeq = ySeq;
        }
    }
}
