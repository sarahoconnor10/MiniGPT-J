package minigptj;

import java.util.Random;

import minigptj.core.LossFunctions;
import minigptj.core.Matrix;
import minigptj.data.CharTokenizer;
import minigptj.data.OneHot;
import minigptj.data.TextDataset;
import minigptj.model.MLPLanguageModel;
import minigptj.optim.SGD;

/**
 * Minimal character-level next-token trainer.
 *
 * Goal:
 *  - Take a text string
 *  - Predict the next character from a small context window
 *  - Train an MLP to minimise cross-entropy loss
 */
public class TrainCharLM {

    public static void main(String[] args) {
        // 1) Training text 
        String text = "Hello world\n";

        // 2) Tokenise
        CharTokenizer tok = CharTokenizer.fromText(text);
        int[] tokens = tok.encode(text);
        int vocabSize = tok.vocabSize();

        // 3) Dataset (context -> next token)
        int contextLen = 3;
        TextDataset ds = new TextDataset(tokens, contextLen);

        // 4) Model + optimiser
        int inputSize = contextLen * vocabSize;
        int hiddenSize = 32;           // small hidden layer
        int outputSize = vocabSize;    // logits over vocab

        MLPLanguageModel model = new MLPLanguageModel(inputSize, hiddenSize, outputSize);
        SGD opt = new SGD(0.1); 

        // 5) Training loop
        Random rng = new Random(42);
        int steps = 2000;
        int batchSize = 16;

        for (int step = 1; step <= steps; step++) {
            // Sample a batch of contexts + targets
            TextDataset.Batch batch = ds.sampleBatch(batchSize, rng);

            // Convert context -> input matrix
            Matrix x = OneHot.encodeContextConcat(batch.x, vocabSize);

            //debugging
            if (step == 1) {
                System.out.println("x = " + x.getRows() + " x " + x.getCols());
                System.out.println("expected inputSize = " + inputSize);
            }

            // Forward: logits -> probs
            Matrix logits = model.forward(x);
            Matrix probs = logits.softmaxRows();

            // Targets: one-hot
            Matrix yTrue = LossFunctions.oneHotFromLabels(batch.y, vocabSize);

            // Loss
            double loss = LossFunctions.crossEntropy(probs, yTrue);

            // Backward: dLogits
            Matrix dLogits = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);
            model.backward(dLogits);

            // Update parameters
            opt.step(model.getLayer1());
            opt.step(model.getLayer2());

            // Print progress
            if (step % 100 == 0) {
                System.out.printf("step %d | loss %.4f%n", step, loss);

                // Quick sample generation 
                String sample = generate(tok, model, vocabSize, contextLen, "H", 60);                
                System.out.println("sample: " + sample.replace("\n", "\\n"));
                System.out.println();
            }
        }
    }

    private static String generate(CharTokenizer tok,
                               MLPLanguageModel model,
                               int vocabSize,
                               int contextLen,
                               String prompt,
                               int maxNewChars) {

        StringBuilder out = new StringBuilder(prompt);

        for (int i = 0; i < maxNewChars; i++) {

            // Build context token IDs from the last contextLen characters
            int[] ctx = new int[contextLen];

            int start = out.length() - contextLen; // where context would start
            for (int j = 0; j < contextLen; j++) {
                int charIndex = start + j;
                if (charIndex < 0) {
                    ctx[j] = CharTokenizer.PAD_ID;
                } else {
                    char c = out.charAt(charIndex);
                    ctx[j] = tok.charToId(c);
                }
            }

            // Batch of 1 context
            int[][] ctxBatch = new int[][]{ ctx };

            // One-hot concat encoding (1 x (contextLen * vocabSize))
            Matrix x = OneHot.encodeContextConcat(ctxBatch, vocabSize);

            Matrix logits = model.forward(x);
            Matrix probs = logits.softmaxRows();

            // Greedy argmax
            int nextId = argmaxRow(probs, 0);

            Character nextChar = tok.idToChar(nextId);
            if (nextChar == null) break;

            out.append(nextChar);

            if (nextChar == '\n') break;
        }

        return out.toString();
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
}