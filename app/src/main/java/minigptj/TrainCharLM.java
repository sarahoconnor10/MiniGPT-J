package minigptj;

import java.util.Random;

import minigptj.core.LossFunctions;
import minigptj.core.Matrix;
import minigptj.data.CharTokenizer;
import minigptj.data.TextDataset;
import minigptj.model.Embedding;
import minigptj.model.MLPLanguageModel;
import minigptj.optim.SGD;

public class TrainCharLM {

    public static void main(String[] args) {
        String text = "Hello world\n";


        CharTokenizer tok = CharTokenizer.fromText(text);
        int[] tokens = tok.encode(text);
        int vocabSize = tok.vocabSize();

        int contextLen = 3;
        TextDataset ds = new TextDataset(tokens, contextLen);

        // Embedding config
        int dModel = 32;

        int inputSize = contextLen * dModel;
        int hiddenSize = 32;
        int outputSize = vocabSize;

        Embedding emb = new Embedding(vocabSize, dModel);
        MLPLanguageModel model = new MLPLanguageModel(inputSize, hiddenSize, outputSize);
        SGD opt = new SGD(0.1);

        Random rng = new Random(42);
        int steps = 2000;
        int batchSize = 16;

        for (int step = 1; step <= steps; step++) {
            TextDataset.Batch batch = ds.sampleBatch(batchSize, rng);

            // ids -> embeddings (batchSize x (contextLen*dModel))
            Matrix x = emb.forward(batch.x);

            if (step == 1) {
                System.out.println("x = " + x.getRows() + " x " + x.getCols());
                System.out.println("expected inputSize = " + inputSize);
            }

            Matrix logits = model.forward(x);
            Matrix probs = logits.softmaxRows();

            Matrix yTrue = LossFunctions.oneHotFromLabels(batch.y, vocabSize);
            double loss = LossFunctions.crossEntropy(probs, yTrue);

            Matrix dLogits = LossFunctions.softmaxCrossEntropyGrad(probs, yTrue);

            Matrix dX = model.backward(dLogits);

            // backprop into embeddings
            emb.backward(dX);

            // update embeddings + linear layers
            opt.step(emb);
            opt.step(model.getLayer1());
            opt.step(model.getLayer2());

            if (step % 100 == 0) {
                System.out.printf("step %d | loss %.4f%n", step, loss);

                String sample = generate(tok, emb, model, vocabSize, contextLen, "H", 60);
                System.out.println("sample: " + sample.replace("\n", "\\n"));
                System.out.println();
            }
        }
    }

    private static String generate(CharTokenizer tok,
                                   Embedding emb,
                                   MLPLanguageModel model,
                                   int vocabSize,
                                   int contextLen,
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
                    char c = out.charAt(charIndex);
                    ctx[j] = tok.charToId(c);
                }
            }

            int[][] ctxBatch = new int[][] { ctx };

            // ids -> embeddings
            Matrix x = emb.forward(ctxBatch);

            Matrix logits = model.forward(x);
            Matrix probs = logits.softmaxRows();

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