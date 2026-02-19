package minigptj.model;

import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.core.ReLU;

public class MLPLanguageModel {
    private final Linear layer1;
    private final ReLU relu;
    private final Linear layer2;

    // cached forward activations (for backward flow)
    private Matrix lastHidden;

    public MLPLanguageModel(int inputSize, int hiddenSize, int outputSize) {
        this.layer1 = new Linear(inputSize, hiddenSize);
        this.relu = new ReLU();
        this.layer2 = new Linear(hiddenSize, outputSize);
    }

    /** Forward pass returning logits (batchSize x vocabSize). */
    public Matrix forward(Matrix x) {
        Matrix h = layer1.forward(x);
        h = relu.forward(h);
        this.lastHidden = h;
        return layer2.forward(h);
    }

    public Matrix backward(Matrix dLogits) {
    Matrix dH = layer2.backward(dLogits);
    dH = relu.backward(dH);
    Matrix dX = layer1.backward(dH);
    return dX;
}

    public Linear getLayer1() { return layer1; }
    public Linear getLayer2() { return layer2; }
}