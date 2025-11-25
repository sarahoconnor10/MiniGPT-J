package minigptj;

import minigptj.core.Activation;
import minigptj.core.Matrix;

public class Runner {
    public static void main(String[] args) {
        Matrix logits = new Matrix(new double[][] {
            {2.0, 1.0, 0.1},
            {1.0, 3.0, 0.5}
        });

        System.out.println("Logits:");
        System.out.println(logits);

        Matrix probs = logits.softmaxRows();
        System.out.println("Softmax probabilities:");
        System.out.println(probs);
    }
}
