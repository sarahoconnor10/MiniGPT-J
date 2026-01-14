package minigptj;

import minigptj.core.Activation;
import minigptj.core.Matrix;
import minigptj.core.SimpleMLP;

public class Runner {
    public static void main(String[] args) {
        SimpleMLP mlp = new SimpleMLP(4, 5, 3);

        // One example, 4 features
        Matrix input = new Matrix(new double[][]{
            {1.0, 0.5, -1.0, 2.0}
        });

        Matrix logits = mlp.forward(input);
        System.out.println("Logits:");
        System.out.println(logits);

        Matrix probs = mlp.forwardWithSoftmax(input);
        System.out.println("Probabilities:");
        System.out.println(probs);
    }
}
