package minigptj;

import minigptj.core.Activation;
import minigptj.core.Matrix;

public class Runner {
    public static void main(String[] args) {
        Matrix m = new Matrix(new double[][] {
            {-1, 2},
            {3, -4}
        });

        Matrix reluResult = m.apply(Activation::relu);
        System.out.println("ReLU:");
        System.out.println(reluResult);
    }
}
