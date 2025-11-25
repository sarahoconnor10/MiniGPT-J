package minigptj;

import minigptj.core.Activation;
import minigptj.core.Matrix;

public class Runner {
    public static void main(String[] args) {
        Matrix m = new Matrix(new double[][] {
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix t = m.transpose();

        System.out.println("Original:");
        System.out.println(m);

        System.out.println("Transposed:");
        System.out.println(t);
    }
}
