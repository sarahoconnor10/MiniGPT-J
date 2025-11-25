package minigptj;

import minigptj.core.Matrix;

public class Runner {
    public static void main(String[] args) {
        double[][] raw = {
                {1.0, 2.0},
                {3.0, 4.0}
        };

        Matrix m = new Matrix(raw);
        System.out.println(m);

        m.set(0, 1, 5.0);
        System.out.println("After change:");
        System.out.println(m);
    }
}
