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
        Matrix a = new Matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix b = new Matrix(new double[][]{
                {10, 20},
                {30, 40}
        });

        Matrix c = a.add(b);
        System.out.println("A + B =");
        System.out.println(c);

        
    }
}
