package minigptj;

import minigptj.core.Matrix;

public class Runner {
    public static void main(String[] args) {
        Matrix a = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix b = new Matrix(new double[][]{
            {7, 8},
            {9, 10},
            {11, 12}
        });

        Matrix c = a.dot(b);
        System.out.println("A dot B:");
        System.out.println(c);

    }
}
