package minigptj.core;

public class Activation {
    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }
}
