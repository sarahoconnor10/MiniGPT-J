package minigptj.core;

/**
 * Utility class containing common neural network activation functions.
 *
 * Activation functions introduce non-linearity into neural networks,
 * allowing models to learn complex patterns beyond simple linear mappings.
 */
public final class Activation {
    /**
     * Prevent instantiation of utility class.
     */
    private Activation() {
    }

    /**
     * Rectified Linear Unit activation function.
     *
     * Returns the input if positive, otherwise returns 0.
     *
     * @param x input value
     * @return activated output
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }

    /**
     * Sigmoid activation function.
     *
     * Maps input values to the range (0, 1).
     *
     * @param x input value
     * @return activated output
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Hyperbolic tangent activation function.
     *
     * Maps input values to the range (-1, 1).
     *
     * @param x input value
     * @return activated output
     */
    public static double tanh(double x) {
        return Math.tanh(x);
    }
}
