package minigptj.core;

import java.util.function.Function;

/**
 * Basic 2D matrix implementation used throughout MiniGPT-J.
 *
 * This class provides the core numerical operations needed for the model,
 * including matrix addition, scalar multiplication, matrix multiplication,
 * transposition, element-wise function application, and row-wise softmax.
 */
public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] data;

    /**
     * Creates a zero-initialised matrix with the given shape.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    /**
     * Creates a matrix by copying values from a 2D array.
     *
     * The input array must be rectangular.
     *
     * @param data source values
     */
    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("All rows must have same length");
            }
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }


    /**
     * Returns the number of rows.
     *
     * @return row count
     */
    public int getRows() {
        return rows;
    }

    /**
     * Returns the number of columns.
     *
     * @return column count
     */
    public int getCols() {
        return cols;
    }

    /**
     * Returns the value at a matrix position.
     *
     * @param row row index
     * @param col column index
     * @return value stored at the requested position
     */
    public double get(int row, int col) {
        return data[row][col];
    }

    /**
     * Sets the value at a matrix position.
     *
     * @param row row index
     * @param col column index
     * @param value value to store
     */
    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    /**
     * Returns a formatted string representation for debugging.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix(").append(rows).append("x").append(cols).append(")\n");
        for (int i = 0; i < rows; i++) {
            sb.append("[ ");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%.3f ", data[i][j]));
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    /**
     * Adds two matrices element-wise.
     *
     * @param other matrix with the same shape
     * @return new matrix containing this + other
     */
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition");
        }

        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i, j) + other.get(i, j));
            }
        }
        return result;
    }

    /**
     * Multiplies every matrix value by a scalar.
     *
     * @param scalar scalar multiplier
     * @return new scaled matrix
     */
    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * Performs matrix multiplication.
     *
     * If this matrix has shape A x B, the other matrix must have shape B x C.
     * The returned matrix has shape A x C.
     *
     * @param other right-hand matrix
     * @return matrix product
     */
    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible matrix dimensions for dot product");
        }

        Matrix result = new Matrix(this.rows, other.cols);

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    /**
     * Applies a function to every element in the matrix.
     *
     * Used for operations such as applying activation functions.
     *
     * @param func function to apply to each value
     * @return new matrix containing transformed values
     */
    public Matrix apply(Function<Double, Double> func) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func.apply(this.data[i][j]);
            }
        }
        return result;
    }

    /**
     * Transposes the matrix.
     *
     * @return new matrix with rows and columns swapped
     */
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, this.data[i][j]);
            }
        }
        return result;
    }

    /**
     * Applies softmax independently to each row.
     *
     * This converts each row into a probability distribution whose values sum
     * to 1. The row maximum is subtracted before exponentiation for numerical
     * stability.
     *
     * @return row-wise softmax probabilities
     */
    public Matrix softmaxRows() {
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }

            double sumExp = 0.0;
            for (int j = 0; j < cols; j++) {
                double exp = Math.exp(data[i][j] - max);
                result.data[i][j] = exp;
                sumExp += exp;
            }

            for (int j = 0; j < cols; j++) {
                result.data[i][j] /= sumExp;
            }
        }

        return result;
    }
}
