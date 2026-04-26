package minigptj;

import java.io.*;
import minigptj.core.Linear;
import minigptj.core.Matrix;
import minigptj.model.CausalSelfAttention;
import minigptj.model.Embedding;

/**
 * ModelIO — save and load all trained model weights to/from a binary file.
 *
 * Format (sequential, no headers):
 *   For each matrix: rows (int), cols (int), then all doubles row-major.
 *
 * Save order (must match load order exactly):
 *   1. Embedding weights
 *   2. Attention Wq weights + bias
 *   3. Attention Wk weights + bias
 *   4. Attention Wv weights + bias
 *   5. Attention Wo weights + bias
 *   6. FFN layer 1 weights + bias
 *   7. FFN layer 2 weights + bias
 *   8. Output projection weights + bias
 *   9. Positional embeddings matrix
 */
public class ModelIO {

    /**
     * Save the full model to a file.
     *
     * @param path     file path to write to (e.g. "model.bin")
     * @param emb      trained Embedding layer
     * @param attn     trained CausalSelfAttention layer
     * @param ffn1     first feed-forward Linear layer
     * @param ffn2     second feed-forward Linear layer
     * @param outProj  output projection Linear layer
     * @param pos      positional embedding matrix (contextLen x dModel)
     */
    public static void save(String path,
                            Embedding emb,
                            CausalSelfAttention attn,
                            Linear ffn1,
                            Linear ffn2,
                            Linear outProj,
                            Matrix pos) throws IOException {

        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path)))) {

            writeMatrix(dos, emb.getWeights());

            writeMatrix(dos, attn.getWq().getWeights());
            writeMatrix(dos, attn.getWq().getBias());
            writeMatrix(dos, attn.getWk().getWeights());
            writeMatrix(dos, attn.getWk().getBias());
            writeMatrix(dos, attn.getWv().getWeights());
            writeMatrix(dos, attn.getWv().getBias());
            writeMatrix(dos, attn.getWo().getWeights());
            writeMatrix(dos, attn.getWo().getBias());

            writeMatrix(dos, ffn1.getWeights());
            writeMatrix(dos, ffn1.getBias());
            writeMatrix(dos, ffn2.getWeights());
            writeMatrix(dos, ffn2.getBias());

            writeMatrix(dos, outProj.getWeights());
            writeMatrix(dos, outProj.getBias());

            writeMatrix(dos, pos);
        }

        System.out.println("Model saved to: " + path);
    }

    /**
     * Load a saved model from file, writing weights directly into the
     * provided (already-constructed) layer objects.
     *
     * @param path     file path to read from (e.g. "model.bin")
     * @param emb      Embedding layer to populate
     * @param attn     CausalSelfAttention layer to populate
     * @param ffn1     first feed-forward Linear layer to populate
     * @param ffn2     second feed-forward Linear layer to populate
     * @param outProj  output projection Linear layer to populate
     * @param pos      positional embedding matrix to populate
     */
    public static void load(String path,
                            Embedding emb,
                            CausalSelfAttention attn,
                            Linear ffn1,
                            Linear ffn2,
                            Linear outProj,
                            Matrix pos) throws IOException {

        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path)))) {

            readInto(dis, emb.getWeights());

            readInto(dis, attn.getWq().getWeights());
            readInto(dis, attn.getWq().getBias());
            readInto(dis, attn.getWk().getWeights());
            readInto(dis, attn.getWk().getBias());
            readInto(dis, attn.getWv().getWeights());
            readInto(dis, attn.getWv().getBias());
            readInto(dis, attn.getWo().getWeights());
            readInto(dis, attn.getWo().getBias());

            readInto(dis, ffn1.getWeights());
            readInto(dis, ffn1.getBias());
            readInto(dis, ffn2.getWeights());
            readInto(dis, ffn2.getBias());

            readInto(dis, outProj.getWeights());
            readInto(dis, outProj.getBias());

            readInto(dis, pos);
        }

        System.out.println("Model loaded from: " + path);
    }

    // --- private helpers ---

    /** Write a matrix as: rows (int), cols (int), then all doubles row-major. */
    private static void writeMatrix(DataOutputStream dos, Matrix m) throws IOException {
        int rows = m.getRows();
        int cols = m.getCols();
        dos.writeInt(rows);
        dos.writeInt(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dos.writeDouble(m.get(i, j));
            }
        }
    }

    /**
     * Read a matrix from the stream and write values into an existing Matrix.
     * Verifies that the saved dimensions match the target matrix dimensions.
     */
    private static void readInto(DataInputStream dis, Matrix target) throws IOException {
        int rows = dis.readInt();
        int cols = dis.readInt();

        if (rows != target.getRows() || cols != target.getCols()) {
            throw new IOException(String.format(
                "Dimension mismatch loading model: file has %dx%d, expected %dx%d",
                rows, cols, target.getRows(), target.getCols()));
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                target.set(i, j, dis.readDouble());
            }
        }
    }
}