package minigptj;

import java.util.Arrays;

import minigptj.core.Activation;
import minigptj.core.Matrix;
import minigptj.core.SimpleMLP;
import minigptj.data.CharTokenizer;

public class Runner {
    public static void main(String[] args) {
        var tok = CharTokenizer.fromText("test\n");
        int[] ids = tok.encode("test\n");
        System.out.println(Arrays.toString(ids));
        System.out.println(tok.decode(ids));
        System.out.println("vocab size = " + tok.vocabSize());
    }
}
