package minigptj;

import java.util.Arrays;

import minigptj.data.CharTokenizer;
import minigptj.data.TextDataset;

public class Runner {
    public static void main(String[] args) {
        String text = "test\n";

        // 1) Tokenise
        var tok = CharTokenizer.fromText(text);
        int[] ids = tok.encode(text);

        System.out.println("Encoded: " + Arrays.toString(ids));
        System.out.println("Decoded: " + tok.decode(ids));
        System.out.println("Vocab size = " + tok.vocabSize());

        // 2) Build dataset
        int contextLen = 3;
        var ds = new TextDataset(ids, contextLen);

        System.out.println("\nDataset examples (contextLen=" + contextLen + "):");
        for (int i = 0; i < ds.size(); i++) {
            int[] ctx = ds.getContext(i);
            int target = ds.getTarget(i);

            // decode context chars one by one (PAD will show as "_")
            StringBuilder ctxStr = new StringBuilder();
            for (int c : ctx) {
                if (c == CharTokenizer.PAD_ID) ctxStr.append('_');
                else if (c == CharTokenizer.UNK_ID) ctxStr.append('�');
                else ctxStr.append(tok.idToChar(c));
            }

            char targetChar;
            if (target == CharTokenizer.PAD_ID) targetChar = '_';
            else if (target == CharTokenizer.UNK_ID) targetChar = '�';
            else targetChar = tok.idToChar(target);

            System.out.printf(
                "i=%d  ctx=%s  (ids=%s)  ->  y=%c (id=%d)%n",
                i,
                ctxStr,
                Arrays.toString(ctx),
                targetChar,
                target
            );
        }
    }
}
