package minigptj.data;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class CharTokenizerTest {

    @Test
    void testVocabSizeIncludesSpecialTokens() {
        CharTokenizer tok = CharTokenizer.fromText("abc");
        // 3 chars + PAD + UNK = 5
        assertEquals(5, tok.vocabSize());
    }

    @Test
    void testEncodeDecodeRoundTrip() {
        CharTokenizer tok = CharTokenizer.fromText("hello");
        int[] ids = tok.encode("hello");
        String decoded = tok.decode(ids);
        assertEquals("hello", decoded);
    }

    @Test
    void testUnknownCharBecomesUnkId() {
        CharTokenizer tok = CharTokenizer.fromText("abc");
        int[] ids = tok.encode("z");
        assertEquals(CharTokenizer.UNK_ID, ids[0]);
    }

    @Test
    void testPadIdIsSkippedInDecode() {
        CharTokenizer tok = CharTokenizer.fromText("ab");
        int[] ids = new int[]{CharTokenizer.PAD_ID, tok.charToId('a')};
        String decoded = tok.decode(ids);
        assertEquals("a", decoded);
    }

    @Test
    void testDeterministicIds() {
        CharTokenizer tok1 = CharTokenizer.fromText("abc");
        CharTokenizer tok2 = CharTokenizer.fromText("abc");
        assertEquals(tok1.charToId('a'), tok2.charToId('a'));
        assertEquals(tok1.charToId('b'), tok2.charToId('b'));
    }

    @Test
    void testIdToCharRoundTrip() {
        CharTokenizer tok = CharTokenizer.fromText("xyz");
        int id = tok.charToId('x');
        assertEquals('x', tok.idToChar(id));
    }
}