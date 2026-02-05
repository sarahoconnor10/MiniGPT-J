package minigptj.data;

import java.util.*;

/**
 * Simple character-level tokenizer.
 *
 * Builds a vocabulary from training text and assigns each unique char an id.
 * Includes special tokens:
 *  - <PAD> = 0 (padding, useful for batching)
 *  - <UNK> = 1 (unknown character)
 */
public class CharTokenizer {

    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final int PAD_ID = 0;
    public static final int UNK_ID = 1;

    private final Map<Character, Integer> charToId;
    private final List<Character> idToChar; // index -> char (null for specials)
    private final Set<Character> vocabChars; // optional convenience

    private CharTokenizer(Map<Character, Integer> charToId, List<Character> idToChar) {
        this.charToId = Collections.unmodifiableMap(charToId);
        this.idToChar = Collections.unmodifiableList(idToChar);
        this.vocabChars = Collections.unmodifiableSet(new HashSet<>(charToId.keySet()));
    }

    /**
     * Build a deterministic character vocab from text.
     * Vocab chars are sorted by unicode code point.
     */
    public static CharTokenizer fromText(String text) {
        if (text == null) throw new IllegalArgumentException("text cannot be null");

        // Collect unique chars
        Set<Character> chars = new HashSet<>();
        for (int i = 0; i < text.length(); i++) {
            chars.add(text.charAt(i));
        }

        // Sort for deterministic IDs
        List<Character> sorted = new ArrayList<>(chars);
        sorted.sort(Comparator.naturalOrder());

        Map<Character, Integer> c2i = new HashMap<>();
        // idToChar: index 0 and 1 reserved for PAD/UNK
        List<Character> i2c = new ArrayList<>();
        i2c.add(null); // 0 -> PAD
        i2c.add(null); // 1 -> UNK

        int nextId = 2;
        for (Character ch : sorted) {
            c2i.put(ch, nextId);
            i2c.add(ch); // i2c[nextId] = ch
            nextId++;
        }

        return new CharTokenizer(c2i, i2c);
    }

    /** Returns vocabulary size including special tokens. */
    public int vocabSize() {
        return idToChar.size();
    }

    /** True if the character exists in this tokenizer's vocab. */
    public boolean hasChar(char c) {
        return vocabChars.contains(c);
    }

    /** Encode a string into token ids. Unknown chars become UNK_ID. */
    public int[] encode(String s) {
        if (s == null) throw new IllegalArgumentException("input string cannot be null");

        int[] ids = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            Integer id = charToId.get(c);
            ids[i] = (id == null) ? UNK_ID : id;
        }
        return ids;
    }

    /**
     * Decode token ids back into a string.
     * PAD and UNK are handled explicitly:
     *  - PAD is skipped by default 
     *  - UNK becomes '�'
     */
    public String decode(int[] ids) {
        if (ids == null) throw new IllegalArgumentException("ids cannot be null");

        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id == PAD_ID) continue; // skip padding
            if (id == UNK_ID) {
                sb.append('�');
                continue;
            }
            if (id < 0 || id >= idToChar.size()) {
                sb.append('�');
                continue;
            }
            Character c = idToChar.get(id);
            sb.append(c == null ? '�' : c);
        }
        return sb.toString();
    }

    /** Convert a token id back to a char, or null for special/invalid ids. */
    public Character idToChar(int id) {
        if (id <= UNK_ID || id >= idToChar.size()) return null;
        return idToChar.get(id);
    }

    /** Convert a char to id, or UNK_ID if missing. */
    public int charToId(char c) {
        return charToId.getOrDefault(c, UNK_ID);
    }

    /** For debugging / printing. */
    public Map<Character, Integer> getCharToId() {
        return charToId;
    }
}