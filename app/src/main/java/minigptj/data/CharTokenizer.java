package minigptj.data;

import java.util.*;

/**
 * Character-level tokenizer used by MiniGPT-J.
 *
 * The tokenizer builds a vocabulary from the training corpus and assigns
 * each unique character a deterministic integer ID.
 *
 * Special tokens:
 *   PAD_ID = 0 -> padding token used for sequence alignment
 *   UNK_ID = 1 -> unknown character token
 *
 * All other characters are assigned IDs starting from 2.
 */
public class CharTokenizer {

    /** String representation of the padding token. */
    public static final String PAD_TOKEN = "<PAD>";

    /** String representation of the unknown-character token. */
    public static final String UNK_TOKEN = "<UNK>";

    /** Token ID reserved for padding. */
    public static final int PAD_ID = 0;

    /** Token ID reserved for unknown characters. */
    public static final int UNK_ID = 1;

    private final Map<Character, Integer> charToId;
    private final List<Character> idToChar; // index -> char (null for specials)
    private final Set<Character> vocabChars; // optional convenience

    /**
     * Creates an immutable tokenizer instance.
     */
    private CharTokenizer(Map<Character, Integer> charToId, List<Character> idToChar) {
        this.charToId = Collections.unmodifiableMap(charToId);
        this.idToChar = Collections.unmodifiableList(idToChar);
        this.vocabChars = Collections.unmodifiableSet(new HashSet<>(charToId.keySet()));
    }

    /**
     * Builds a tokenizer vocabulary from training text.
     *
     * Characters are sorted by Unicode code point so that ID assignment
     * is deterministic across runs.
     *
     * @param text training corpus text
     * @return tokenizer instance
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
            i2c.add(ch);
            nextId++;
        }

        return new CharTokenizer(c2i, i2c);
    }

    /**
     * Returns vocabulary size including special tokens.
     *
     * @return number of token IDs known by this tokenizer
     */
    public int vocabSize() {
        return idToChar.size();
    }

    /**
     * Returns true if the tokenizer vocabulary contains the character.
     *
     * @param c character to check
     * @return true if c is present in the learned vocabulary
     */
    public boolean hasChar(char c) {
        return vocabChars.contains(c);
    }

    /**
     * Encodes a string into token IDs.
     *
     * Unknown characters are mapped to UNK_ID.
     *
     * @param s input string
     * @return token ID array
     */
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
     * Decodes token IDs back into a string.
     *
     * PAD tokens are skipped.
     * Invalid or unknown IDs are replaced with the replacement character.
     *
     * @param ids token ID array
     * @return decoded string
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

    /**
     * Converts a token ID back into its character representation.
     *
     * Returns null for special or invalid token IDs.
     *
     * @param id token ID
     * @return corresponding character or null
     */
    public Character idToChar(int id) {
        if (id <= UNK_ID || id >= idToChar.size()) return null;
        return idToChar.get(id);
    }

    /**
     * Converts a character into its token ID.
     *
     * Unknown characters return UNK_ID.
     *
     * @param c input character
     * @return token ID
     */
    public int charToId(char c) {
        return charToId.getOrDefault(c, UNK_ID);
    }

    /**
     * Returns the character-to-ID mapping.
     *
     * Primarily useful for debugging.
     *
     * @return immutable character-to-ID map
     */
    public Map<Character, Integer> getCharToId() {
        return charToId;
    }
}
