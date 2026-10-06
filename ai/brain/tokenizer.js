// Character tokenizer with a FIXED vocabulary (printable ASCII + newline),
// so adding new text later never breaks the saved weights.
const CHARS = ["\n"];
for (let c = 32; c <= 126; c++) CHARS.push(String.fromCharCode(c));
const MAP = new Map(CHARS.map((c, i) => [c, i]));
const SPACE = MAP.get(" ");

module.exports = {
  VOCAB_SIZE: CHARS.length,
  encode: (s) => Array.from(s, (c) => (MAP.has(c) ? MAP.get(c) : SPACE)),
  decode: (ids) => ids.map((i) => CHARS[i]).join(""),
};
