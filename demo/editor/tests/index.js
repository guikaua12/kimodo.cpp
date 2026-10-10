// Entry point for the directory form `node --test demo/editor/tests/`.
// Node >= 21 treats `--test` arguments as files/globs and runs a directory as a
// module, which resolves to this index.js; it imports every *.test.mjs here so
// the whole suite runs. (`node --test "demo/editor/tests/*.test.mjs"` or a plain
// `node --test` inside demo/editor run the same files directly.)
import fs from 'node:fs';

const dir = new URL('./', import.meta.url);
for (const name of fs.readdirSync(dir).filter((n) => n.endsWith('.test.mjs')).sort()) {
  await import(new URL(name, dir).href);
}
