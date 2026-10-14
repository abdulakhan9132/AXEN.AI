const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync('app/ui/static/js/markdown.js', 'utf8'), context);
const parse = context.window.AxenMarkdown.parse;

test('generated Markdown image produces exactly one card', () => {
  const html = parse('![Generated image](/images/city.png)');
  assert.equal((html.match(/class="generated-image-card"/g) || []).length, 1);
  assert.equal((html.match(/<img /g) || []).length, 1);
});
test('standalone image path works without rewriting attributes', () => {
  const html = parse('/images/city.png');
  assert.equal((html.match(/<img /g) || []).length, 1);
  assert.ok(html.includes('src="/images/city.png"'));
});
test('generated filenames with underscores remain valid image URLs', () => {
  const html = parse('![Image](/images/a_futuristic_city_at_sunset_123_0.png)');
  assert.equal((html.match(/<img /g) || []).length, 1);
  assert.ok(html.includes('src="/images/a_futuristic_city_at_sunset_123_0.png"'));
});
test('both image reply formats reserve visible space before lazy loading', () => {
  for (const message of ['![Image](/images/tree.png)', '/images/tree.png']) {
    const html = parse(message);
    assert.match(html, /<img [^>]*width="512" height="512"/);
  }
});
test('untrusted model HTML and JavaScript links cannot execute', () => {
  const html = parse('<script>alert(1)</script>\n[bad](javascript:alert)\n![remote](https://example.invalid/tracker.png)');
  assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('href="javascript:'));
  assert.ok(!html.includes('<img'));
});
test('code, diff, blockquotes and copy controls still render', () => {
  const html = parse('> quoted\n\n```diff\n+ fixed\n- broken\n```');
  assert.ok(html.includes('<blockquote>quoted</blockquote>'));
  assert.ok(html.includes('language-diff'));
  assert.ok(html.includes('AxenUI.copyCode'));
  assert.ok(html.includes('+ fixed'));
});
test('code highlighting escapes arbitrary HTML', () => {
  const html = parse('```python\ndef test():\n    return "<script>"\n```');
  assert.ok(html.includes('syntax-keyword'));
  assert.ok(html.includes('syntax-string'));
  assert.ok(!html.includes('<script>'));
});
