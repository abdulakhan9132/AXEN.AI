/**
 * app/ui/static/js/markdown.js
 * ============================
 * Self-contained, zero-dependency Markdown parser with developer-grade
 * code blocks, copy functionality, table support, and HTML sanitization.
 * Works 100% offline.
 */

(function (global) {
  'use strict';

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function parseMarkdown(md) {
    if (!md) return '';

    // Normalize line breaks
    let src = md.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // 1. Extract and protect fenced code blocks
    const codeBlocks = [];
    src = src.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, function (match, lang, code) {
      const id = `%%CODEBLOCK${codeBlocks.length}%%`;
      codeBlocks.push({ lang: lang.trim() || 'code', code: code.trimEnd() });
      return id;
    });

    // 2. Extract and protect inline code
    const inlineCodes = [];
    src = src.replace(/`([^`\n]+)`/g, function (match, code) {
      const id = `%%INLINECODE${inlineCodes.length}%%`;
      inlineCodes.push(escapeHtml(code));
      return id;
    });

    // 3. Tables
    src = src.replace(/(([^\n]+\|[^\n]+\n)+)/g, function (match) {
      const rows = match.trim().split('\n');
      if (rows.length >= 2 && rows[1].includes('---')) {
        let html = '<div class="table-container"><table class="markdown-table">';
        // Header
        const headers = rows[0].split('|').map(s => s.trim()).filter(s => s.length > 0);
        html += '<thead><tr>';
        headers.forEach(h => {
          html += `<th>${escapeHtml(h)}</th>`;
        });
        html += '</tr></thead><tbody>';
        // Rows
        for (let i = 2; i < rows.length; i++) {
          const cells = rows[i].split('|').map(s => s.trim()).filter(s => s.length > 0);
          if (cells.length) {
            html += '<tr>';
            cells.forEach(c => {
              html += `<td>${escapeHtml(c)}</td>`;
            });
            html += '</tr>';
          }
        }
        html += '</tbody></table></div>';
        return html;
      }
      return match;
    });

    // 4. Headings
    src = src.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    src = src.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    src = src.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    src = src.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // 5. Blockquotes
    src = src.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // 6. Emphasis (bold, italic, strikethrough)
    src = src.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    src = src.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    src = src.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    src = src.replace(/_([^_]+)_/g, '<em>$1</em>');
    src = src.replace(/~~([^~]+)~~/g, '<del>$1</del>');

    // 6.5 Images (markdown syntax: ![alt](url))
    src = src.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function (match, alt, url) {
      const cleanAlt = escapeHtml(alt || 'Generated Image');
      return `
        <div class="generated-image-card">
          <img src="${url}" alt="${cleanAlt}" class="generated-image-preview" loading="lazy" />
          <div class="image-card-actions">
            <a href="${url}" download class="image-download-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              <span>Download</span>
            </a>
          </div>
        </div>
      `.trim();
    });

    // 6.6 Standalone /images/... URLs in agent replies
    src = src.replace(/(?:URL:\s*)?(\/images\/[a-zA-Z0-9_\-\.]+\.(?:png|jpg|jpeg|webp))/g, function (match, url) {
      return `
        <div class="generated-image-card">
          <img src="${url}" alt="Generated Image" class="generated-image-preview" loading="lazy" />
          <div class="image-card-actions">
            <a href="${url}" download class="image-download-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              <span>Download</span>
            </a>
          </div>
        </div>
      `.trim();
    });

    // 7. Links
    src = src.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // 8. Lists
    // Unordered lists
    src = src.replace(/^[\*\-] (.*$)/gim, '<ul><li>$1</li></ul>');
    src = src.replace(/<\/ul>\s*<ul>/g, '');

    // Ordered lists
    src = src.replace(/^\d+\. (.*$)/gim, '<ol><li>$1</li></ol>');
    src = src.replace(/<\/ol>\s*<ol>/g, '');

    // 9. Paragraphs and line breaks
    const blocks = src.split('\n\n');
    const processedBlocks = blocks.map(block => {
      block = block.trim();
      if (!block) return '';
      if (
        block.startsWith('<h') ||
        block.startsWith('<ul') ||
        block.startsWith('<ol') ||
        block.startsWith('<blockquote') ||
        block.startsWith('<div class="table-container"') ||
        block.startsWith('<div class="generated-image-card"') ||
        block.startsWith('%%CODEBLOCK')
      ) {
        return block;
      }
      return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
    });
    src = processedBlocks.filter(b => b.length > 0).join('\n');

    // 10. Restore inline code
    inlineCodes.forEach((code, index) => {
      src = src.replaceAll(`%%INLINECODE${index}%%`, `<code class="inline-code">${code}</code>`);
    });

    // 11. Restore fenced code blocks with copy button
    codeBlocks.forEach((item, index) => {
      const escapedCode = escapeHtml(item.code);
      const copyId = `code-btn-${Math.random().toString(36).substring(2, 9)}`;
      const blockHtml = `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-block-lang">${escapeHtml(item.lang)}</span>
            <button class="code-copy-btn" id="${copyId}" onclick="window.AxenUI.copyCode(this)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              <span>Copy</span>
            </button>
          </div>
          <pre class="code-block-pre"><code class="code-block-content language-${escapeHtml(item.lang)}">${escapedCode}</code></pre>
        </div>
      `.trim();
      src = src.replaceAll(`%%CODEBLOCK${index}%%`, blockHtml);
    });

    return src;
  }

  global.AxenMarkdown = {
    parse: parseMarkdown,
    escapeHtml: escapeHtml,
  };
})(window);
