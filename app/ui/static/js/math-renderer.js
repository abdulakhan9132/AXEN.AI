(function (global) {
  'use strict';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function plainMath(value) {
    return value.replace(/\*\*/g, '^').replace(/(\d)\*([a-zA-Z])/g, '$1$2').replace(/\*/g, '\\cdot ').replace(/×/g, '\\times ').replace(/−/g, '-');
  }
  function protect(source) {
    const math = [];
    function token(expression, display) {
      const index = math.push({expression, display}) - 1;
      return '%%AXENMATH' + index + '%%';
    }
    let text = source.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)/g,
      (match, block, bracket, inline, dollar) => token(block || bracket || inline || dollar, !!(block || bracket)));
    // Only standalone mathematical expressions, never prose or code. Keep
    // ambiguous x4 / x*3 literal; do not invent powers absent from the answer.
    text = text.replace(/^([^\n]+)$/gm, line => {
      const expression = line.trim();
      if (expression.includes('%%') || expression.length > 1000 || !/[\d]/.test(expression)
          || !/[=+*/^−×]/.test(expression) || /[A-Za-z]{3,}/.test(expression)
          || !/^[\w\s'()+\-−*/^=.,{}×:]+$/.test(expression)) return line;
      return token(plainMath(expression), true);
    });
    return {text, restore(html) {
      math.forEach((item, index) => {
        const tag = item.display ? 'div' : 'span';
        const markup = `<${tag} class="aexn-math" data-latex="${escape(item.expression)}" data-display="${item.display}">${escape(item.expression)}</${tag}>`;
        html = html.replaceAll('%%AXENMATH' + index + '%%', markup);
      });
      return html;
    }};
  }
  function render(root) {
    if (!global.katex || !root) return;
    root.querySelectorAll('.aexn-math:not([data-rendered])').forEach(node => {
      node.dataset.rendered = 'true';
      if (node.dataset.latex.length > 2000) return;
      try {
        global.katex.render(node.dataset.latex, node, {displayMode: node.dataset.display === 'true',
          output: 'htmlAndMathml', throwOnError: false, trust: false, strict: 'ignore', maxExpand: 1000, maxSize: 20, macros: {}});
      } catch (_) { node.textContent = node.dataset.latex; }
    });
  }
  global.AxenMath = {protect, render};
})(window);
