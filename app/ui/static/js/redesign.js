(function () {
  'use strict';
  const byId = id => document.getElementById(id);
  function init() {
    const info = document.querySelector('.info-panel');
    const panel = document.createElement('section');
    panel.id = 'right-workspace-panel';
    panel.className = 'info-card workspace-panel hidden';
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-labelledby', 'workspace-panel-title');
    panel.innerHTML = '<div class="workspace-panel-header"><h3 id="workspace-panel-title"></h3><button type="button" id="workspace-panel-close" aria-label="Close panel">×</button></div><div id="workspace-panel-content"></div>';
    info.appendChild(panel);
    const views = {};
    ['projects', 'memory', 'brain', 'evolution'].forEach(name => {
      views[name] = byId('view-' + name);
      byId('workspace-panel-content').appendChild(views[name]);
      byId('tab-' + name).setAttribute('aria-controls', panel.id);
      byId('tab-' + name).setAttribute('aria-expanded', 'false');
    });
    const footer = document.querySelector('.sidebar-footer');
    info.querySelector('.info-card').appendChild(footer);
    let selectedPanel = null;
    function closePanel(restoreFocus = false) {
      const previous = selectedPanel;
      selectedPanel = null;
      panel.classList.add('hidden');
      info.classList.remove('workspace-open');
      Object.values(views).forEach(view => view.classList.add('hidden'));
      document.querySelectorAll('.sidebar-nav-tabs .nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.id === 'tab-chats');
        tab.setAttribute('aria-selected', String(tab.id === 'tab-chats'));
        if (tab.hasAttribute('aria-expanded')) tab.setAttribute('aria-expanded', 'false');
      });
      if (restoreFocus && previous) byId('tab-' + previous).focus();
    }
    window.AxenPanels = {open(name) {
      if (!views[name] || selectedPanel === name) { closePanel(); return; }
      closePanel(); selectedPanel = name;
      byId('view-chats').classList.remove('hidden');
      views[name].classList.remove('hidden');
      panel.classList.remove('hidden'); info.classList.add('workspace-open');
      document.querySelector('.app-container').classList.remove('hide-details');
      byId('btn-details').setAttribute('aria-expanded', 'true');
      byId('workspace-panel-title').textContent = name.charAt(0).toUpperCase() + name.slice(1);
      byId('tab-chats').classList.remove('active');
      byId('tab-chats').setAttribute('aria-selected', 'false');
      byId('tab-' + name).classList.add('active');
      byId('tab-' + name).setAttribute('aria-selected', 'true');
      byId('tab-' + name).setAttribute('aria-expanded', 'true');
      if (innerWidth < 760) {
        byId('sidebar').classList.add('collapsed');
        byId('btn-floating-toggle').setAttribute('aria-expanded', 'false');
      }
      byId('workspace-panel-close').focus();
    }};
    byId('workspace-panel-close').addEventListener('click', () => closePanel(true));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && selectedPanel) { event.preventDefault(); closePanel(true); }
    });
    const composer = byId('composer-floating-container');
    const welcome = byId('welcome-screen');
    const main = byId('main-content');
    const messages = byId('messages-wrapper');
    const input = byId('composer-input');
    function syncView() {
      const empty = !welcome.classList.contains('hidden');
      main.classList.toggle('empty-chat', empty);
      main.classList.toggle('conversation-view', !empty);
      const slot = byId(empty ? 'welcome-composer-slot' : 'conversation-composer-slot');
      byId('conversation-composer-slot').classList.toggle('hidden', empty);
      if (composer.parentNode !== slot) slot.insertBefore(composer, slot.firstChild);
      const attachments = byId('composer-attachments-tray');
      if (attachments.parentNode !== slot) slot.insertBefore(attachments, composer);
      const title = byId('chat-panel-title');
      const active = document.querySelector('.conversation-item.active .conv-title');
      title.textContent = empty ? 'New chat' : active ? active.textContent : 'Conversation';
      if (window.AxenMath) window.AxenMath.render(messages);
    }
    syncView();
    new MutationObserver(syncView).observe(welcome, {attributes: true, attributeFilter: ['class']});
    new MutationObserver(syncView).observe(messages, {childList: true});
    const recent = byId('conversation-list');
    new MutationObserver(syncView).observe(recent, {childList: true, subtree: true});

    document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
      input.value = button.dataset.prompt; input.dispatchEvent(new Event('input', {bubbles: true}));
      input.focus(); closeTools();
    }));
    document.querySelectorAll('[data-nav]').forEach(button => button.addEventListener('click', () => byId(button.dataset.nav)?.click()));
    document.querySelectorAll('[data-link]').forEach(button => button.addEventListener('click', () => { location.href = button.dataset.link; }));

    const tools = byId('btn-tools'), menu = byId('composer-tools-menu');
    function closeTools() { menu.classList.add('hidden'); tools.setAttribute('aria-expanded', 'false'); }
    tools.addEventListener('click', () => {
      const open = menu.classList.toggle('hidden') === false;
      tools.setAttribute('aria-expanded', String(open));
      if (open) menu.querySelector('button').focus();
    });
    byId('tools-upload').addEventListener('click', () => { byId('file-input-chat').click(); closeTools(); });
    document.addEventListener('click', event => { if (!tools.contains(event.target) && !menu.contains(event.target)) closeTools(); });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        if (!menu.classList.contains('hidden')) { closeTools(); tools.focus(); }
        byId('composer-plus-menu').classList.add('hidden');
        if (innerWidth < 760) {
          const focusInSidebar = byId('sidebar').contains(document.activeElement);
          byId('sidebar').classList.add('collapsed');
          byId('btn-floating-toggle').setAttribute('aria-expanded', 'false');
          if (focusInSidebar) byId('btn-floating-toggle').focus();
        }
      }
      if (!menu.classList.contains('hidden') && ['ArrowDown','ArrowUp'].includes(event.key)) {
        event.preventDefault(); const items = Array.from(menu.querySelectorAll('button'));
        const index = items.indexOf(document.activeElement), step = event.key === 'ArrowDown' ? 1 : -1;
        items[(index + step + items.length) % items.length].focus();
      }
    });
    const mobile = matchMedia('(max-width: 759px)');
    function responsive() {
      byId('sidebar').classList.toggle('collapsed', mobile.matches);
      byId('btn-floating-toggle').setAttribute('aria-expanded', String(!mobile.matches));
    }
    responsive(); mobile.addEventListener('change', responsive);

    const status = byId('status-text');
    const runtime = byId('info-runtime-status');
    new MutationObserver(() => { if (runtime) runtime.textContent = status.textContent; }).observe(status, {childList: true, subtree: true});
    byId('btn-details')?.addEventListener('click', () => {
      const hidden = document.querySelector('.app-container').classList.toggle('hide-details');
      byId('btn-details').setAttribute('aria-expanded', String(!hidden));
    });
    refreshDetails();
  }
  async function refreshDetails() {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) return;
      const status = await response.json();
      const tools = (status.tools || []).map(tool => typeof tool === 'string' ? tool : tool.name || '');
      const patterns = {math: /math/, calculator: /calculator/, code: /python|code/, memory: /memory/, image: /image/};
      document.querySelectorAll('[data-capability]').forEach(node => {
        const available = tools.some(tool => patterns[node.dataset.capability].test(tool));
        node.textContent = available ? 'Available' : 'Not reported';
        node.classList.toggle('available', available);
      });
      const model = byId('info-model-name');
      if (model) model.textContent = typeof status.model === 'string' ? status.model : 'Local runtime';
      const responseSessions = await fetch('/api/sessions');
      if (!responseSessions.ok) return;
      const sessions = (await responseSessions.json()).sessions || [];
      byId('info-chat-count').textContent = String(sessions.length);
      if (sessions.every(session => Number.isInteger(session.message_count)))
        byId('info-message-count').textContent = String(sessions.reduce((sum, session) => sum + session.message_count, 0));
    } catch (_) { /* Unavailable data remains explicitly unknown. */ }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
