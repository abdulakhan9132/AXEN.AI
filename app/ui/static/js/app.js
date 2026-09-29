/**
 * app/ui/static/js/app.js
 * =======================
 * AXEN Frontend Controller
 * 
 * Features:
 * - Real-time conversational interface with AssistantController
 * - Zero cloud APIs, full offline execution
 * - Dynamic tool execution status ("Calculating...", "Running Python...", etc.)
 * - Markdown rendering with developer code blocks & copy buttons
 * - Offline Web Speech API voice dictation
 * - Conversation session history, search, and switching
 * - Sidebar collapse toggle with state persistence
 */

(function () {
  'use strict';

  // DOM Elements
  const sidebar = document.getElementById('sidebar');
  const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const btnFloatingToggle = document.getElementById('btn-floating-toggle');
  const sidebarBackdrop = document.getElementById('sidebar-backdrop');
  const btnNewChat = document.getElementById('btn-new-chat');
  const conversationList = document.getElementById('conversation-list');
  const btnSearch = document.getElementById('btn-search');
  const searchBox = document.getElementById('sidebar-search-box');
  const searchInput = document.getElementById('sidebar-search-input');
  const welcomeScreen = document.getElementById('welcome-screen');
  const messagesScrollArea = document.getElementById('messages-scroll-area');
  const messagesWrapper = document.getElementById('messages-wrapper');
  const composerInput = document.getElementById('composer-input');
  const btnSend = document.getElementById('btn-send');
  const btnVoice = document.getElementById('btn-voice');
  const btnPlus = document.getElementById('btn-plus');
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');

  // Application State
  let activeSessionId = null;
  let conversations = [];
  let isGenerating = false;
  let speechRecognition = null;
  let isRecording = false;
  let activeRequestId = null;
  let activeAbortController = null;

  // =========================================================================
  // Initialization
  // =========================================================================

  async function init() {
    setupEventListeners();
    setupSpeechRecognition();
    restoreSidebarState();
    await fetchStatus();
    await loadSessions();
    autoResizeTextarea();
  }

  // =========================================================================
  // API Calls
  // =========================================================================

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      updateStatusUI(data);
    } catch (err) {
      console.warn('Status fetch error:', err);
    }
  }

  function updateStatusUI(data) {
    if (!statusText) return;
    if (data.cuda_available) {
      statusText.textContent = `RTX 3050 • Local Model`;
    } else if (data.model_loaded) {
      statusText.textContent = `Local Model • Ready`;
    } else if (data.model_loading) {
      statusText.textContent = `Loading Local Model...`;
    } else {
      statusText.textContent = `AXEN Local • Ready`;
    }
  }

  async function loadSessions(searchQuery = '') {
    try {
      const url = searchQuery ? `/api/sessions?q=${encodeURIComponent(searchQuery)}` : '/api/sessions';
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      conversations = data.sessions || [];
      activeSessionId = data.active_session_id;
      renderConversationList();

      if (activeSessionId) {
        await loadSessionMessages(activeSessionId);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }

  async function loadSessionMessages(sessionId) {
    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      if (!res.ok) return;
      const session = await res.json();
      activeSessionId = session.id;
      renderMessages(session.messages || []);
      renderConversationList();
    } catch (err) {
      console.error('Failed to load session messages:', err);
    }
  }

  async function createNewChat() {
    if (isGenerating) return;
    try {
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Chat' }),
      });
      if (!res.ok) return;
      const data = await res.json();
      activeSessionId = data.session_id;
      renderMessages([]);
      await loadSessions();
      composerInput.value = '';
      autoResizeTextarea();
      composerInput.focus();
    } catch (err) {
      console.error('Failed to create new chat:', err);
    }
  }

  async function deleteSession(e, sessionId) {
    e.stopPropagation();
    if (!confirm('Delete this conversation?')) return;
    try {
      const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      if (!res.ok) return;
      const data = await res.json();
      activeSessionId = data.active_session_id;
      await loadSessions();
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  }

  // =========================================================================
  // Rendering UI
  // =========================================================================

  function renderConversationList() {
    if (!conversationList) return;
    conversationList.innerHTML = '';

    if (conversations.length === 0) {
      conversationList.innerHTML = `
        <div class="empty-chat-state">
          <span>No conversations yet</span>
        </div>
      `;
      return;
    }

    conversations.forEach(conv => {
      const item = document.createElement('div');
      item.className = `conversation-item ${conv.id === activeSessionId ? 'active' : ''}`;
      item.onclick = () => {
        if (!isGenerating && conv.id !== activeSessionId) {
          loadSessionMessages(conv.id);
        }
      };

      const titleSpan = document.createElement('span');
      titleSpan.className = 'conv-title';
      titleSpan.textContent = conv.title || 'Conversation';
      titleSpan.title = conv.title || 'Conversation';

      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'conv-actions';

      const delBtn = document.createElement('button');
      delBtn.className = 'btn-conv-delete';
      delBtn.title = 'Delete Chat';
      delBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;
      delBtn.onclick = (e) => deleteSession(e, conv.id);

      actionsDiv.appendChild(delBtn);
      item.appendChild(titleSpan);
      item.appendChild(actionsDiv);
      conversationList.appendChild(item);
    });
  }

  function renderMessages(messages) {
    if (!messagesWrapper) return;
    messagesWrapper.innerHTML = '';

    if (!messages || messages.length === 0) {
      welcomeScreen.classList.remove('hidden');
      return;
    }

    welcomeScreen.classList.add('hidden');

    messages.forEach(msg => {
      appendMessageToDOM(msg.role, msg.content, msg.metadata);
    });

    scrollToBottom();
  }

  function appendMessageToDOM(role, content, metadata = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    if (role === 'user') {
      const bubble = document.createElement('div');
      bubble.className = 'user-bubble';
      bubble.textContent = content;
      row.appendChild(bubble);
    } else {
      // Assistant avatar
      const avatar = document.createElement('div');
      avatar.className = 'assistant-avatar';
      avatar.innerHTML = `
        <div class="axen-logo-small" style="transform: scale(0.65);">
          <div class="sq-orange"></div>
          <div class="sq-black-lg"></div>
          <div class="sq-black-sm"></div>
        </div>
      `;

      const wrapper = document.createElement('div');
      wrapper.className = 'assistant-content-wrapper';

      // Tool status badge if tool was used
      if (metadata && metadata.tool_used && metadata.selected_tool) {
        const badge = document.createElement('div');
        badge.className = 'tool-status-badge';
        const toolLabel = formatToolLabel(metadata.selected_tool);
        badge.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span>${toolLabel}</span>
        `;
        wrapper.appendChild(badge);
      }

      const body = document.createElement('div');
      body.className = 'assistant-body';
      body.innerHTML = window.AxenMarkdown ? window.AxenMarkdown.parse(content) : content;

      wrapper.appendChild(body);
      row.appendChild(avatar);
      row.appendChild(wrapper);
    }

    messagesWrapper.appendChild(row);
    scrollToBottom();
  }

  function formatToolLabel(toolName) {
    if (toolName === 'calculator') return 'Calculator Verified';
    if (toolName === 'python') return 'Python Executed';
    if (toolName === 'math_solver') return 'Math Solver Verified';
    if (toolName === 'computer') return 'Computer Systems Verified';
    if (toolName === 'file') return 'File Tool';
    if (toolName === 'memory') return 'Memory Recalled';
    return toolName;
  }

  function scrollToBottom() {
    if (messagesScrollArea) {
      messagesScrollArea.scrollTop = messagesScrollArea.scrollHeight;
    }
  }

  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // =========================================================================
  // Thinking / Processing State Management (Matches Reference Screenshot)
  // =========================================================================

  const SEND_ICON_SVG = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
    </svg>
  `;

  const RADIAL_SPINNER_SVG = `
    <svg class="radial-spinner" width="22" height="22" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="3.5" r="1.5" fill="#1F2937" opacity="1.0"/>
      <circle cx="18" cy="6" r="1.5" fill="#1F2937" opacity="0.875"/>
      <circle cx="20.5" cy="12" r="1.5" fill="#1F2937" opacity="0.75"/>
      <circle cx="18" cy="18" r="1.5" fill="#1F2937" opacity="0.625"/>
      <circle cx="12" cy="20.5" r="1.5" fill="#1F2937" opacity="0.5"/>
      <circle cx="6" cy="18" r="1.5" fill="#1F2937" opacity="0.375"/>
      <circle cx="3.5" cy="12" r="1.5" fill="#1F2937" opacity="0.25"/>
      <circle cx="6" cy="6" r="1.5" fill="#1F2937" opacity="0.125"/>
    </svg>
  `;

  function showThinkingUI(statusText = 'Thinking') {
    removeThinkingUI();
    welcomeScreen.classList.add('hidden');

    const container = document.createElement('div');
    container.className = 'axen-thinking-container';
    container.id = 'axen-thinking-live';

    const clean = (statusText || 'Thinking').replace(/\.+$/, '').trim();

    container.innerHTML = `
      <div class="axen-thinking-logo" aria-hidden="true">
        <div class="sq-orange"></div>
        <div class="sq-black-lg"></div>
        <div class="sq-black-sm"></div>
      </div>
      <div class="axen-thinking-status">
        <span class="axen-thinking-label" id="axen-thinking-label">${escapeHtml(clean)}</span>
        <span class="axen-thinking-dots" aria-hidden="true">
          <span>.</span><span>.</span><span>.</span><span>.</span><span>.</span><span>.</span>
        </span>
      </div>
    `;

    messagesWrapper.appendChild(container);
    scrollToBottom();
  }

  function updateThinkingStatus(statusText) {
    const label = document.getElementById('axen-thinking-label');
    const clean = (statusText || 'Thinking').replace(/\.+$/, '').trim();
    if (label) {
      label.textContent = clean;
    } else {
      showThinkingUI(clean);
    }
  }

  function removeThinkingUI() {
    const el = document.getElementById('axen-thinking-live');
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
  }

  function setComposerLoadingState(loading) {
    if (loading) {
      btnSend.classList.add('loading');
      btnSend.innerHTML = RADIAL_SPINNER_SVG;
      btnSend.title = 'Stop generating (Click to cancel)';
      btnSend.setAttribute('aria-label', 'Stop generating');
      btnSend.disabled = false;
      composerInput.disabled = true;
      btnVoice.disabled = true;
      btnVoice.style.opacity = '0.35';
    } else {
      btnSend.classList.remove('loading');
      btnSend.innerHTML = SEND_ICON_SVG;
      btnSend.title = 'Send message';
      btnSend.setAttribute('aria-label', 'Send message');
      btnSend.disabled = false;
      composerInput.disabled = false;
      btnVoice.disabled = false;
      btnVoice.style.opacity = '1';
      activeRequestId = null;
      activeAbortController = null;
      composerInput.focus();
    }
  }

  async function cancelActiveRequest() {
    if (!isGenerating) return;

    const reqId = activeRequestId;
    if (activeAbortController) {
      try {
        activeAbortController.abort();
      } catch (e) {
        console.warn('Abort error:', e);
      }
    }

    if (reqId) {
      try {
        fetch('/api/chat/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ request_id: reqId }),
        }).catch((e) => console.warn('Cancel ping failed:', e));
      } catch (e) {
        console.warn('Cancel error:', e);
      }
    }

    removeThinkingUI();
    isGenerating = false;
    setComposerLoadingState(false);
    showToast('Generation cancelled.');
  }

  function appendErrorMessage(message) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'assistant-avatar';
    avatar.innerHTML = `
      <div class="axen-logo-small" style="transform: scale(0.65);">
        <div class="sq-orange"></div>
        <div class="sq-black-lg"></div>
        <div class="sq-black-sm"></div>
      </div>
    `;

    const wrapper = document.createElement('div');
    wrapper.className = 'assistant-content-wrapper';

    const body = document.createElement('div');
    body.className = 'assistant-body';
    body.innerHTML = `<span style="color: #DC2626; font-weight: 500;">${escapeHtml(message)}</span>`;

    wrapper.appendChild(body);
    row.appendChild(avatar);
    row.appendChild(wrapper);
    messagesWrapper.appendChild(row);
    scrollToBottom();
  }

  // =========================================================================
  // Chat Dispatch via Real-Time SSE Stream with Instant Fallback
  // =========================================================================

  async function sendMessage() {
    const text = composerInput.value.trim();
    if (!text || isGenerating) return;

    // Immediately lock submission to prevent duplicate clicks
    isGenerating = true;
    setComposerLoadingState(true);

    // Immediately display user message
    welcomeScreen.classList.add('hidden');
    appendMessageToDOM('user', text);
    composerInput.value = '';
    autoResizeTextarea();

    // Setup request tracking & cancellation controller
    const requestId = 'req_' + Math.random().toString(36).substring(2, 10) + Date.now();
    activeRequestId = requestId;
    activeAbortController = new AbortController();

    let finalResultReceived = false;

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: activeSessionId,
          request_id: requestId,
        }),
        signal: activeAbortController.signal,
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;
          let eventType = 'message';
          let dataStr = '';

          const lines = part.split('\n');
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              dataStr = line.slice(6);
            }
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventType === 'state') {
              if (parsed.state === 'ERROR') {
                finalResultReceived = true;
                removeThinkingUI();
                appendErrorMessage(parsed.text || 'Something went wrong while processing your request.');
              } else if (parsed.state === 'COMPLETED' && parsed.meta && parsed.meta.cancelled) {
                finalResultReceived = true;
                removeThinkingUI();
                showToast('Generation cancelled.');
              } else {
                updateThinkingStatus(parsed.text || parsed.state);
              }
            } else if (eventType === 'result') {
              finalResultReceived = true;
              removeThinkingUI();
              appendMessageToDOM('assistant', parsed.response, {
                tool_used: parsed.tool_used,
                selected_tool: parsed.tool_name,
                verification: parsed.verification,
              });
            } else if (eventType === 'error') {
              finalResultReceived = true;
              removeThinkingUI();
              appendErrorMessage(parsed.error || parsed.detail || 'Something went wrong while processing your request.');
            }
          } catch (e) {
            console.warn('Error parsing SSE event:', e, dataStr);
          }
        }
      }

      if (!finalResultReceived && !activeAbortController.signal.aborted) {
        removeThinkingUI();
        appendErrorMessage('Something went wrong while processing your request.');
      }

      await loadSessions();

    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request aborted by user');
        removeThinkingUI();
      } else {
        console.error('SSE chat error, falling back to standard /api/chat:', err);
        try {
          updateThinkingStatus('Thinking');
          const fallbackRes = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: text,
              session_id: activeSessionId,
              request_id: requestId,
            }),
          });
          if (!fallbackRes.ok) throw new Error(`Fallback failed: ${fallbackRes.status}`);
          const data = await fallbackRes.json();
          removeThinkingUI();
          appendMessageToDOM('assistant', data.response, {
            tool_used: data.tool_used,
            selected_tool: data.tool_name,
            verification: data.verification,
          });
          await loadSessions();
        } catch (fallbackErr) {
          removeThinkingUI();
          appendErrorMessage('Something went wrong while processing your request.');
        }
      }
    } finally {
      isGenerating = false;
      setComposerLoadingState(false);
    }
  }

  // =========================================================================
  // Voice Dictation (Native Offline Web Speech API)
  // =========================================================================

  function setupSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      btnVoice.title = 'Voice input not supported by browser';
      btnVoice.onclick = () => {
        showToast('Voice dictation is supported in modern Chrome/Edge browsers, or type directly.');
      };
      return;
    }

    speechRecognition = new SpeechRec();
    speechRecognition.continuous = false;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'en-US';

    speechRecognition.onstart = () => {
      isRecording = true;
      btnVoice.classList.add('recording');
      btnVoice.title = 'Listening... Click to stop';
    };

    speechRecognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      composerInput.value = transcript;
      autoResizeTextarea();
    };

    speechRecognition.onerror = (event) => {
      console.warn('Speech recognition error:', event.error);
      stopRecording();
    };

    speechRecognition.onend = () => {
      stopRecording();
    };

    btnVoice.onclick = () => {
      if (isRecording) {
        speechRecognition.stop();
      } else {
        try {
          speechRecognition.start();
        } catch (e) {
          console.warn('Recognition start error:', e);
        }
      }
    };
  }

  function stopRecording() {
    isRecording = false;
    btnVoice.classList.remove('recording');
    btnVoice.title = 'Voice input';
  }

  // =========================================================================
  // Event Listeners & Helpers
  // =========================================================================

  function setupEventListeners() {
    // Send message or Cancel if currently generating
    btnSend.onclick = () => {
      if (isGenerating) {
        cancelActiveRequest();
      } else {
        sendMessage();
      }
    };

    composerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) {
          sendMessage();
        }
      }
    });

    composerInput.addEventListener('input', autoResizeTextarea);

    // New Chat
    btnNewChat.onclick = createNewChat;

    // Sidebar Toggle
    btnToggleSidebar.onclick = toggleSidebar;
    if (btnFloatingToggle) {
      btnFloatingToggle.onclick = toggleSidebar;
    }
    if (sidebarBackdrop) {
      sidebarBackdrop.onclick = toggleSidebar;
    }

    // Search Toggle
    btnSearch.onclick = () => {
      searchBox.classList.toggle('visible');
      if (searchBox.classList.contains('visible')) {
        searchInput.focus();
      } else {
        searchInput.value = '';
        loadSessions();
      }
    };

    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      loadSessions(q);
    });

    // Quick Action Plus Button
    btnPlus.onclick = () => {
      showQuickActionsModal();
    };
  }

  function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    const isCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem('axen_sidebar_collapsed', isCollapsed ? '1' : '0');
  }

  function restoreSidebarState() {
    const saved = localStorage.getItem('axen_sidebar_collapsed');
    if (saved === '1' && window.innerWidth > 768) {
      sidebar.classList.add('collapsed');
    }
  }

  function autoResizeTextarea() {
    composerInput.style.height = 'auto';
    const newHeight = Math.min(composerInput.scrollHeight, 140);
    composerInput.style.height = (newHeight > 24 ? newHeight : 24) + 'px';
  }

  function showToast(message) {
    const container = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 3500);
  }

  function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
  }

  function showQuickActionsModal() {
    const prompts = [
      'What is (45 * 88) / 12 + 100?',
      'Run this Python code: print([x**2 for x in range(10)])',
      'What is the greatest common divisor of 84 and 36?',
      'Remember that my preferred framework is Vanilla JS and Python.',
      'What do you know about me?',
    ];
    const picked = prompts[Math.floor(Math.random() * prompts.length)];
    composerInput.value = picked;
    autoResizeTextarea();
    composerInput.focus();
  }

  // Global helper for code block copying
  window.AxenUI = {
    copyCode: function (button) {
      const pre = button.closest('.code-block-container').querySelector('.code-block-content');
      if (!pre) return;
      const text = pre.innerText || pre.textContent;
      navigator.clipboard.writeText(text).then(() => {
        const span = button.querySelector('span');
        const orig = span.textContent;
        span.textContent = 'Copied!';
        button.style.color = '#10B981';
        setTimeout(() => {
          span.textContent = orig;
          button.style.color = '';
        }, 2000);
      });
    },
  };

  // Launch on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
