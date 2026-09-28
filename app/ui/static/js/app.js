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
    if (toolName === 'file') return 'File Tool';
    return toolName;
  }

  function scrollToBottom() {
    if (messagesScrollArea) {
      messagesScrollArea.scrollTop = messagesScrollArea.scrollHeight;
    }
  }

  // =========================================================================
  // Chat Dispatch
  // =========================================================================

  async function sendMessage() {
    const text = composerInput.value.trim();
    if (!text || isGenerating) return;

    isGenerating = true;
    btnSend.disabled = true;

    // Immediately display user message
    welcomeScreen.classList.add('hidden');
    appendMessageToDOM('user', text);
    composerInput.value = '';
    autoResizeTextarea();

    // Create temporary thinking indicator message
    const tempRow = document.createElement('div');
    tempRow.className = 'message-row assistant';
    tempRow.id = 'temp-assistant-response';

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

    const statusBadge = document.createElement('div');
    statusBadge.className = 'tool-status-badge';
    statusBadge.id = 'live-tool-status';
    statusBadge.innerHTML = `
      <div class="spinner"></div>
      <span id="live-status-text">Thinking...</span>
    `;

    const body = document.createElement('div');
    body.className = 'assistant-body';

    wrapper.appendChild(statusBadge);
    wrapper.appendChild(body);
    tempRow.appendChild(avatar);
    tempRow.appendChild(wrapper);
    messagesWrapper.appendChild(tempRow);
    scrollToBottom();

    // Send to backend
    try {
      // Use standard POST endpoint with structured result
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: activeSessionId,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();

      // Remove temp element and append permanent message
      if (tempRow && tempRow.parentNode) {
        tempRow.parentNode.removeChild(tempRow);
      }

      appendMessageToDOM('assistant', data.response, {
        tool_used: data.tool_used,
        selected_tool: data.tool_name,
        verification: data.verification,
      });

      // Refresh session list to update titles/ordering
      await loadSessions();

    } catch (err) {
      console.error('Chat dispatch error:', err);
      if (tempRow) {
        const liveStatus = tempRow.querySelector('#live-tool-status');
        if (liveStatus) liveStatus.style.display = 'none';
        const bodyElem = tempRow.querySelector('.assistant-body');
        if (bodyElem) {
          bodyElem.innerHTML = `<span style="color: #DC2626;">Error: Could not reach AXEN backend. Details: ${err.message}</span>`;
        }
      }
    } finally {
      isGenerating = false;
      btnSend.disabled = false;
      composerInput.focus();
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
    // Send message
    btnSend.onclick = sendMessage;

    composerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
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
