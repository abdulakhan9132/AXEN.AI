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

  // File Upload & Attachment Elements
  const composerPlusMenu = document.getElementById('composer-plus-menu');
  const menuUploadFiles = document.getElementById('menu-upload-files');
  const menuAddKnowledge = document.getElementById('menu-add-knowledge');
  const menuCancel = document.getElementById('menu-cancel');
  const fileInputChat = document.getElementById('file-input-chat');
  const fileInputKnowledge = document.getElementById('file-input-knowledge');
  const composerAttachmentsTray = document.getElementById('composer-attachments-tray');
  const dragDropOverlay = document.getElementById('drag-drop-overlay');
  const mainContent = document.getElementById('main-content');

  // Sidebar Navigation & Views
  const tabChats = document.getElementById('tab-chats');
  const tabProjects = document.getElementById('tab-projects');
  const tabFiles = document.getElementById('tab-files');
  const tabKnowledge = document.getElementById('tab-knowledge');
  const tabBrain = document.getElementById('tab-brain');
  const viewChats = document.getElementById('view-chats');
  const viewProjects = document.getElementById('view-projects');
  const viewFiles = document.getElementById('view-files');
  const viewKnowledge = document.getElementById('view-knowledge');
  const viewBrain = document.getElementById('view-brain');
  const filesList = document.getElementById('files-list');
  const filesSearchInput = document.getElementById('files-search-input');
  const knowledgeDocsList = document.getElementById('knowledge-docs-list');
  const knowledgeSearchInput = document.getElementById('knowledge-search-input');
  const btnUploadDirect = document.getElementById('btn-upload-direct');
  const btnKnowledgeUploadDirect = document.getElementById('btn-knowledge-upload-direct');

  // Project Workspace DOM Elements
  const btnNewProject = document.getElementById('btn-new-project');
  const projectsList = document.getElementById('projects-list');
  const projectsSearchInput = document.getElementById('projects-search-input');
  const btnProjectsSearch = document.getElementById('btn-projects-search');
  const projectWorkspaceHeader = document.getElementById('project-workspace-header');
  const projectHeaderIcon = document.getElementById('project-header-icon');
  const projectHeaderTitle = document.getElementById('project-header-title');
  const projectHeaderDesc = document.getElementById('project-header-desc');
  const btnProjectExit = document.getElementById('btn-project-exit');

  // Subpanel Views & Tab buttons
  const tabProjChats = document.getElementById('tab-proj-chats');
  const tabProjFiles = document.getElementById('tab-proj-files');
  const tabProjKnowledge = document.getElementById('tab-proj-knowledge');
  const tabProjMemory = document.getElementById('tab-proj-memory');
  const tabProjSettings = document.getElementById('tab-proj-settings');
  const subpanelProjectFiles = document.getElementById('subpanel-project-files');
  const subpanelProjectKnowledge = document.getElementById('subpanel-project-knowledge');
  const subpanelProjectMemory = document.getElementById('subpanel-project-memory');
  const subpanelProjectSettings = document.getElementById('subpanel-project-settings');

  // Subpanel Action Controls
  const btnProjUploadFile = document.getElementById('btn-proj-upload-file');
  const fileInputProject = document.getElementById('file-input-project');
  const projFilesContainer = document.getElementById('proj-files-container');
  const btnProjAddKnowledge = document.getElementById('btn-proj-add-knowledge');
  const projKnowledgeContainer = document.getElementById('proj-knowledge-container');
  const projMemKeyInput = document.getElementById('proj-mem-key-input');
  const projMemValInput = document.getElementById('proj-mem-val-input');
  const btnProjSaveMem = document.getElementById('btn-proj-save-mem');
  const projMemoryContainer = document.getElementById('proj-memory-container');
  const projSettingsInstructions = document.getElementById('proj-settings-instructions');
  const toggleProjMemory = document.getElementById('toggle-proj-memory');
  const toggleProjKnowledge = document.getElementById('toggle-proj-knowledge');
  const toggleProjFiles = document.getElementById('toggle-proj-files');
  const toggleProjWeb = document.getElementById('toggle-proj-web');
  const btnSaveProjectSettings = document.getElementById('btn-save-project-settings');
  const btnArchiveProjectAction = document.getElementById('btn-archive-project-action');
  const btnDeleteProjectTrigger = document.getElementById('btn-delete-project-trigger');

  // Modals
  const modalNewProject = document.getElementById('modal-new-project');
  const btnModalProjectClose = document.getElementById('btn-modal-project-close');
  const btnModalProjectCancel = document.getElementById('btn-modal-project-cancel');
  const btnModalProjectSubmit = document.getElementById('btn-modal-project-submit');
  const newProjectName = document.getElementById('new-project-name');
  const newProjectDesc = document.getElementById('new-project-desc');
  const newProjectIcon = document.getElementById('new-project-icon');
  const newProjectInstructions = document.getElementById('new-project-instructions');

  const modalDeleteProject = document.getElementById('modal-delete-project');
  const btnModalDeleteClose = document.getElementById('btn-modal-delete-close');
  const btnModalDeleteCancel = document.getElementById('btn-modal-delete-cancel');
  const btnModalDeleteConfirm = document.getElementById('btn-modal-delete-confirm');
  const deleteProjectNameDisplay = document.getElementById('delete-project-name-display');

  // Centralized upload limits
  const ALLOWED_EXTENSIONS = [
    '.pdf', '.txt', '.md', '.markdown', '.docx',
    '.csv', '.json',
    '.py', '.js', '.ts', '.html', '.css', '.yaml', '.yml', '.xml',
    '.sql', '.java', '.c', '.cpp', '.h', '.hpp', '.sh', '.ps1',
    '.png', '.jpg', '.jpeg', '.webp'
  ];
  const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
  const MAX_FILES_PER_MESSAGE = 10;

  // Application State
  let activeSessionId = null;
  let activeProjectId = localStorage.getItem('axen_active_project_id') || null;
  let activeProjectData = null;
  let activeSubpanel = 'chat';
  let conversations = [];
  let isGenerating = false;
  let speechRecognition = null;
  let isRecording = false;
  let activeRequestId = null;
  let activeAbortController = null;
  let activeAttachments = []; // Array of { file_id, original_filename, size, file_type, category, status, previewUrl }

  // =========================================================================
  // Initialization
  // =========================================================================

  async function init() {
    setupEventListeners();
    setupProjectHandlers();
    setupFileHandlers();
    setupSpeechRecognition();
    restoreSidebarState();
    await fetchStatus();
    setInterval(() => {
      pollBrainStatus();
    }, 6000);
    if (activeProjectId) {
      await selectProject(activeProjectId);
    } else {
      await loadSessions();
    }
    autoResizeTextarea();
  }

  // =========================================================================
  // API Calls
  // =========================================================================

  async function fetchStatus() {
    await pollBrainStatus();
  }

  async function pollBrainStatus() {
    try {
      const res = await fetch('/api/brain');
      if (!res.ok) throw new Error('Brain endpoint error');
      const data = await res.json();
      updateBrainUI(data);
    } catch (err) {
      const stateEl = document.getElementById('brain-state');
      if (stateEl) stateEl.textContent = 'Status unavailable';
    }
  }

  function updateBrainUI(data) {
    if (!data) return;
    const dot = document.getElementById('status-dot');
    const pill = document.getElementById('status-pill');
    const lossText = data.loss != null ? Number(data.loss).toFixed(3) : '--';
    if (statusText) {
      statusText.textContent = data.trained
        ? `Brain: ${Number(data.steps || 0).toLocaleString()} steps, loss ${lossText}`
        : 'Untrained: run training';
    }
    if (dot) {
      dot.className = data.sleeping ? 'status-dot pulsing' : (data.trained ? 'status-dot online' : 'status-dot offline');
      dot.style.background = data.sleeping ? '#F59E0B' : (data.trained ? '#10B981' : '#EF4444');
    }
    if (pill) pill.title = data.sleeping ? 'Axen is sleeping and learning' : (data.trained ? 'Axen brain is ready' : 'Train Axen to start chatting');

    // Update brain view panel elements
    const pEl = document.getElementById('brain-params');
    const sEl = document.getElementById('brain-steps');
    const lEl = document.getElementById('brain-loss');
    const stEl = document.getElementById('brain-state');
    const fEl = document.getElementById('brain-facts-count');
    const eEl = document.getElementById('brain-exchanges-count');
    const sleepBtn = document.getElementById('btn-sleep-learn');
    const sleepText = document.getElementById('btn-sleep-text');

    if (pEl) pEl.textContent = Number(data.params || 0).toLocaleString();
    if (sEl) sEl.textContent = (data.steps || 0).toLocaleString();
    if (lEl) lEl.textContent = data.loss != null ? Number(data.loss).toFixed(3) : '--';
    if (stEl) stEl.textContent = data.sleeping ? 'Sleeping...' : 'Awake';

    if (data.personality) {
      for (const [trait, val] of Object.entries(data.personality)) {
        const bar = document.getElementById(`trait-${trait}`);
        if (bar) bar.style.width = Math.round(val * 100) + '%';
      }
    }

    if (data.memory) {
      if (fEl) fEl.textContent = data.memory.facts || 0;
      if (eEl) eEl.textContent = data.memory.exchanges || 0;
    }

    if (sleepBtn && sleepText) {
      if (data.sleeping) {
        sleepBtn.classList.add('sleeping');
        sleepText.textContent = 'Sleeping and learning...';
        sleepBtn.disabled = true;
      } else {
        sleepBtn.classList.remove('sleeping');
        sleepText.textContent = 'Sleep and Learn (500 steps)';
        sleepBtn.disabled = false;
      }
    }
  }

  async function loadSessions(searchQuery = '') {
    try {
      let url = searchQuery ? `/api/sessions?q=${encodeURIComponent(searchQuery)}` : '/api/sessions';
      if (activeProjectId) {
        url += (url.includes('?') ? '&' : '?') + `project_id=${encodeURIComponent(activeProjectId)}`;
      }
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      conversations = data.sessions || [];
      activeSessionId = data.active_session_id;
      renderConversationList();

      if (activeSessionId) {
        await loadSessionMessages(activeSessionId);
      } else {
        renderMessages([]);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }

  async function refreshSessionsList(searchQuery = '') {
    try {
      let url = searchQuery ? `/api/sessions?q=${encodeURIComponent(searchQuery)}` : '/api/sessions';
      if (activeProjectId) {
        url += (url.includes('?') ? '&' : '?') + `project_id=${encodeURIComponent(activeProjectId)}`;
      }
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      conversations = data.sessions || [];
      renderConversationList();
    } catch (err) {
      console.error('Failed to refresh sessions:', err);
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
        body: JSON.stringify({
          title: 'New Chat',
          project_id: activeProjectId || null
        }),
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
      const headingEl = welcomeScreen.querySelector('.welcome-heading');
      const subtitleEl = welcomeScreen.querySelector('.welcome-subtitle');
      if (activeProjectId && activeProjectData) {
        if (headingEl) headingEl.textContent = activeProjectData.name;
        if (subtitleEl) subtitleEl.textContent = activeProjectData.description || 'Workspace is ready. Ask anything, manage files, or configure instructions.';
      } else {
        if (headingEl) headingEl.textContent = 'How Can I Help You Today?';
        if (subtitleEl) subtitleEl.textContent = 'Ask Anything, Get Ideas, Write, Analyze, Design, Or Solve Problems With AI.';
      }
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

      if (metadata && metadata.attachments && metadata.attachments.length) {
        const attContainer = document.createElement('div');
        attContainer.className = 'message-attachments';
        metadata.attachments.forEach(fname => {
          const badge = document.createElement('span');
          badge.className = 'message-attachment-badge';
          badge.innerHTML = `📄 ${escapeHtml(fname)}`;
          attContainer.appendChild(badge);
        });
        bubble.appendChild(attContainer);
      }

      const textSpan = document.createElement('span');
      textSpan.textContent = content;
      bubble.appendChild(textSpan);
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

      // Feedback buttons (thumbs up / down)
      const actions = document.createElement('div');
      actions.className = 'message-actions';
      actions.innerHTML = `
        <button class="btn-msg-feedback btn-thumb-up" title="Good response (thumbs up)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
          </svg>
        </button>
        <button class="btn-msg-feedback btn-thumb-down" title="Poor response (thumbs down)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
          </svg>
        </button>
      `;
      const upBtn = actions.querySelector('.btn-thumb-up');
      const downBtn = actions.querySelector('.btn-thumb-down');
      if (upBtn) {
        upBtn.onclick = () => {
          upBtn.classList.add('active-up');
          if (downBtn) downBtn.classList.remove('active-down');
          const prevUser = getPreviousUserMessageText(row);
          sendFeedback(prevUser, content, true);
        };
      }
      if (downBtn) {
        downBtn.onclick = () => {
          downBtn.classList.add('active-down');
          if (upBtn) upBtn.classList.remove('active-up');
          const prevUser = getPreviousUserMessageText(row);
          sendFeedback(prevUser, content, false);
        };
      }

      wrapper.appendChild(body);
      wrapper.appendChild(actions);
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
    if (toolName === 'file') return 'File Ingested / Analyzed';
    if (toolName === 'file_extended') return 'File Ingested / Analyzed';
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

  function getPreviousUserMessageText(assistantRow) {
    let prev = assistantRow ? assistantRow.previousElementSibling : null;
    while (prev) {
      if (prev.classList && prev.classList.contains('user')) {
        const textSpan = prev.querySelector('.user-bubble span');
        return textSpan ? textSpan.textContent.trim() : '';
      }
      prev = prev.previousElementSibling;
    }
    return '';
  }

  async function sendFeedback(userText, axenText, up) {
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user: userText || '', axen: axenText || '', up: !!up }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error('Could not save feedback');
      showToast(up ? 'Feedback saved. Axen is learning from it.' : 'Feedback saved.');
      pollBrainStatus();
    } catch (e) {
      console.warn('Feedback send error:', e);
    }
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

  const STOP_ICON_SVG = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect>
    </svg>
  `;

  function setComposerLoadingState(loading) {
    if (loading) {
      btnSend.classList.add('loading');
      btnSend.innerHTML = STOP_ICON_SVG;
      btnSend.title = 'Stop generating';
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
      activeAbortController = null;
      composerInput.focus();
    }
  }

  function cancelActiveRequest() {
    if (!isGenerating) return;
    if (activeAbortController) {
      try {
        activeAbortController.abort();
      } catch (e) {
        console.warn('Abort error:', e);
      }
    }
    isGenerating = false;
    setComposerLoadingState(false);
    showToast('Generation stopped.');
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

  async function ask(messages, { onStage, onToken, onDone, onError }, signal) {
    const thinkModeSelect = document.getElementById('select-think-mode');
    const thinkMode = thinkModeSelect ? thinkModeSelect.value : 'auto';
    const modelSelect = document.getElementById('select-active-model');
    const model = modelSelect ? modelSelect.value : 'router';
    const payload = { messages, think_mode: thinkMode, mode: thinkMode, model: model };

    let endpoint = '/api/v2/chat';
    let response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });
    
    // Fallback to older Node.js backend if Python v2 is unavailable
    if (response.status === 404) {
      endpoint = '/api/chat';
      response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      });
    }

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Chat request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalAnswer = '';
    let streamError = false;

    const handleEvent = (frame) => {
      const dataLines = [];
      for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) return;

      let payload;
      try {
        payload = JSON.parse(dataLines.join('\n'));
      } catch (error) {
        onError('Axen returned an unreadable response.');
        return;
      }

      if (payload.type === 'stage') {
        onStage(payload.data);
      } else if (payload.type === 'token') {
        onToken(payload.data);
      } else if (payload.type === 'done') {
        finalAnswer = payload.data || '';
        if (!finalAnswer) onError('Axen completed the request without a response.');
        else onDone(finalAnswer);
      } else if (payload.type === 'error') {
        streamError = true;
        onError(payload.data || 'Axen could not process this request.');
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        handleEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
      }
      if (done) break;
    }
    if (buffer.trim()) handleEvent(buffer);
    if (!finalAnswer && !streamError) onError('Axen closed the chat stream before returning an answer.');
    return finalAnswer;
  }

  async function persistSessionMessage(role, content) {
    if (!activeSessionId || !content) return;
    try {
      const response = await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, content }),
      });
      if (!response.ok) throw new Error(`Session save failed (${response.status})`);
    } catch (error) {
      console.warn('Session message save error:', error);
    }
  }

  // =========================================================================
  // Chat Dispatch via Streaming SSE with Collapsible Thinking & Real-Time Token Generation
  // =========================================================================

  async function sendMessage() {
    const text = composerInput.value.trim();
    if (!text || isGenerating) return;

    const messageText = text;
    const priorMessages = Array.from(messagesWrapper.querySelectorAll('.message-row')).map((messageRow) => {
      const role = messageRow.classList.contains('user') ? 'user' : 'assistant';
      const contentNode = messageRow.querySelector(role === 'user' ? '.user-bubble' : '.assistant-body');
      return contentNode ? { role, content: contentNode.innerText.trim() } : null;
    }).filter((message) => message && message.content).slice(-10);
    isGenerating = true;
    setComposerLoadingState(true);

    // Display user message immediately
    welcomeScreen.classList.add('hidden');
    appendMessageToDOM('user', messageText);
    composerInput.value = '';
    autoResizeTextarea();

    // Ensure an active session exists
    if (!activeSessionId) {
      try {
        const sRes = await fetch('/api/sessions/new', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: messageText.slice(0, 30) }),
        });
        if (sRes.ok) {
          const sData = await sRes.json();
          activeSessionId = sData.session_id || (sData.session && sData.session.id);
        }
      } catch (e) {
        console.warn('Session init error:', e);
      }
    }
    await persistSessionMessage('user', messageText);

    activeAbortController = new AbortController();
    activeRequestId = (window.crypto && window.crypto.randomUUID)
      ? window.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    // Build Assistant message row in DOM
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

    // Collapsible Thinking Panel with timer and stages
    const thinking = document.createElement('details');
    thinking.className = 'axen-thinking-collapsible';
    thinking.open = true;

    const summary = document.createElement('summary');
    summary.className = 'axen-thinking-summary';
    summary.innerHTML = `
      <span class="axen-thinking-icon">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 14 14"></polyline>
        </svg>
      </span>
      <span class="axen-thinking-title">Thinking</span>
      <span class="axen-thinking-timer">0.0s</span>
    `;

    const stageList = document.createElement('ul');
    stageList.className = 'axen-stage-list';

    thinking.appendChild(summary);
    thinking.appendChild(stageList);
    wrapper.appendChild(thinking);

    const titleEl = summary.querySelector('.axen-thinking-title');
    const timerEl = summary.querySelector('.axen-thinking-timer');

    // Assistant response body for live token streaming
    const body = document.createElement('div');
    body.className = 'assistant-body';
    wrapper.appendChild(body);

    // Feedback buttons (thumbs up / thumbs down)
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.style.display = 'none';
    actions.innerHTML = `
      <button class="btn-msg-feedback btn-thumb-up" title="Good response (thumbs up)">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
        </svg>
      </button>
      <button class="btn-msg-feedback btn-thumb-down" title="Poor response (thumbs down)">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
        </svg>
      </button>
    `;
    wrapper.appendChild(actions);

    row.appendChild(avatar);
    row.appendChild(wrapper);
    messagesWrapper.appendChild(row);
    scrollToBottom();

    // Live thinking timer
    const startTime = Date.now();
    const timerId = setInterval(() => {
      timerEl.textContent = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
    }, 100);

    let accumulatedAnswer = '';

    try {
      const responseText = await ask([...priorMessages, { role: 'user', content: messageText }], {
        onStage: (stageText) => {
          titleEl.textContent = stageText;
          const li = document.createElement('li');
          li.textContent = stageText;
          stageList.appendChild(li);
          scrollToBottom();
        },
        onToken: (tok) => {
          accumulatedAnswer += tok;
          body.innerHTML = window.AxenMarkdown ? window.AxenMarkdown.parse(accumulatedAnswer) : escapeHtml(accumulatedAnswer);
          scrollToBottom();
        },
        onDone: (finalText) => {
          clearInterval(timerId);
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
          timerEl.textContent = elapsed + 's';
          titleEl.textContent = `Thought for ${elapsed}s`;
          thinking.open = false; // Collapse thinking panel when generation finishes

          const resText = finalText != null && finalText !== '' ? finalText : accumulatedAnswer;
          body.innerHTML = window.AxenMarkdown ? window.AxenMarkdown.parse(resText) : escapeHtml(resText);

          // Reveal thumbs up / thumbs down
          actions.style.display = 'flex';
          const upBtn = actions.querySelector('.btn-thumb-up');
          const downBtn = actions.querySelector('.btn-thumb-down');
          if (upBtn) {
            upBtn.onclick = () => {
              upBtn.classList.add('active-up');
              if (downBtn) downBtn.classList.remove('active-down');
              sendFeedback(messageText, resText, true);
            };
          }
          if (downBtn) {
            downBtn.onclick = () => {
              downBtn.classList.add('active-down');
              if (upBtn) upBtn.classList.remove('active-up');
              sendFeedback(messageText, resText, false);
            };
          }

        },
        onError: (errText) => {
          clearInterval(timerId);
          body.innerHTML = `<span style="color: #EF4444; font-weight: 500;">${escapeHtml(errText)}</span>`;
        }
      }, activeAbortController.signal);
      await persistSessionMessage('assistant', responseText);

    } catch (err) {
      clearInterval(timerId);
      if (err.name === 'AbortError') {
        showToast('Generation stopped.');
        if (!accumulatedAnswer) {
          body.innerHTML = `<span style="color: #9CA3AF; font-style: italic;">Stopped.</span>`;
        }
      } else {
        console.error('Chat error:', err);
        body.innerHTML = `<span style="color: #EF4444; font-weight: 500;">Error: ${escapeHtml(err.message)}</span>`;
      }
    } finally {
      clearInterval(timerId);
      isGenerating = false;
      setComposerLoadingState(false);
      activeRequestId = null;
      activeAbortController = null;
      pollBrainStatus();
      refreshSessionsList();
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
  }

  // =========================================================================
  // File Upload, Attachments & Drag-and-Drop Management
  // =========================================================================

  function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function getCategoryClass(category) {
    switch (category) {
      case 'document': return 'pdf';
      case 'data': return 'data';
      case 'code': return 'code';
      case 'image': return 'image';
      default: return 'doc';
    }
  }

  function renderAttachmentsTray() {
    if (!composerAttachmentsTray) return;
    if (activeAttachments.length === 0) {
      composerAttachmentsTray.classList.add('hidden');
      composerAttachmentsTray.innerHTML = '';
      return;
    }

    composerAttachmentsTray.classList.remove('hidden');
    composerAttachmentsTray.innerHTML = '';

    activeAttachments.forEach((att, idx) => {
      const chip = document.createElement('div');
      chip.className = 'attachment-chip';

      let mediaHtml = '';
      if (att.category === 'image' && att.previewUrl) {
        mediaHtml = `<img class="attachment-thumb" src="${att.previewUrl}" alt="${escapeHtml(att.original_filename)}" />`;
      } else {
        const catClass = getCategoryClass(att.category);
        const ext = (att.original_filename.split('.').pop() || 'FILE').toUpperCase().slice(0, 4);
        mediaHtml = `<div class="attachment-icon ${catClass}">${escapeHtml(ext)}</div>`;
      }

      const statusBadge = att.status === 'uploading' ? '<span style="color:#F05A28; font-size:10px;">uploading...</span>' : '';

      chip.innerHTML = `
        ${mediaHtml}
        <div class="attachment-info">
          <span class="attachment-name" title="${escapeHtml(att.original_filename)}">${escapeHtml(att.original_filename)}</span>
          <span class="attachment-meta">
            ${formatFileSize(att.size)}
            ${statusBadge}
          </span>
        </div>
        <button class="attachment-remove" title="Remove attachment" aria-label="Remove attachment">&times;</button>
      `;

      chip.querySelector('.attachment-remove').onclick = (e) => {
        e.stopPropagation();
        removeAttachment(idx);
      };

      composerAttachmentsTray.appendChild(chip);
    });
  }

  function removeAttachment(index) {
    if (index >= 0 && index < activeAttachments.length) {
      const removed = activeAttachments.splice(index, 1)[0];
      if (removed.file_id) {
        fetch(`/api/files/${removed.file_id}`, { method: 'DELETE' }).catch(() => {});
      }
      renderAttachmentsTray();
    }
  }

  async function handleFilesSelected(fileList, targetKnowledge = false) {
    if (!fileList || fileList.length === 0) return;

    if (!targetKnowledge && (activeAttachments.length + fileList.length > MAX_FILES_PER_MESSAGE)) {
      showToast(`Maximum ${MAX_FILES_PER_MESSAGE} files per message allowed.`);
      return;
    }

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();

      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        showToast("AXEN doesn't currently support this file type.");
        continue;
      }

      if (file.size === 0) {
        showToast(`File "${file.name}" is empty (0 bytes).`);
        continue;
      }

      if (file.size > MAX_FILE_SIZE) {
        showToast(`File "${file.name}" exceeds 25 MB limit.`);
        continue;
      }

      if (!targetKnowledge && activeAttachments.some(a => a.original_filename === file.name && a.size === file.size)) {
        showToast(`"${file.name}" is already attached.`);
        continue;
      }

      await uploadFile(file, targetKnowledge);
    }
  }

  async function uploadFile(file, targetKnowledge = false) {
    let previewUrl = null;
    const isImage = file.type.startsWith('image/');
    if (isImage) {
      try {
        previewUrl = URL.createObjectURL(file);
      } catch (e) {}
    }

    const tempItem = {
      file_id: null,
      original_filename: file.name,
      size: file.size,
      status: 'uploading',
      category: isImage ? 'image' : 'document',
      previewUrl: previewUrl,
    };

    if (!targetKnowledge) {
      activeAttachments.push(tempItem);
      renderAttachmentsTray();
    } else {
      showToast(`Uploading ${file.name} to Knowledge...`);
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        const errMsg = data.detail || "AXEN doesn't currently support this file type.";
        showToast(errMsg);
        if (!targetKnowledge) {
          activeAttachments = activeAttachments.filter(a => a !== tempItem);
          renderAttachmentsTray();
        }
        return;
      }

      if (targetKnowledge) {
        showToast(`Indexing ${file.name} into Knowledge...`);
        const ingestRes = await fetch(`/api/files/${data.file_id}/ingest-knowledge`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ collection: 'default' }),
        });
        const ingestData = await ingestRes.json();
        if (ingestRes.ok) {
          showToast(`Added to Knowledge: ${data.original_filename} (${ingestData.chunk_count} chunks indexed)`);
          if (viewKnowledge && !viewKnowledge.classList.contains('hidden')) {
            loadKnowledgeView();
          }
        } else {
          showToast(`Ingestion failed: ${ingestData.detail || 'Error'}`);
        }
      } else {
        tempItem.file_id = data.file_id;
        tempItem.status = 'ready';
        tempItem.original_filename = data.original_filename;
        tempItem.size = data.size;
        tempItem.category = data.category;
        tempItem.file_type = data.file_type;
        renderAttachmentsTray();
      }

      if (viewFiles && !viewFiles.classList.contains('hidden')) {
        loadFilesView();
      }
    } catch (err) {
      console.error('File upload error:', err);
      showToast(`Upload failed for ${file.name}`);
      if (!targetKnowledge) {
        activeAttachments = activeAttachments.filter(a => a !== tempItem);
        renderAttachmentsTray();
      }
    }
  }

  function setupFileHandlers() {
    // Plus button menu toggle
    if (btnPlus && composerPlusMenu) {
      btnPlus.onclick = (e) => {
        e.stopPropagation();
        composerPlusMenu.classList.toggle('hidden');
      };

      document.addEventListener('click', (e) => {
        if (!composerPlusMenu.contains(e.target) && e.target !== btnPlus && !btnPlus.contains(e.target)) {
          composerPlusMenu.classList.add('hidden');
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !composerPlusMenu.classList.contains('hidden')) {
          composerPlusMenu.classList.add('hidden');
        }
      });
    }

    if (menuCancel && composerPlusMenu) {
      menuCancel.onclick = () => {
        composerPlusMenu.classList.add('hidden');
      };
    }

    if (menuUploadFiles && fileInputChat) {
      menuUploadFiles.onclick = () => {
        composerPlusMenu.classList.add('hidden');
        fileInputChat.value = '';
        fileInputChat.click();
      };

      fileInputChat.onchange = (e) => {
        handleFilesSelected(e.target.files, false);
      };
    }

    if (menuAddKnowledge && fileInputKnowledge) {
      menuAddKnowledge.onclick = () => {
        composerPlusMenu.classList.add('hidden');
        fileInputKnowledge.value = '';
        fileInputKnowledge.click();
      };

      fileInputKnowledge.onchange = (e) => {
        handleFilesSelected(e.target.files, true);
      };
    }

    // Model selector change handler
    const selectActiveModel = document.getElementById('select-active-model');
    if (selectActiveModel) {
      selectActiveModel.addEventListener('change', async (e) => {
        const modelId = e.target.value;
        try {
          const res = await fetch('/api/models/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId }),
          });
          if (res.ok) {
            showToast(`Active model: ${e.target.options[e.target.selectedIndex].text}`);
          }
        } catch (err) {
          console.warn('Model switch error:', err);
        }
      });
    }

    // Thinking mode selector change handler
    const selectThinkMode = document.getElementById('select-think-mode');
    if (selectThinkMode) {
      selectThinkMode.addEventListener('change', (e) => {
        showToast(`Reasoning mode: ${e.target.options[e.target.selectedIndex].text}`);
      });
    }

    // Sidebar Navigation Tabs & Views
    const tabMemory = document.getElementById('tab-memory');
    const viewMemory = document.getElementById('view-memory');
    const memoryItemsList = document.getElementById('memory-items-list');
    const memorySearchInput = document.getElementById('memory-search-input');
    const btnMemoryRefresh = document.getElementById('btn-memory-refresh');

    function switchSidebarTab(tabName) {
      [tabChats, tabProjects, tabFiles, tabKnowledge, tabMemory, tabBrain].forEach(t => t && t.classList.remove('active'));
      [viewChats, viewProjects, viewFiles, viewKnowledge, viewMemory, viewBrain].forEach(v => v && v.classList.add('hidden'));

      if (tabName === 'projects') {
        if (tabProjects) tabProjects.classList.add('active');
        if (viewProjects) viewProjects.classList.remove('hidden');
        loadProjectsList();
      } else if (tabName === 'files') {
        if (tabFiles) tabFiles.classList.add('active');
        if (viewFiles) viewFiles.classList.remove('hidden');
        loadFilesView();
      } else if (tabName === 'knowledge') {
        if (tabKnowledge) tabKnowledge.classList.add('active');
        if (viewKnowledge) viewKnowledge.classList.remove('hidden');
        loadKnowledgeView();
      } else if (tabName === 'memory') {
        if (tabMemory) tabMemory.classList.add('active');
        if (viewMemory) viewMemory.classList.remove('hidden');
        loadMemoryView();
      } else if (tabName === 'brain') {
        if (tabBrain) tabBrain.classList.add('active');
        if (viewBrain) viewBrain.classList.remove('hidden');
        pollBrainStatus();
      } else {
        if (tabChats) tabChats.classList.add('active');
        if (viewChats) viewChats.classList.remove('hidden');
      }
    }

    if (tabChats) tabChats.onclick = () => switchSidebarTab('chats');
    if (tabProjects) tabProjects.onclick = () => switchSidebarTab('projects');
    if (tabFiles) tabFiles.onclick = () => switchSidebarTab('files');
    if (tabKnowledge) tabKnowledge.onclick = () => switchSidebarTab('knowledge');
    if (tabMemory) tabMemory.onclick = () => switchSidebarTab('memory');
    if (tabBrain) tabBrain.onclick = () => switchSidebarTab('brain');
    if (statusPill) statusPill.onclick = () => switchSidebarTab('brain');

    const btnSleep = document.getElementById('btn-sleep-learn');
    if (btnSleep) {
      btnSleep.onclick = async () => {
        try {
          const sRes = await fetch('/api/sleep', { method: 'POST' });
          const sData = await sRes.json();
          if (sData.started) {
            showToast('Sleep learning started (500 steps in background)...');
          } else {
            showToast('Already learning or cannot start right now.');
          }
          pollBrainStatus();
        } catch (e) {
          showToast('Failed to start sleep learning');
        }
      };
    }

    if (btnMemoryRefresh) {
      btnMemoryRefresh.onclick = () => loadMemoryView();
    }
    if (memorySearchInput) {
      memorySearchInput.oninput = (e) => loadMemoryView(e.target.value.trim());
    }

    async function loadMemoryView(searchQuery = '') {
      if (!memoryItemsList) return;
      try {
        let url = searchQuery ? `/api/memory?q=${encodeURIComponent(searchQuery)}` : '/api/memory';
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        const mems = data.memories || [];

        memoryItemsList.innerHTML = '';
        if (mems.length === 0) {
          memoryItemsList.innerHTML = '<div class="empty-view-placeholder" style="color:var(--text-muted); padding:16px 0; font-size:13px;">No saved memories found.</div>';
          return;
        }

        mems.forEach(m => {
          const item = document.createElement('div');
          item.className = 'file-list-item';
          item.style.marginBottom = '8px';
          item.style.padding = '8px 12px';
          item.style.background = 'var(--bg-secondary)';
          item.style.borderRadius = '8px';
          item.style.display = 'flex';
          item.style.alignItems = 'center';
          item.style.justifyContent = 'space-between';

          const scopeBadge = m.scope === 'project' ? `<span style="font-size:10px; background:#F05A28; color:#fff; padding:2px 6px; border-radius:4px; margin-left:6px;">Project</span>` : `<span style="font-size:10px; background:#4B5563; color:#fff; padding:2px 6px; border-radius:4px; margin-left:6px;">Global</span>`;

          item.innerHTML = `
            <div style="flex:1; min-width:0; margin-right:8px;">
              <div style="display:flex; align-items:center;">
                <span style="font-weight:600; font-size:13px; color:var(--text-primary); text-transform:capitalize;">${escapeHtml(m.key || 'memory')}</span>
                ${scopeBadge}
              </div>
              <div style="font-size:12px; color:var(--text-muted); margin-top:3px; word-break:break-word;">${escapeHtml(m.value || '')}</div>
            </div>
            <button class="btn-item-action" title="Delete memory" style="background:none; border:none; cursor:pointer; color:var(--text-muted); font-size:16px;">&times;</button>
          `;

          const btnDel = item.querySelector('.btn-item-action');
          btnDel.onclick = async () => {
            const delUrl = `/api/memory?key=${encodeURIComponent(m.key)}&scope=${encodeURIComponent(m.scope || 'global')}${m.project_id ? '&project_id=' + encodeURIComponent(m.project_id) : ''}`;
            await fetch(delUrl, { method: 'DELETE' });
            loadMemoryView(searchQuery);
          };

          memoryItemsList.appendChild(item);
        });
      } catch (err) {
        console.error('Error loading memory view:', err);
      }
    }

    if (btnUploadDirect && fileInputChat) {
      btnUploadDirect.onclick = () => {
        fileInputChat.value = '';
        fileInputChat.click();
      };
    }

    if (btnKnowledgeUploadDirect && fileInputKnowledge) {
      btnKnowledgeUploadDirect.onclick = () => {
        fileInputKnowledge.value = '';
        fileInputKnowledge.click();
      };
    }

    if (filesSearchInput) {
      filesSearchInput.addEventListener('input', (e) => loadFilesView(e.target.value.trim()));
    }

    if (knowledgeSearchInput) {
      knowledgeSearchInput.addEventListener('input', (e) => loadKnowledgeView(e.target.value.trim()));
    }

    // Drag and drop setup on mainContent
    if (mainContent && dragDropOverlay) {
      let dragCounter = 0;

      window.addEventListener('dragenter', (e) => {
        if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
          e.preventDefault();
          dragCounter++;
          dragDropOverlay.classList.remove('hidden');
        }
      });

      window.addEventListener('dragover', (e) => {
        if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
          e.preventDefault();
        }
      });

      window.addEventListener('dragleave', (e) => {
        dragCounter--;
        if (dragCounter <= 0) {
          dragCounter = 0;
          dragDropOverlay.classList.add('hidden');
        }
      });

      window.addEventListener('drop', (e) => {
        dragCounter = 0;
        dragDropOverlay.classList.add('hidden');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          e.preventDefault();
          handleFilesSelected(e.dataTransfer.files, false);
        }
      });
    }
  }

  async function loadFilesView(query = '') {
    if (!filesList) return;
    try {
      const res = await fetch('/api/files');
      if (!res.ok) return;
      const data = await res.json();
      let files = data.files || [];
      if (query) {
        files = files.filter(f => (f.filename || '').toLowerCase().includes(query.toLowerCase()));
      }
      if (files.length === 0) {
        filesList.innerHTML = '<div class="empty-view-placeholder">No uploaded files yet.</div>';
        return;
      }
      filesList.innerHTML = '';
      files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-list-item';
        const ext = (f.extension || '').replace('.', '').toUpperCase().slice(0, 4) || 'FILE';
        const catClass = getCategoryClass(f.category);
        const dateStr = f.created_at ? new Date(f.created_at * 1000).toLocaleDateString() : '';
        item.innerHTML = `
          <div class="attachment-icon ${catClass}">${escapeHtml(ext)}</div>
          <div class="file-item-info">
            <span class="file-item-name" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span>
            <span class="file-item-sub">
              <span>${escapeHtml(f.size_formatted || formatFileSize(f.size_bytes))}</span>
              <span>•</span>
              <span>${dateStr}</span>
              <span class="file-item-badge ${f.status}">${f.status}</span>
            </span>
          </div>
          <button class="btn-item-action" title="Delete file" aria-label="Delete file">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
        `;
        item.querySelector('.btn-item-action').onclick = async (e) => {
          e.stopPropagation();
          try {
            const delRes = await fetch(`/api/files/${f.file_id}`, { method: 'DELETE' });
            if (delRes.ok) {
              showToast(`Deleted ${f.filename}`);
              loadFilesView(filesSearchInput ? filesSearchInput.value.trim() : '');
            } else {
              showToast(`Could not delete ${f.filename}`);
            }
          } catch (err) {
            showToast('Delete failed');
          }
        };
        filesList.appendChild(item);
      });
    } catch (e) {
      console.warn('Error loading files view:', e);
    }
  }

  async function loadKnowledgeView(query = '') {
    if (!knowledgeDocsList) return;
    try {
      const res = await fetch('/api/knowledge/documents');
      if (!res.ok) return;
      const data = await res.json();
      let docs = data.documents || [];
      if (query) {
        docs = docs.filter(d => (d.filename || '').toLowerCase().includes(query.toLowerCase()));
      }
      if (docs.length === 0) {
        knowledgeDocsList.innerHTML = '<div class="empty-view-placeholder">No indexed documents yet.</div>';
        return;
      }
      knowledgeDocsList.innerHTML = '';
      docs.forEach(d => {
        const item = document.createElement('div');
        item.className = 'knowledge-list-item';
        const dateStr = d.indexed_at ? new Date(d.indexed_at * 1000).toLocaleDateString() : '';
        item.innerHTML = `
          <div class="attachment-icon doc">RAG</div>
          <div class="knowledge-item-info">
            <span class="knowledge-item-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
            <span class="knowledge-item-sub">
              <span>${d.chunks_count || 1} chunks</span>
              <span>•</span>
              <span>${dateStr}</span>
            </span>
          </div>
          <button class="btn-item-action" title="Delete document" aria-label="Delete document">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
        `;
        item.querySelector('.btn-item-action').onclick = async (e) => {
          e.stopPropagation();
          try {
            const delRes = await fetch(`/api/knowledge/documents/${d.document_id}`, { method: 'DELETE' });
            if (delRes.ok) {
              showToast(`Removed ${d.filename} from Knowledge`);
              loadKnowledgeView(knowledgeSearchInput ? knowledgeSearchInput.value.trim() : '');
            } else {
              showToast(`Could not delete document`);
            }
          } catch (err) {
            showToast('Delete failed');
          }
        };
        knowledgeDocsList.appendChild(item);
      });
    } catch (e) {
      console.warn('Error loading knowledge view:', e);
    }
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

  // =========================================================================
  // Projects & Workspaces System Management
  // =========================================================================

  function setupProjectHandlers() {
    if (btnNewProject) {
      btnNewProject.onclick = openNewProjectModal;
    }

    if (btnModalProjectClose) {
      btnModalProjectClose.onclick = closeNewProjectModal;
    }

    if (btnModalProjectCancel) {
      btnModalProjectCancel.onclick = closeNewProjectModal;
    }

    if (btnModalProjectSubmit) {
      btnModalProjectSubmit.onclick = handleCreateProject;
    }

    if (btnProjectExit) {
      btnProjectExit.onclick = exitProject;
    }

    if (projectsSearchInput) {
      projectsSearchInput.oninput = (e) => loadProjectsList(e.target.value.trim());
    }

    // Subpanel tab switching
    const subpanelBtns = [
      { btn: tabProjChats, panel: 'chat' },
      { btn: tabProjFiles, panel: 'files' },
      { btn: tabProjKnowledge, panel: 'knowledge' },
      { btn: tabProjMemory, panel: 'memory' },
      { btn: tabProjSettings, panel: 'settings' }
    ];

    subpanelBtns.forEach(({ btn, panel }) => {
      if (btn) {
        btn.onclick = () => switchProjectSubpanel(panel);
      }
    });

    // Subpanel action bindings
    if (btnProjUploadFile && fileInputProject) {
      btnProjUploadFile.onclick = () => {
        fileInputProject.value = '';
        fileInputProject.click();
      };

      fileInputProject.onchange = async (e) => {
        const files = e.target.files;
        if (!files || files.length === 0 || !activeProjectId) return;
        for (let i = 0; i < files.length; i++) {
          await uploadProjectFile(files[i]);
        }
      };
    }

    if (btnProjSaveMem) {
      btnProjSaveMem.onclick = handleSaveProjectMemory;
    }

    if (btnSaveProjectSettings) {
      btnSaveProjectSettings.onclick = handleSaveProjectSettings;
    }

    if (btnArchiveProjectAction) {
      btnArchiveProjectAction.onclick = handleArchiveProject;
    }

    if (btnDeleteProjectTrigger) {
      btnDeleteProjectTrigger.onclick = openDeleteProjectModal;
    }

    if (btnModalDeleteClose) {
      btnModalDeleteClose.onclick = closeDeleteProjectModal;
    }

    if (btnModalDeleteCancel) {
      btnModalDeleteCancel.onclick = closeDeleteProjectModal;
    }

    if (btnModalDeleteConfirm) {
      btnModalDeleteConfirm.onclick = executeDeleteProject;
    }
  }

  function openNewProjectModal() {
    if (modalNewProject) {
      modalNewProject.classList.remove('hidden');
      if (newProjectName) newProjectName.value = '';
      if (newProjectDesc) newProjectDesc.value = '';
      if (newProjectIcon) newProjectIcon.value = '📁';
      if (newProjectInstructions) newProjectInstructions.value = '';
      if (newProjectName) newProjectName.focus();
    }
  }

  function closeNewProjectModal() {
    if (modalNewProject) modalNewProject.classList.add('hidden');
  }

  function openDeleteProjectModal() {
    if (!activeProjectData || !modalDeleteProject) return;
    if (deleteProjectNameDisplay) {
      deleteProjectNameDisplay.textContent = `"${activeProjectData.name}"`;
    }
    modalDeleteProject.classList.remove('hidden');
  }

  function closeDeleteProjectModal() {
    if (modalDeleteProject) modalDeleteProject.classList.add('hidden');
  }

  async function handleCreateProject() {
    const name = newProjectName ? newProjectName.value.trim() : '';
    if (!name) {
      showToast('Project name is required.');
      return;
    }
    const description = newProjectDesc ? newProjectDesc.value.trim() : '';
    const icon = (newProjectIcon ? newProjectIcon.value.trim() : '') || '📁';
    const instructions = newProjectInstructions ? newProjectInstructions.value.trim() : '';

    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          description: description,
          icon: icon,
          instructions: instructions,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showToast(data.detail || 'Could not create project');
        return;
      }
      closeNewProjectModal();
      showToast(`Created project "${name}"`);
      await loadProjectsList();
      await selectProject(data.id);
    } catch (e) {
      console.error('Create project error:', e);
      showToast('Failed to create project');
    }
  }

  async function loadProjectsList(filterQuery = '') {
    if (!projectsList) return;
    try {
      const res = await fetch('/api/projects');
      if (!res.ok) return;
      const data = await res.json();
      let projects = data.projects || [];

      if (filterQuery) {
        const q = filterQuery.toLowerCase();
        projects = projects.filter(p =>
          (p.name && p.name.toLowerCase().includes(q)) ||
          (p.description && p.description.toLowerCase().includes(q))
        );
      }

      projectsList.innerHTML = '';
      if (projects.length === 0) {
        projectsList.innerHTML = `
          <div class="empty-view-placeholder">
            <span>No projects found</span>
          </div>
        `;
        return;
      }

      projects.forEach(p => {
        const item = document.createElement('div');
        item.className = `project-list-item ${p.id === activeProjectId ? 'active' : ''}`;
        item.innerHTML = `
          <div class="project-item-icon">${escapeHtml(p.icon || '📁')}</div>
          <div class="project-item-content">
            <span class="project-item-name">${escapeHtml(p.name)}</span>
            <span class="project-item-desc">${escapeHtml(p.description || 'Workspace')}</span>
          </div>
          ${p.archived ? '<span class="project-item-badge">Archived</span>' : ''}
        `;
        item.onclick = () => selectProject(p.id);
        projectsList.appendChild(item);
      });
    } catch (e) {
      console.error('Error loading projects list:', e);
    }
  }

  async function selectProject(projectId) {
    try {
      const res = await fetch(`/api/projects/${projectId}`);
      if (!res.ok) {
        exitProject();
        return;
      }
      const proj = await res.json();
      activeProjectId = proj.id;
      activeProjectData = proj;
      localStorage.setItem('axen_active_project_id', proj.id);

      // Update workspace header banner
      if (projectWorkspaceHeader) projectWorkspaceHeader.classList.remove('hidden');
      if (projectHeaderIcon) projectHeaderIcon.textContent = proj.icon || '📁';
      if (projectHeaderTitle) projectHeaderTitle.textContent = proj.name;
      if (projectHeaderDesc) projectHeaderDesc.textContent = proj.description || 'Project Workspace';

      switchProjectSubpanel('chat');
      await loadSessions();
      renderConversationList();
      if (viewProjects && !viewProjects.classList.contains('hidden')) {
        loadProjectsList();
      }
    } catch (e) {
      console.error('Select project error:', e);
      exitProject();
    }
  }

  function exitProject() {
    activeProjectId = null;
    activeProjectData = null;
    localStorage.removeItem('axen_active_project_id');
    if (projectWorkspaceHeader) projectWorkspaceHeader.classList.add('hidden');
    switchProjectSubpanel('chat');
    loadSessions();
    if (viewProjects && !viewProjects.classList.contains('hidden')) {
      loadProjectsList();
    }
  }

  function switchProjectSubpanel(panel) {
    activeSubpanel = panel;
    [tabProjChats, tabProjFiles, tabProjKnowledge, tabProjMemory, tabProjSettings].forEach(b => {
      if (b) b.classList.remove('active');
    });
    [subpanelProjectFiles, subpanelProjectKnowledge, subpanelProjectMemory, subpanelProjectSettings].forEach(s => {
      if (s) s.classList.add('hidden');
    });

    const composerContainer = document.getElementById('composer-floating-container');

    if (panel === 'chat') {
      if (tabProjChats) tabProjChats.classList.add('active');
      if (messagesScrollArea) messagesScrollArea.classList.remove('hidden');
      if (composerContainer) composerContainer.classList.remove('hidden');
    } else {
      if (messagesScrollArea) messagesScrollArea.classList.add('hidden');
      if (composerContainer) composerContainer.classList.add('hidden');

      if (panel === 'files') {
        if (tabProjFiles) tabProjFiles.classList.add('active');
        if (subpanelProjectFiles) subpanelProjectFiles.classList.remove('hidden');
        loadProjectFiles();
      } else if (panel === 'knowledge') {
        if (tabProjKnowledge) tabProjKnowledge.classList.add('active');
        if (subpanelProjectKnowledge) subpanelProjectKnowledge.classList.remove('hidden');
        loadProjectKnowledge();
      } else if (panel === 'memory') {
        if (tabProjMemory) tabProjMemory.classList.add('active');
        if (subpanelProjectMemory) subpanelProjectMemory.classList.remove('hidden');
        loadProjectMemory();
      } else if (panel === 'settings') {
        if (tabProjSettings) tabProjSettings.classList.add('active');
        if (subpanelProjectSettings) subpanelProjectSettings.classList.remove('hidden');
        loadProjectSettings();
      }
    }
  }

  async function loadProjectFiles() {
    if (!activeProjectId || !projFilesContainer) return;
    try {
      const res = await fetch(`/api/projects/${activeProjectId}/files`);
      if (!res.ok) return;
      const data = await res.json();
      const files = data.files || [];

      projFilesContainer.innerHTML = '';
      if (files.length === 0) {
        projFilesContainer.innerHTML = '<div class="empty-view-placeholder">No files in this project workspace. Click Upload File above to add files.</div>';
        return;
      }

      files.forEach(f => {
        const row = document.createElement('div');
        row.className = 'file-list-item';
        row.innerHTML = `
          <div class="file-item-icon doc">DOC</div>
          <div class="file-item-info">
            <span class="file-item-name">${escapeHtml(f.original_filename || f.file_id)}</span>
            <span class="file-item-meta">${formatFileSize(f.size || 0)}</span>
          </div>
          <button class="btn-item-action" title="Remove from project">&times;</button>
        `;
        const delBtn = row.querySelector('.btn-item-action');
        delBtn.onclick = async () => {
          await fetch(`/api/projects/${activeProjectId}/files/${f.file_id}`, { method: 'DELETE' });
          loadProjectFiles();
        };
        projFilesContainer.appendChild(row);
      });
    } catch (e) {
      console.error('Load project files error:', e);
    }
  }

  async function uploadProjectFile(file) {
    if (!activeProjectId) return;
    const formData = new FormData();
    formData.append('file', file);
    showToast(`Uploading ${file.name} to project...`);

    try {
      const uploadRes = await fetch('/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) {
        showToast(uploadData.detail || 'Upload failed');
        return;
      }

      const attachRes = await fetch(`/api/projects/${activeProjectId}/files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: uploadData.file_id }),
      });

      if (attachRes.ok) {
        showToast(`Attached ${file.name} to workspace`);
        loadProjectFiles();
      }
    } catch (e) {
      console.error('Project file upload error:', e);
      showToast('Failed to upload project file');
    }
  }

  async function loadProjectKnowledge() {
    if (!activeProjectId || !projKnowledgeContainer) return;
    try {
      const res = await fetch(`/api/projects/${activeProjectId}/knowledge`);
      if (!res.ok) return;
      const data = await res.json();
      const collections = data.collections || [];

      projKnowledgeContainer.innerHTML = '';
      if (collections.length === 0) {
        projKnowledgeContainer.innerHTML = '<div class="empty-view-placeholder">No isolated knowledge collections in this workspace.</div>';
        return;
      }

      collections.forEach(c => {
        const item = document.createElement('div');
        item.className = 'file-list-item';
        item.innerHTML = `
          <div class="file-item-icon">KNOW</div>
          <div class="file-item-info">
            <span class="file-item-name">${escapeHtml(c.collection_name)}</span>
            <span class="file-item-meta">${escapeHtml(c.description || 'Collection')}</span>
          </div>
        `;
        projKnowledgeContainer.appendChild(item);
      });
    } catch (e) {
      console.error('Load project knowledge error:', e);
    }
  }

  async function loadProjectMemory() {
    if (!activeProjectId || !projMemoryContainer) return;
    try {
      const res = await fetch(`/api/projects/${activeProjectId}/memory`);
      if (!res.ok) return;
      const data = await res.json();
      const memories = data.memories || {};
      const keys = Object.keys(memories);

      projMemoryContainer.innerHTML = '';
      if (keys.length === 0) {
        projMemoryContainer.innerHTML = '<div class="empty-view-placeholder">No memories stored for this project yet. Add facts above.</div>';
        return;
      }

      keys.forEach(k => {
        const card = document.createElement('div');
        card.className = 'file-list-item';
        card.innerHTML = `
          <div class="file-item-info">
            <span class="file-item-name" style="font-weight:700; color:var(--axen-orange);">${escapeHtml(k)}</span>
            <span class="file-item-meta">${escapeHtml(memories[k])}</span>
          </div>
          <button class="btn-item-action" title="Delete memory">&times;</button>
        `;
        const delBtn = card.querySelector('.btn-item-action');
        delBtn.onclick = async () => {
          await fetch(`/api/projects/${activeProjectId}/memory/${encodeURIComponent(k)}`, { method: 'DELETE' });
          loadProjectMemory();
        };
        projMemoryContainer.appendChild(card);
      });
    } catch (e) {
      console.error('Load project memory error:', e);
    }
  }

  async function handleSaveProjectMemory() {
    if (!activeProjectId) return;
    const key = projMemKeyInput ? projMemKeyInput.value.trim() : '';
    const val = projMemValInput ? projMemValInput.value.trim() : '';
    if (!key || !val) {
      showToast('Key and value are both required.');
      return;
    }

    try {
      const res = await fetch(`/api/projects/${activeProjectId}/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key, value: val }),
      });
      if (res.ok) {
        showToast(`Saved memory for ${key}`);
        if (projMemKeyInput) projMemKeyInput.value = '';
        if (projMemValInput) projMemValInput.value = '';
        loadProjectMemory();
      }
    } catch (e) {
      console.error('Save project memory error:', e);
      showToast('Failed to save memory');
    }
  }

  function loadProjectSettings() {
    if (!activeProjectData) return;
    const settings = activeProjectData.settings || {};
    if (projSettingsInstructions) projSettingsInstructions.value = settings.instructions || '';
    if (toggleProjMemory) toggleProjMemory.checked = settings.memory_enabled !== false;
    if (toggleProjKnowledge) toggleProjKnowledge.checked = settings.knowledge_access_enabled !== false;
    if (toggleProjFiles) toggleProjFiles.checked = settings.file_access_enabled !== false;
    if (toggleProjWeb) toggleProjWeb.checked = settings.web_access_enabled !== false;
  }

  async function handleSaveProjectSettings() {
    if (!activeProjectId) return;
    const instructions = projSettingsInstructions ? projSettingsInstructions.value.trim() : '';
    const settings = {
      instructions: instructions,
      memory_enabled: toggleProjMemory ? toggleProjMemory.checked : true,
      knowledge_access_enabled: toggleProjKnowledge ? toggleProjKnowledge.checked : true,
      file_access_enabled: toggleProjFiles ? toggleProjFiles.checked : true,
      web_access_enabled: toggleProjWeb ? toggleProjWeb.checked : true,
    };

    try {
      const res = await fetch(`/api/projects/${activeProjectId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: settings }),
      });
      if (res.ok) {
        const updated = await res.json();
        activeProjectData = updated;
        showToast('Settings saved successfully.');
      } else {
        showToast('Could not save settings');
      }
    } catch (e) {
      console.error('Save settings error:', e);
      showToast('Error saving settings');
    }
  }

  async function handleArchiveProject() {
    if (!activeProjectId) return;
    try {
      const res = await fetch(`/api/projects/${activeProjectId}/archive`, { method: 'POST' });
      if (res.ok) {
        showToast('Project archived.');
        exitProject();
      }
    } catch (e) {
      console.error('Archive error:', e);
    }
  }

  async function executeDeleteProject() {
    if (!activeProjectId) return;
    try {
      const res = await fetch(`/api/projects/${activeProjectId}`, { method: 'DELETE' });
      if (res.ok) {
        closeDeleteProjectModal();
        showToast('Project deleted.');
        exitProject();
      } else {
        showToast('Could not delete project');
      }
    } catch (e) {
      console.error('Delete project error:', e);
      showToast('Error deleting project');
    }
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
