(function () {
  'use strict';
  const headers = { 'Content-Type': 'application/json', 'X-AXEN-Control': 'operator' };
  let stream = null;
  let refreshPending = false;
  const nodes = {};
  const logs = { research: [], skill: [], code: [], training: [] };

  function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  async function control(action) {
    nodes.error.textContent = '';
    try {
      const result = await fetch('/api/autonomy/control', {
        method: 'POST', headers,
        body: JSON.stringify({ action, mode: action === 'start' ? nodes.mode.value : null,
          offline_demo: nodes.offline.checked }),
      });
      const data = await result.json();
      if (!result.ok) throw new Error(data.detail || 'Control failed');
      renderStatus(data);
    } catch (error) { nodes.error.textContent = error.message; }
  }

  function renderStatus(data) {
    const background = data.background_agent;
    nodes.agentStatus.textContent = background
      ? `Agent 1 · Chat answers\nAgent 2 · ${background.status}\nAutomatic evaluation: ${background.enabled ? 'On' : 'Off'}\nReviews: ${JSON.stringify(background.reviews)}\nAutomatic tested deployment: ${background.auto_apply_tested_changes ? 'On' : 'Off'}\nCode targets: ${background.code_targets}\nModel training: ${background.training_configured ? 'Configured' : 'Not configured'}\n${background.scope}`
      : 'Background agent status unavailable until the server restarts.';
    if (!nodes.agentStatus.dataset.ready) {
      nodes.advanced.open = !background || !background.enabled;
      if (background) nodes.agentStatus.dataset.ready = 'true';
    }
    nodes.current.textContent = `Worker: ${data.status}\nMode: ${data.mode}\n` +
      `Goal: ${data.current_goal || 'None'}\nSession started: ${data.session_started_at || 'Not started'}\n` +
      `Remaining session budget: ${data.remaining_seconds}s\nEmergency stop: ${data.emergency_latched ? 'Latched' : 'Clear'}\n` +
      `Resources: ${JSON.stringify(data.resources)}`;
    nodes.queue.textContent = JSON.stringify(data.goals.map(g => ({
      objective: g.objective, reason: g.reason, status: g.status, step: g.step,
      improvement_demonstrated: g.improvement_demonstrated, error: g.error,
    })), null, 2);
    nodes.start.disabled = data.mode !== 'OFF' || data.emergency_latched;
    nodes.pause.disabled = !['running', 'idle', 'resource_paused'].includes(data.status);
    nodes.resume.disabled = data.status !== 'paused';
    nodes.reset.disabled = !data.emergency_latched;
  }

  async function refresh() {
    if (refreshPending) return;
    refreshPending = true;
    try {
      const response = await fetch('/api/autonomy/status');
      if (!response.ok) throw new Error('Worker status unavailable');
      renderStatus(await response.json());
    } catch (error) { nodes.error.textContent = error.message; }
    finally { refreshPending = false; }
  }

  function receive(event) {
    nodes.step.textContent = `${event.at}: ${event.kind}` +
      (event.kind === 'activity' ? ` / ${event.data.step} / ${event.data.objective}` : '');
    let group = null;
    if (['source', 'findings', 'cross_check', 'research_limitation', 'video', 'connector_read'].includes(event.kind)) group = 'research';
    if (['baseline', 'practice', 'evaluation', 'answer_evaluation', 'improvement_needed', 'knowledge_stored', 'completed', 'failed', 'background_error'].includes(event.kind)) group = 'skill';
    if (['code_evolution', 'approval_requested', 'automatic_change_applied', 'automatic_change_deferred'].includes(event.kind)) group = 'code';
    if (event.kind.startsWith('training_')) group = 'training';
    if (group) {
      logs[group].push(event);
      logs[group] = logs[group].slice(-30);
      nodes[group].replaceChildren();
      for (const item of logs[group]) {
        const row = element('div', undefined, nodes[group]);
        element('strong', `${item.at} · ${item.kind}`, row);
        if (item.kind === 'source' && item.data.url.startsWith('https://')) {
          const link = element('a', item.data.url, row);
          link.href = item.data.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
        }
        element('pre', JSON.stringify(item.data, null, 2), row);
      }
    }
    refresh();
  }

  document.addEventListener('DOMContentLoaded', () => {
    const parent = document.querySelector('#view-evolution .evolution-panel-container');
    if (!parent) return;
    const panel = element('section'); panel.className = 'autonomy-panel'; parent.prepend(panel);
    element('h3', 'Two-agent self-improvement', panel);
    element('p', 'Chat answers have priority. The second agent evaluates answers automatically and schedules bounded improvement work when resources are available.', panel);
    nodes.agentStatus = element('pre', 'Checking agents…', panel); nodes.agentStatus.className = 'autonomy-evidence';
    nodes.advanced = element('details', undefined, panel);
    element('summary', 'Advanced manual session controls', nodes.advanced);
    const controls = element('div', undefined, nodes.advanced); controls.className = 'autonomy-controls';
    const modeLabel = element('label', 'Operating mode ', controls);
    nodes.mode = element('select', undefined, modeLabel);
    for (const mode of ['LEARNING', 'DEVELOPMENT', 'SUPERVISED EVOLUTION', 'OFF']) {
      const option = element('option', mode, nodes.mode); option.value = mode;
    }
    const offline = element('label', undefined, nodes.advanced);
    nodes.offline = element('input', undefined, offline); nodes.offline.type = 'checkbox';
    offline.append(document.createTextNode(' Explicit offline demo using local lesson fixture'));
    for (const [key, title, action] of [
      ['start', 'Start learning', 'start'], ['pause', 'Pause', 'pause'],
      ['resume', 'Resume', 'resume'], ['stop', 'Stop', 'stop'],
      ['emergency', 'Emergency stop', 'emergency-stop'], ['reset', 'Reset emergency stop', 'reset-emergency'],
    ]) {
      nodes[key] = element('button', title, ['emergency', 'reset'].includes(key) ? panel : controls); nodes[key].type = 'button';
      nodes[key].onclick = () => control(action);
      if (key === 'emergency') nodes[key].className = 'autonomy-emergency';
    }
    nodes.error = element('p', '', panel); nodes.error.setAttribute('role', 'alert');
    for (const [key, title] of [['current', 'Current activity'], ['step', 'Latest worker event'],
      ['queue', 'Goal queue'], ['research', 'Research activity'], ['skill', 'Skill development'],
      ['code', 'Code evolution'], ['training', 'Neural training']]) {
      element('h4', title, panel);
      nodes[key] = element(['current', 'queue', 'step'].includes(key) ? 'pre' : 'div',
        'No worker evidence recorded yet.', panel);
      nodes[key].className = 'autonomy-evidence';
    }
    refresh();
    const statusTimer = setInterval(refresh, 10000);
    stream = new EventSource('/api/autonomy/stream');
    stream.onmessage = message => {
      try { receive(JSON.parse(message.data)); }
      catch (error) { nodes.error.textContent = 'Invalid worker event'; }
    };
    stream.onerror = () => { nodes.error.textContent = 'Live connection interrupted; reconnecting automatically.'; };
    window.addEventListener('beforeunload', () => { stream.close(); clearInterval(statusTimer); });
  });
})();
