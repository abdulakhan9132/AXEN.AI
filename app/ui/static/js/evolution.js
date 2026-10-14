/**
 * app/ui/static/js/evolution.js
 * =============================
 * Controller for the Self-Evolution Engine UI Dashboard.
 */

(function () {
  'use strict';
  const escape = value => String(value == null ? '' : value).replace(/[&<>"']/g,
    char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

  window.EvolutionDashboard = {
    init: function () {
      const btnRefresh = document.getElementById('btn-evolution-refresh');
      if (btnRefresh) {
        btnRefresh.onclick = () => this.loadDashboard();
      }
    },

    loadDashboard: async function () {
      await Promise.all([
        this.fetchStatus(),
        this.fetchApprovals(),
        this.fetchExperiments(),
      ]);
    },

    fetchStatus: async function () {
      try {
        const res = await fetch('/api/evolution/status');
        if (!res.ok) return;
        const data = await res.json();
        const totalEl = document.getElementById('evo-total-exp');
        const pendingEl = document.getElementById('evo-pending-app');
        const successEl = document.getElementById('evo-success-imp');
        const failedEl = document.getElementById('evo-failed-exp');

        if (totalEl) totalEl.textContent = data.total_experiments || 0;
        if (pendingEl) pendingEl.textContent = data.pending_approvals || 0;
        if (successEl) successEl.textContent = data.successful_improvements || 0;
        if (failedEl) failedEl.textContent = data.failed_experiments || 0;
      } catch (err) {
        console.warn('Failed to load evolution status:', err);
      }
    },

    fetchApprovals: async function () {
      const container = document.getElementById('pending-approvals-list');
      if (!container) return;

      try {
        const res = await fetch('/api/evolution/approvals');
        if (!res.ok) return;
        const data = await res.json();
        const approvals = data.approvals || [];

        if (approvals.length === 0) {
          container.innerHTML = '<div style="font-size:11px; color:#9ca3af; text-align:center; padding:12px;">No pending code improvements.</div>';
          return;
        }

        container.innerHTML = '';
        approvals.forEach((appr) => {
          const card = document.createElement('div');
          card.className = 'approval-card';
          card.innerHTML = `
            <div class="approval-card-header">
              <span class="approval-target-file" title="${escape(appr.target_file)}">${escape(appr.target_file || appr.title || 'Target file')}</span>
              <span class="approval-risk-badge">${escape(appr.risk_level || 'medium')}</span>
            </div>
            <div class="approval-diff-preview">${escape(appr.diff ? appr.diff.slice(0, 300) : JSON.stringify(appr.benchmark_summary || {}))}</div>
            <div class="approval-btn-group">
              <button class="btn-approve" data-id="${escape(appr.request_id)}">Approve & Merge</button>
              <button class="btn-reject" data-id="${escape(appr.request_id)}">Reject</button>
            </div>
          `;

          const approveBtn = card.querySelector('.btn-approve');
          const rejectBtn = card.querySelector('.btn-reject');

          approveBtn.onclick = async () => {
            approveBtn.disabled = true;
            approveBtn.textContent = 'Merging...';
            try {
              const postRes = await fetch(`/api/evolution/approve/${appr.request_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-AXEN-Control': 'operator' },
                body: JSON.stringify({ reviewer: 'human_operator', apply_merge: true }),
              });
              if (postRes.ok) {
                if (window.showToast) window.showToast('Candidate approved and merged successfully!');
                EvolutionDashboard.loadDashboard();
              } else {
                alert('Approval failed');
              }
            } catch (e) {
              alert('Network error approving change');
            }
          };

          rejectBtn.onclick = async () => {
            const reason = prompt('Reason for rejection:') || 'User declined proposal';
            try {
              const postRes = await fetch(`/api/evolution/reject/${appr.request_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-AXEN-Control': 'operator' },
                body: JSON.stringify({ reviewer: 'human_operator', reason: reason }),
              });
              if (postRes.ok) {
                if (window.showToast) window.showToast('Candidate rejected.');
                EvolutionDashboard.loadDashboard();
              }
            } catch (e) {
              alert('Network error rejecting change');
            }
          };

          container.appendChild(card);
        });
      } catch (err) {
        console.warn('Failed to load pending approvals:', err);
      }
    },

    fetchExperiments: async function () {
      const container = document.getElementById('experiments-list');
      if (!container) return;

      try {
        const res = await fetch('/api/evolution/experiments?limit=15');
        if (!res.ok) return;
        const data = await res.json();
        const experiments = data.experiments || [];

        if (experiments.length === 0) {
          container.innerHTML = '<div style="font-size:11px; color:#9ca3af; text-align:center; padding:12px;">No experiments run yet.</div>';
          return;
        }

        container.innerHTML = '';
        experiments.forEach((exp) => {
          const card = document.createElement('div');
          card.className = 'experiment-card';
          const statusClass = `status-${(exp.status || 'created').toLowerCase().replace(/[^a-z_]/g, '')}`;

          card.innerHTML = `
            <div class="experiment-card-top">
              <span class="experiment-target">${escape(exp.target_module || exp.target_file || 'Experiment')}</span>
              <span class="experiment-status ${statusClass}">${escape(exp.status)}</span>
            </div>
            <div class="experiment-desc">${escape(exp.weakness_description || 'Routine improvement cycle')}</div>
            <div style="font-size: 9px; color: #6b7280; display:flex; justify-content:space-between; margin-top:2px;">
              <span>Candidates: ${(exp.candidate_ids || []).length}</span>
              <span>Gen: ${escape(exp.generation)}</span>
            </div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        console.warn('Failed to load experiments list:', err);
      }
    },
  };

  document.addEventListener('DOMContentLoaded', () => {
    window.EvolutionDashboard.init();
  });
})();
