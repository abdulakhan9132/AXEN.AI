// Axen's language brain: a small GPT-style transformer written from scratch.
// Plain Node.js, no libraries, hand-written backpropagation, Adam optimizer.
const fs = require("fs");
const { VOCAB_SIZE } = require("./tokenizer");
const F = Float32Array;

const DEFAULT_CFG = { T: 48, C: 48, H: 4, L: 2, lr: 0.003, batch: 8 };

function randn() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// y[t][j] = sum_i x[t][i] * W[i*out + j]
function matmul(x, W, n, inD, outD) {
  const y = new F(n * outD);
  for (let t = 0; t < n; t++)
    for (let i = 0; i < inD; i++) {
      const xv = x[t * inD + i];
      if (xv === 0) continue;
      const wo = i * outD, yo = t * outD;
      for (let j = 0; j < outD; j++) y[yo + j] += xv * W[wo + j];
    }
  return y;
}
// returns dx, accumulates dW
function matmulBack(dy, x, W, dW, n, inD, outD) {
  const dx = new F(n * inD);
  for (let t = 0; t < n; t++)
    for (let i = 0; i < inD; i++) {
      const xv = x[t * inD + i];
      const wo = i * outD, yo = t * outD;
      let s = 0;
      for (let j = 0; j < outD; j++) {
        const g = dy[yo + j];
        s += g * W[wo + j];
        dW[wo + j] += xv * g;
      }
      dx[t * inD + i] = s;
    }
  return dx;
}
// RMSNorm (no gain): y = x / sqrt(mean(x^2) + eps)
function rms(x, n, C) {
  const y = new F(n * C), r = new F(n);
  for (let t = 0; t < n; t++) {
    let s = 0;
    for (let c = 0; c < C; c++) s += x[t * C + c] ** 2;
    const inv = 1 / Math.sqrt(s / C + 1e-5);
    r[t] = inv;
    for (let c = 0; c < C; c++) y[t * C + c] = x[t * C + c] * inv;
  }
  return { y, r };
}
function rmsBack(dy, x, r, n, C) {
  const dx = new F(n * C);
  for (let t = 0; t < n; t++) {
    let dot = 0;
    for (let c = 0; c < C; c++) dot += dy[t * C + c] * x[t * C + c];
    const inv = r[t];
    for (let c = 0; c < C; c++)
      dx[t * C + c] = inv * dy[t * C + c] - (inv * inv * inv * x[t * C + c] * dot) / C;
  }
  return dx;
}

class Model {
  constructor(cfg) {
    this.cfg = { ...DEFAULT_CFG, ...(cfg || {}) };
    this.step = 0;      // Adam timestep
    this.trained = 0;   // total training steps ever done
    this.loss = null;
    this.P = {};
    this.initParams();
    this.resetOptimizer();
  }

  initParams() {
    const { T, C, L } = this.cfg, V = VOCAB_SIZE;
    const add = (name, n, std) => {
      const a = new F(n);
      for (let i = 0; i < n; i++) a[i] = randn() * std;
      this.P[name] = a;
    };
    add("wte", V * C, 0.1);
    add("wpe", T * C, 0.05);
    for (let l = 0; l < L; l++) {
      for (const n of ["q", "k", "v"]) add(`l${l}.${n}`, C * C, 1 / Math.sqrt(C));
      add(`l${l}.o`, C * C, 0.5 / Math.sqrt(C));
      add(`l${l}.f1`, C * 4 * C, 1 / Math.sqrt(C));
      add(`l${l}.f2`, 4 * C * C, 0.5 / Math.sqrt(4 * C));
    }
  }

  resetOptimizer() {
    this.G = {}; this.M = {}; this.Vv = {};
    for (const k in this.P) {
      this.G[k] = new F(this.P[k].length);
      this.M[k] = new F(this.P[k].length);
      this.Vv[k] = new F(this.P[k].length);
    }
  }

  paramCount() {
    let n = 0;
    for (const k in this.P) n += this.P[k].length;
    return n;
  }

  // Forward pass over one sequence. Returns logits and a cache for backprop.
  forward(tokens) {
    const { C, H, L } = this.cfg, V = VOCAB_SIZE, P = this.P;
    const n = tokens.length, hs = C / H, scale = 1 / Math.sqrt(hs);
    let h = new F(n * C);
    for (let t = 0; t < n; t++)
      for (let c = 0; c < C; c++) h[t * C + c] = P.wte[tokens[t] * C + c] + P.wpe[t * C + c];

    const layers = [];
    for (let l = 0; l < L; l++) {
      const c = { xin: h };
      const a1 = rms(h, n, C); c.a1 = a1.y; c.r1 = a1.r;
      c.q = matmul(c.a1, P[`l${l}.q`], n, C, C);
      c.k = matmul(c.a1, P[`l${l}.k`], n, C, C);
      c.v = matmul(c.a1, P[`l${l}.v`], n, C, C);
      c.att = new F(H * n * n);
      c.out = new F(n * C);
      for (let hd = 0; hd < H; hd++)
        for (let t = 0; t < n; t++) {
          const ao = (hd * n + t) * n;
          let max = -1e30;
          for (let s = 0; s <= t; s++) {
            let d = 0;
            for (let e = 0; e < hs; e++) d += c.q[t * C + hd * hs + e] * c.k[s * C + hd * hs + e];
            d *= scale; c.att[ao + s] = d;
            if (d > max) max = d;
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) { c.att[ao + s] = Math.exp(c.att[ao + s] - max); sum += c.att[ao + s]; }
          for (let s = 0; s <= t; s++) {
            c.att[ao + s] /= sum;
            const w = c.att[ao + s];
            for (let e = 0; e < hs; e++) c.out[t * C + hd * hs + e] += w * c.v[s * C + hd * hs + e];
          }
        }
      const o = matmul(c.out, P[`l${l}.o`], n, C, C);
      c.h2 = new F(n * C);
      for (let i = 0; i < n * C; i++) c.h2[i] = h[i] + o[i];
      const a2 = rms(c.h2, n, C); c.a2 = a2.y; c.r2 = a2.r;
      c.f = matmul(c.a2, P[`l${l}.f1`], n, C, 4 * C);
      c.fa = c.f.map((v) => (v > 0 ? v : 0));
      const m = matmul(c.fa, P[`l${l}.f2`], n, 4 * C, C);
      h = new F(n * C);
      for (let i = 0; i < n * C; i++) h[i] = c.h2[i] + m[i];
      layers.push(c);
    }
    const af = rms(h, n, C);
    const logits = new F(n * V);
    for (let t = 0; t < n; t++)
      for (let v = 0; v < V; v++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += af.y[t * C + c] * P.wte[v * C + c];
        logits[t * V + v] = s;
      }
    return { logits, cache: { tokens, layers, hfinal: h, af: af.y, rf: af.r } };
  }

  // Loss + gradients for one sequence (gradients are ACCUMULATED into this.G).
  lossAndGrad(tokens, targets, gradScale) {
    const { C, H, L } = this.cfg, V = VOCAB_SIZE, P = this.P, G = this.G;
    const n = tokens.length, hs = C / H, scale = 1 / Math.sqrt(hs);
    const { logits, cache } = this.forward(tokens);
    let loss = 0;
    const dlog = new F(n * V);
    for (let t = 0; t < n; t++) {
      let max = -1e30;
      for (let v = 0; v < V; v++) if (logits[t * V + v] > max) max = logits[t * V + v];
      let sum = 0;
      for (let v = 0; v < V; v++) { dlog[t * V + v] = Math.exp(logits[t * V + v] - max); sum += dlog[t * V + v]; }
      for (let v = 0; v < V; v++) dlog[t * V + v] /= sum;
      loss -= Math.log(dlog[t * V + targets[t]] + 1e-9);
      dlog[t * V + targets[t]] -= 1;
      for (let v = 0; v < V; v++) dlog[t * V + v] *= gradScale / n;
    }
    loss /= n;

    // output layer (tied with wte)
    const daf = new F(n * C);
    for (let t = 0; t < n; t++)
      for (let v = 0; v < V; v++) {
        const g = dlog[t * V + v];
        if (g === 0) continue;
        for (let c = 0; c < C; c++) {
          daf[t * C + c] += g * P.wte[v * C + c];
          G.wte[v * C + c] += g * cache.af[t * C + c];
        }
      }
    let dh = rmsBack(daf, cache.hfinal, cache.rf, n, C);

    for (let l = L - 1; l >= 0; l--) {
      const c = cache.layers[l];
      // MLP
      const dfa = matmulBack(dh, c.fa, P[`l${l}.f2`], G[`l${l}.f2`], n, 4 * C, C);
      const df = dfa.map((g, i) => (c.f[i] > 0 ? g : 0));
      const da2 = matmulBack(df, c.a2, P[`l${l}.f1`], G[`l${l}.f1`], n, C, 4 * C);
      const dh2 = rmsBack(da2, c.h2, c.r2, n, C);
      for (let i = 0; i < n * C; i++) dh2[i] += dh[i];
      // attention
      const dout = matmulBack(dh2, c.out, P[`l${l}.o`], G[`l${l}.o`], n, C, C);
      const dq = new F(n * C), dk = new F(n * C), dv = new F(n * C);
      for (let hd = 0; hd < H; hd++)
        for (let t = 0; t < n; t++) {
          const ao = (hd * n + t) * n;
          const datt = new F(t + 1);
          let dot = 0;
          for (let s = 0; s <= t; s++) {
            let d = 0;
            for (let e = 0; e < hs; e++) {
              d += dout[t * C + hd * hs + e] * c.v[s * C + hd * hs + e];
              dv[s * C + hd * hs + e] += c.att[ao + s] * dout[t * C + hd * hs + e];
            }
            datt[s] = d; dot += c.att[ao + s] * d;
          }
          for (let s = 0; s <= t; s++) {
            const ds = c.att[ao + s] * (datt[s] - dot) * scale;
            for (let e = 0; e < hs; e++) {
              dq[t * C + hd * hs + e] += ds * c.k[s * C + hd * hs + e];
              dk[s * C + hd * hs + e] += ds * c.q[t * C + hd * hs + e];
            }
          }
        }
      const da1 = matmulBack(dq, c.a1, P[`l${l}.q`], G[`l${l}.q`], n, C, C);
      const dak = matmulBack(dk, c.a1, P[`l${l}.k`], G[`l${l}.k`], n, C, C);
      const dav = matmulBack(dv, c.a1, P[`l${l}.v`], G[`l${l}.v`], n, C, C);
      for (let i = 0; i < n * C; i++) da1[i] += dak[i] + dav[i];
      const dx = rmsBack(da1, c.xin, c.r1, n, C);
      for (let i = 0; i < n * C; i++) dx[i] += dh2[i];
      dh = dx;
    }
    for (let t = 0; t < n; t++)
      for (let c = 0; c < C; c++) {
        G.wte[cache.tokens[t] * C + c] += dh[t * C + c];
        G.wpe[t * C + c] += dh[t * C + c];
      }
    return loss;
  }

  adamStep(lr) {
    this.step++;
    this.trained++;
    const b1 = 0.9, b2 = 0.99, eps = 1e-8;
    let norm = 0;
    for (const k in this.G) for (let i = 0; i < this.G[k].length; i++) norm += this.G[k][i] ** 2;
    norm = Math.sqrt(norm);
    const clip = norm > 1 ? 1 / norm : 1;
    const bc1 = 1 - Math.pow(b1, this.step), bc2 = 1 - Math.pow(b2, this.step);
    for (const k in this.P) {
      const p = this.P[k], g = this.G[k], m = this.M[k], v = this.Vv[k];
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * clip;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
        g[i] = 0;
      }
    }
  }

  // One training step on a random batch. ids = array of token ids for the whole corpus.
  trainStep(ids) {
    const { T, batch, lr } = this.cfg;
    let total = 0;
    for (let b = 0; b < batch; b++) {
      const s = Math.floor(Math.random() * (ids.length - T - 1));
      total += this.lossAndGrad(ids.slice(s, s + T), ids.slice(s + 1, s + T + 1), 1 / batch);
    }
    this.adamStep(lr);
    this.loss = total / batch;
    return this.loss;
  }

  // Generate text. Streams each new token id through onToken(id).
  async generate(promptIds, { temperature = 0.8, topK = 20, maxTokens = 120, stop = [], onToken } = {}) {
    const { T } = this.cfg, V = VOCAB_SIZE;
    const ids = promptIds.slice();
    const out = [];
    for (let i = 0; i < maxTokens; i++) {
      const ctx = ids.slice(-T);
      const { logits } = this.forward(ctx);
      const off = (ctx.length - 1) * V;
      let cand = [];
      for (let v = 0; v < V; v++) cand.push([v, logits[off + v] / Math.max(temperature, 0.05)]);
      cand.sort((a, b) => b[1] - a[1]);
      cand = cand.slice(0, Math.max(1, topK));
      const mx = cand[0][1];
      const ex = cand.map(([v, l]) => [v, Math.exp(l - mx)]);
      const tot = ex.reduce((a, b) => a + b[1], 0);
      let r = Math.random() * tot, pick = ex[0][0];
      for (const [v, w] of ex) { r -= w; if (r <= 0) { pick = v; break; } }
      ids.push(pick); out.push(pick);
      if (onToken) onToken(pick);
      // Let Node flush streamed token events before computing the next token.
      await new Promise((resolve) => setImmediate(resolve));
      if (stop.length) {
        const txt = require("./tokenizer").decode(out);
        if (stop.some((s) => txt.endsWith(s))) break;
      }
    }
    return out;
  }

  save(file) {
    const params = {};
    for (const k in this.P) params[k] = Array.from(this.P[k]);
    fs.mkdirSync(require("path").dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ cfg: this.cfg, trained: this.trained, loss: this.loss, params }));
  }

  static load(file) {
    const j = JSON.parse(fs.readFileSync(file, "utf8"));
    const m = new Model(j.cfg);
    for (const k in j.params) m.P[k] = F.from(j.params[k]);
    m.trained = j.trained || 0;
    m.loss = j.loss ?? null;
    m.resetOptimizer(); // Adam state restarts; weights are kept
    return m;
  }
}

module.exports = { Model, DEFAULT_CFG };
