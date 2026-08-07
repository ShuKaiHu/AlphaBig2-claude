/* AlphaBig2 web 對戰台 — 前端狀態機
 * 所有遊戲邏輯都在後端(engine/big2Game);這裡只做顯示與輸入。 */
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = {
  game: null,        // 後端回傳的 state_json
  selected: new Set(), // 選取中的手牌 card id
  logLines: [],      // {player,label,type,me,ctrl,time,note}
  busy: false,
};

const SUIT_CHAR = { d: '♦', c: '♣', h: '♥', s: '♠' };
const SUIT_RED = { d: true, h: true, c: false, s: false };

// ── helpers ──────────────────────────────────────────────────────────────────

function cardEl(c, opts = {}) {
  const el = document.createElement('div');
  el.className = 'card' + (SUIT_RED[c.suit] ? ' red' : '') + (opts.small ? ' small' : '');
  el.innerHTML = `<span>${c.rank}</span><span class="suit">${SUIT_CHAR[c.suit]}</span>`;
  el.dataset.id = c.id;
  return el;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function post(path, body) {
  return api(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

function cfgFromForm() {
  return {
    checkpoint: $('#cfg-checkpoint').value,
    mode: $('#cfg-mode').value,
    sims: +$('#cfg-sims').value,
    dets: +$('#cfg-dets').value,
    c_puct: +$('#cfg-cpuct').value,
    dirichlet: +$('#cfg-dirichlet').value,
    belief: $('#cfg-belief').checked,
    use_value_net: $('#cfg-valuenet').checked,
    human_seat: $('#cfg-seat').value,
    seed: $('#cfg-seed').value.trim(),
    reveal: $('#cfg-reveal').checked,
    log_ai_analysis: $('#cfg-ainotes').checked,
  };
}

// ── log ──────────────────────────────────────────────────────────────────────

function pushAiMoves(moves) {
  for (const m of moves) {
    state.logLines.push({
      player: m.player, label: m.label, type: m.type, me: false,
      ctrl: m.control_break, time: m.time, note: m.policy_note,
    });
  }
}

function renderLog() {
  const g = state.game;
  const box = $('#tab-log');
  box.innerHTML = '';
  // 以後端 history 為準(含 auto-pass);AI 耗時/policy note 另列於下方
  for (const h of (g ? g.history : [])) {
    const div = document.createElement('div');
    const me = g && h.player === g.human_seat;
    div.className = 'logline ' + (h.type === 'pass' ? 'pass' : '') + (me ? ' me' : '');
    div.innerHTML =
      `<span class="who">${me ? '你' : 'P' + h.player}</span>${h.label}` +
      (h.control_break ? '<span class="ctrl">⚑控場</span>' : '');
    box.appendChild(div);
  }
  // AI note 行(最近 30 條)
  const notes = state.logLines.filter(l => l.note).slice(-30);
  if (notes.length) {
    const h4 = document.createElement('h4');
    h4.className = 'sect'; h4.textContent = 'AI policy notes(最近)';
    box.appendChild(h4);
    for (const l of notes) {
      const div = document.createElement('div');
      div.className = 'logline';
      div.innerHTML = `<span class="who">P${l.player}</span>${l.label}` +
        `<span class="t">${l.time}s</span>` +
        `<div class="note">top3: ${l.note.top3.map(t => `${t.label} ${(t.p * 100).toFixed(1)}%`).join(' | ')}` +
        ` · V=[${l.note.value.join(', ')}]</div>`;
      box.appendChild(div);
    }
  }
  box.scrollTop = box.scrollHeight;
}

// ── render ───────────────────────────────────────────────────────────────────

function seatEl(p) {
  const g = state.game;
  const me = p === g.human_seat;
  const div = document.createElement('div');
  div.className = 'seat' + (me ? ' me' : '') + (g.current_player === p ? ' current' : '');
  const passed = g.passed[p] ? '<span class="tag passed">PASS</span>' : '';
  const cur = g.current_player === p ? '<span class="tag" style="color:var(--accent)">← 行動中</span>' : '';
  const lastH = [...g.history].reverse().find(h => h.player === p);
  div.innerHTML =
    `<div class="name">${me ? '你' : 'AI'} <span class="dim">P${p}</span>${passed}${cur}</div>` +
    `<div class="cnt">${g.counts[p]} <span class="dim" style="font-size:11px">張</span></div>` +
    `<div class="last">${lastH ? '上手:' + lastH.label : ''}</div>`;
  if (!me && g.all_hands) {
    const peek = document.createElement('div');
    peek.className = 'peek cardrow';
    peek.style.justifyContent = 'flex-start';
    for (const c of g.all_hands[p]) peek.appendChild(cardEl(c, { small: true }));
    div.appendChild(peek);
  }
  return div;
}

function findMatchingLegal() {
  const g = state.game;
  if (!g || !g.legal.length) return null;
  const sel = [...state.selected].sort((a, b) => a - b).join(',');
  if (!sel) return null;
  return g.legal.find(l => l.cards.map(c => c.id).sort((a, b) => a - b).join(',') === sel) || null;
}

function render() {
  const g = state.game;
  if (!g) return;
  $('#session-info').textContent =
    `#${g.id} · ${g.cfg.checkpoint.replace('.pt', '')} · ${g.cfg.mode}` +
    (g.cfg.mode !== 'greedy' ? ` s${g.cfg.sims}` + (g.cfg.mode === 'mcts_det' ? `×d${g.cfg.dets}` : '') : '') +
    (g.seed !== null ? ` · seed=${g.seed}` : '');

  // seats(其他三家)
  const seats = $('#seats');
  seats.innerHTML = '';
  for (let p = 1; p <= 4; p++) if (p !== g.human_seat) seats.appendChild(seatEl(p));

  // center trick
  const myTurn = !g.done && g.current_player === g.human_seat;
  if (g.done) {
    $('#trick-label').textContent = '對局結束';
    $('#trick-cards').innerHTML = '';
  } else if (g.control === 1) {
    $('#trick-label').innerHTML = myTurn
      ? '<b style="color:var(--green)">你有控場權 — 任意出牌</b>' + (g.must_play_club3 ? '(必須含 ♣3)' : '')
      : `P${g.current_player} 控場中…`;
    $('#trick-cards').innerHTML = '';
  } else {
    $('#trick-label').innerHTML = `檯面(P${g.last_play.player})${myTurn ? ' — <b>要壓過:</b>' : ''}`;
    const tc = $('#trick-cards');
    tc.innerHTML = '';
    for (const c of g.last_play.cards) tc.appendChild(cardEl(c));
  }

  // my hand
  $('#my-label').innerHTML = `你的手牌 <span class="dim">P${g.human_seat} · ${g.hand.length} 張</span>` +
    (myTurn ? ' <b style="color:var(--accent)">— 輪到你</b>' : (g.done ? '' : ' <span class="dim">等待中…</span>'));
  const hand = $('#my-hand');
  hand.innerHTML = '';
  for (const c of g.hand) {
    const el = cardEl(c);
    if (myTurn) {
      el.classList.add('clickable');
      if (state.selected.has(c.id)) el.classList.add('selected');
      el.onclick = () => { toggleCard(c.id); };
    }
    hand.appendChild(el);
  }

  // legal actions
  const list = $('#legal-list');
  list.innerHTML = '';
  const match = findMatchingLegal();
  g.legal.forEach((l, i) => {
    if (l.type === 'pass') return;
    const b = document.createElement('div');
    b.className = 'legal' + (match && match.action === l.action ? ' match' : '');
    b.innerHTML = `<span class="idx">${i}</span>${l.label}`;
    b.onclick = () => playAction(l.action);
    list.appendChild(b);
  });

  const passOk = myTurn && g.legal.some(l => l.type === 'pass');
  $('#btn-pass').disabled = !passOk;
  $('#btn-play').disabled = !(myTurn && match);
  const nNonPass = g.legal.filter(l => l.type !== 'pass').length;
  $('#sel-info').textContent = myTurn
    ? (state.selected.size ? (match ? `→ ${match.label}` : '(選取不成合法牌型)') : (nNonPass ? `${nNonPass} 種出法` : '壓不過,只能 PASS'))
    : '';
  $('#btn-undo').disabled = state.busy || !g.n_actions;

  // overlay
  if (g.done && g.rewards) {
    const rows = [1, 2, 3, 4].map(p => {
      const sc = g.rewards[p - 1];
      const cls = sc > 0 ? 'score-pos' : (sc < 0 ? 'score-neg' : '');
      return `<tr><td>${p === g.human_seat ? '你' : 'AI'} P${p}</td><td class="${cls}">${sc > 0 ? '+' : ''}${sc}</td></tr>`;
    }).join('');
    const mine = g.rewards[g.human_seat - 1];
    $('#overlay-box').innerHTML =
      `<h3>${mine > 0 ? '🏆 你贏了' : '你輸了'} <span class="${mine > 0 ? 'score-pos' : 'score-neg'}">${mine > 0 ? '+' : ''}${mine}</span></h3>` +
      `<table>${rows}</table><button class="primary" onclick="newGame()">再來一局 ⏎</button>`;
    $('#overlay').classList.remove('hidden');
    loadStats();
  } else {
    $('#overlay').classList.add('hidden');
  }

  renderLog();
}

function toggleCard(id) {
  if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
  render();
}

// ── actions ──────────────────────────────────────────────────────────────────

function setBusy(b) {
  state.busy = b;
  $('#conn').classList.toggle('ok', !b);
  document.body.style.cursor = b ? 'progress' : '';
}

async function newGame() {
  if (state.busy) return;
  setBusy(true);
  try {
    state.logLines = []; state.selected.clear();
    const g = await post('/api/game', cfgFromForm());
    pushAiMoves(g.ai_moves || []);
    state.game = g;
    render();
    maybeAutoHint();
  } catch (e) { alert('開局失敗:' + e.message); }
  setBusy(false);
}

async function playAction(a) {
  const g = state.game;
  if (state.busy || !g || g.done || g.current_player !== g.human_seat) return;
  setBusy(true);
  try {
    const r = await post(`/api/game/${g.id}/play`, { action: a });
    pushAiMoves(r.ai_moves || []);
    state.selected.clear();
    state.game = r;
    render();
    maybeAutoHint();
  } catch (e) { alert(e.message); }
  setBusy(false);
}

async function undo() {
  const g = state.game;
  if (state.busy || !g || !g.n_actions) return;
  setBusy(true);
  try {
    state.selected.clear();
    state.game = await post(`/api/game/${g.id}/undo`);
    render();
  } catch (e) { alert(e.message); }
  setBusy(false);
}

// ── analysis ─────────────────────────────────────────────────────────────────

async function runAnalysis() {
  const g = state.game;
  if (!g || g.done) return;
  activateTab('analysis');
  const box = $('#tab-analysis');
  box.innerHTML = '<div class="dim pad">computing…</div>';
  try {
    const a = await api(`/api/game/${g.id}/analysis?topk=10`);
    let html = `<h4 class="sect">policy top-k ${a.is_my_turn ? '(你的合法動作)' : '(非你回合,未套 mask)'}</h4>`;
    const maxp = Math.max(...a.top_actions.map(t => t.p), 1e-6);
    for (const t of a.top_actions) {
      html += `<div class="abar"><span class="p">${(t.p * 100).toFixed(1)}%</span>` +
        `<div class="bar" style="width:${(t.p / maxp) * 140}px"></div>` +
        `<span class="lbl" style="cursor:pointer" onclick="playAction(${t.action})">${t.label}</span></div>`;
    }
    html += `<h4 class="sect">value head(各家期望正規化分數,tanh ∈ [-1,1])</h4><div class="vrow">`;
    a.value.forEach((v, i) => {
      const p = i + 1;
      const col = v > 0 ? 'var(--green)' : 'var(--red)';
      html += `<div class="vcell${p === g.human_seat ? ' me' : ''}">P${p}${p === g.human_seat ? '(你)' : ''}<br><b style="color:${col}">${v.toFixed(3)}</b></div>`;
    });
    html += `</div><h4 class="sect">belief head — model 猜各家持牌 P(card)</h4>`;
    html += beliefTable(a.belief);
    box.innerHTML = html;
  } catch (e) { box.innerHTML = `<div class="dim pad">分析失敗:${e.message}</div>`; }
}

const RANK_LABELS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2'];
function beliefTable(belief) {
  let html = '<div class="belief"><table><tr><th></th>';
  for (const r of RANK_LABELS) html += `<th colspan="4">${r}</th>`;
  html += '</tr>';
  for (const [opp, probs] of Object.entries(belief)) {
    html += `<tr><td class="brow-label">P${opp}</td>`;
    probs.forEach((p, ci) => {
      const alpha = Math.min(1, p * 1.6);
      const suit = SUIT_CHAR[['d', 'c', 'h', 's'][ci % 4]];
      const rank = RANK_LABELS[Math.floor(ci / 4)];
      html += `<td><div class="bcell" title="${rank}${suit}: ${(p * 100).toFixed(0)}%" ` +
        `style="background:rgba(88,166,255,${alpha.toFixed(2)})"></div></td>`;
    });
    html += '</tr>';
  }
  return html + '</table><div class="dim" style="margin-top:3px">每列 52 格 = 3♦3♣3♥3♠ 4♦… 順序;越亮 = model 越確定該家有這張。</div></div>';
}

function maybeAutoHint() {
  const g = state.game;
  if ($('#cfg-autohint').checked && g && !g.done && g.current_player === g.human_seat) runAnalysis();
}

// ── stats ────────────────────────────────────────────────────────────────────

async function loadStats() {
  try {
    const s = await api('/api/stats');
    const box = $('#tab-stats');
    if (!s.games) { box.innerHTML = '<div class="dim pad">尚無完賽紀錄。</div>'; return; }
    box.innerHTML =
      `<div class="statgrid">` +
      `<div class="statbox"><div class="v">${s.games}</div><div class="k">完賽局數</div></div>` +
      `<div class="statbox"><div class="v">${(s.human_win_rate * 100).toFixed(0)}%</div><div class="k">你的勝率</div></div>` +
      `<div class="statbox"><div class="v">${s.human_avg_score > 0 ? '+' : ''}${s.human_avg_score}</div><div class="k">你的平均分</div></div>` +
      `<div class="statbox"><div class="v" style="font-size:13px">${s.last10.map(x => (x > 0 ? '+' : '') + x).join(', ')}</div><div class="k">最近 10 局</div></div>` +
      `</div><div class="dim pad" style="font-size:11px">單局方差大;要可信請 ≥100 局(log: web/human_games_web.jsonl)。model 分數 ≈ 你的平均分取負。</div>`;
  } catch (e) { /* server 沒起 stats 就算了 */ }
}

// ── tabs / keyboard / init ───────────────────────────────────────────────────

function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tabpane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'stats') loadStats();
}

document.querySelectorAll('.tab').forEach(t => t.onclick = () => activateTab(t.dataset.tab));
$('#btn-new').onclick = newGame;
$('#btn-undo').onclick = undo;
$('#btn-hint').onclick = runAnalysis;
$('#btn-pass').onclick = () => {
  const g = state.game;
  const pass = g && g.legal.find(l => l.type === 'pass');
  if (pass) playAction(pass.action);
};
$('#btn-play').onclick = () => {
  const m = findMatchingLegal();
  if (m) playAction(m.action);
};

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const g = state.game;
  if (e.key === 'Enter') {
    if (!g || g.done) newGame();
    else { const m = findMatchingLegal(); if (m) playAction(m.action); }
  } else if (e.key === 'p') { $('#btn-pass').click(); }
  else if (e.key === 'u') { undo(); }
  else if (e.key === 'a') { runAnalysis(); }
  else if (/^[0-9]$/.test(e.key) && g && !g.done) {
    const nonpass = g.legal.filter(l => l.type !== 'pass');
    const i = +e.key;
    if (i < nonpass.length) playAction(nonpass[i].action);
  }
});

async function init() {
  try {
    const c = await api('/api/checkpoints');
    const sel = $('#cfg-checkpoint');
    sel.innerHTML = '';
    for (const ck of c.checkpoints) {
      const o = document.createElement('option');
      o.value = ck.name; o.textContent = ck.name.replace('.pt', '');
      if (ck.name === c.default) o.selected = true;
      sel.appendChild(o);
    }
    $('#ckpt-note').textContent = c.checkpoints.length
      ? 'v9* 系列含全資訊 value net;baseline/v6/v8 沒有(該選項自動無效)。'
      : '⚠ engine/checkpoints/saved/ 沒有 .pt 檔!';
    $('#conn').classList.add('ok');
  } catch (e) {
    $('#ckpt-note').textContent = '⚠ 連不上後端:' + e.message;
  }
  newGame();
}
init();
