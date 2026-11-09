"use strict";
/* Chấm Phiếu THPT – ứng dụng điện thoại. Toàn bộ nhận dạng chạy trên máy (engine-worker.js). */
const APP_VERSION = "1.2.0";

// ============================================================ tiện ích
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
}
const fmt = (x) => (Math.round((x || 0) * 100) / 100).toString().replace(".", ",");
const num = (s) => { const v = parseFloat(String(s).replace(",", ".")); return isFinite(v) ? v : null; };
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const normMade = (m) => (m || "").replace(/^0+(?=\d)/, "");
const validP3 = (v) => /^-?\d*,?\d+$/.test(v);
const LETTERS = "ABCD";
const FORMS = {
  default: { label: "Form mặc định", p1: 40, parts: true },
  english: { label: "Form tiếng Anh", p1: 50, parts: false },
};
const formOf = (x) => (x && x.form) || "default";
let toastTimer;
function toast(msg, ms = 2600, onClick) {
  const t = $("#toast");
  t.textContent = msg; t.classList.add("show");
  t.style.pointerEvents = onClick ? "auto" : "none"; t.onclick = onClick || null;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}
const buzz = (p) => { if (settings.vibrate && navigator.vibrate) navigator.vibrate(p); };
const timeStr = (t) => new Date(t).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });

// ============================================================ lưu trữ
const LS = {
  get(k, d) { try { const v = localStorage.getItem("cp_" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { localStorage.setItem("cp_" + k, JSON.stringify(v)); },
};
const DEFAULTS = { p1: 0.25, p1En: 0.25, p3: 0.25, p2: [0, 0.1, 0.25, 0.5, 1], delta: 30, autoKey: true, vibrate: true, continuous: false, form: "default" };
let settings = Object.assign({}, DEFAULTS, LS.get("settings", {}));
let keys = LS.get("keys", []);
let activeKey = LS.get("activeKey", {});          // bộ đáp án mặc định của từng kiểu phiếu
if (typeof activeKey === "string" || activeKey === null) activeKey = activeKey ? { default: activeKey } : {};
const saveSettings = () => LS.set("settings", settings);
const saveKeys = () => { LS.set("keys", keys); LS.set("activeKey", activeKey); };
const formKeys = (form = settings.form) => keys.filter((k) => formOf(k) === form);

const db = {
  _p: null,
  open() {
    return this._p || (this._p = new Promise((ok, bad) => {
      const r = indexedDB.open("cham-phieu", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("results", { keyPath: "id" });
      r.onsuccess = () => ok(r.result); r.onerror = () => bad(r.error);
    }));
  },
  async tx(mode, fn) {
    const d = await this.open();
    return new Promise((ok, bad) => {
      const t = d.transaction("results", mode), s = t.objectStore("results");
      const req = fn(s);
      t.oncomplete = () => ok(req && req.result); t.onerror = () => bad(t.error);
    });
  },
  all() { return this.tx("readonly", (s) => s.getAll()).then((a) => a.sort((x, y) => y.time - x.time)); },
  get(id) { return this.tx("readonly", (s) => s.get(id)); },
  put(r) { return this.tx("readwrite", (s) => s.put(r)); },
  del(id) { return this.tx("readwrite", (s) => s.delete(id)); },
  clear() { return this.tx("readwrite", (s) => s.clear()); },
};

// ============================================================ bộ chấm (Web Worker)
const engine = {
  w: null, ready: false, seq: 0, wait: new Map(), readyWaiters: [],
  start() {
    this.ready = false;
    this.w = new Worker("engine-worker.js");
    this.w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") {
        setChip(m.pct >= 100 ? "ready" : "", m.pct >= 100 ? "Sẵn sàng" : `${m.stage} ${m.pct}%`);
        if (m.pct >= 100) { this.ready = true; this.readyWaiters.splice(0).forEach((f) => f()); }
      } else if (m.type === "fatal") {
        setChip("error", "Lỗi tải bộ chấm – chạm để thử lại");
        this.fatal = m.error;
        for (const [, p] of this.wait) p.bad(new Error("Bộ chấm chưa khởi động được: " + m.error));
        this.wait.clear();
      } else if (m.type === "reply") {
        const p = this.wait.get(m.id); this.wait.delete(m.id);
        if (p) m.ok ? p.ok(m.data) : p.bad(new Error(m.error));
      }
    };
  },
  call(cmd, args, transfer) {
    if (this.fatal) return Promise.reject(new Error("Bộ chấm chưa khởi động được (cần mạng ở lần mở đầu tiên)."));
    const id = ++this.seq;
    return new Promise((ok, bad) => { this.wait.set(id, { ok, bad }); this.w.postMessage({ id, cmd, args }, transfer || []); });
  },
  whenReady() { return this.ready ? Promise.resolve() : new Promise((f) => this.readyWaiters.push(f)); },
  restart() { this.w && this.w.terminate(); this.fatal = null; this.wait.clear(); setChip("", "Đang khởi động…"); this.start(); },
};
function setChip(state, text) {
  const c = $("#engine-chip");
  c.className = "chip engine " + state; c.querySelector("span").textContent = text;
}

// ============================================================ điều hướng + nút Back của Android
const layers = [];
let backResolve = null;
function pushLayer(name, close) { layers.push({ name, close }); history.pushState({ layer: layers.length }, ""); }
function popLayer() { return new Promise((ok) => { backResolve = ok; history.back(); }); }
window.addEventListener("popstate", () => {
  const l = layers.pop();
  if (l) l.close();
  if (backResolve) { const f = backResolve; backResolve = null; f(); }
});
const topLayer = () => layers.length && layers[layers.length - 1].name;

let curTab = "scan";
const TITLES = { scan: "Chấm Phiếu", keys: "Đáp án", history: "Kết quả đã chấm", settings: "Cài đặt", result: "Kết quả" };
function showView(name) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  $("#title").textContent = TITLES[name];
  $("#btn-back").hidden = name !== "result";
  document.body.classList.toggle("in-result", name === "result");   // màn kết quả: ẩn thanh tab cho rộng chỗ
  window.scrollTo(0, 0);
}
function switchTab(tab) {
  if (topLayer() === "result") { popLayer().then(() => switchTab(tab)); return; }
  curTab = tab;
  $$(".tabbar button").forEach((b) => b.classList.toggle("active", b.dataset.view === tab));
  showView(tab);
  if (tab === "history") renderHistory();
  if (tab === "keys") renderKeys();
  if (tab === "scan") renderScan();
  if (tab === "settings") renderSettings();
}
function openResultView() {
  if (topLayer() !== "result") pushLayer("result", () => { showView(curTab); if (curTab === "scan") renderScan(); if (curTab === "history") renderHistory(); });
  showView("result");
}

// ============================================================ đáp án đang dùng
const keyCounts = (k) => ({ p1: Object.keys(k.p1 || {}).length, p2: Object.keys(k.p2 || {}).length, p3: Object.keys(k.p3 || {}).length });
const keyDesc = (k) => {
  const c = keyCounts(k);
  if (!FORMS[formOf(k)].parts) return c.p1 ? `${c.p1} câu` : "Chưa có câu nào";
  return [c.p1 && `P.I: ${c.p1} câu`, c.p2 && `P.II: ${c.p2} câu`, c.p3 && `P.III: ${c.p3} câu`].filter(Boolean).join(" · ") || "Chưa có câu nào";
};
function pickKey(made, form = settings.form) {
  const list = formKeys(form);
  if (!list.length) return null;
  if (settings.autoKey && made && !made.includes("?")) {
    const k = list.find((k) => k.made && normMade(k.made) === normMade(made));
    if (k) return k;
  }
  return list.find((k) => k.id === activeKey[form]) || list[0];
}
const scoreSettings = (form) => ({ p1: form === "english" ? settings.p1En : settings.p1, p2: settings.p2, p3: settings.p3 });
function renderFormSwitch() {
  $$("[data-formsel]").forEach((el) => el.replaceChildren(...Object.entries(FORMS).map(([id, f]) => h("button", {
    class: id === settings.form ? "active" : "",
    onclick: () => {
      if (settings.form === id) return;
      settings.form = id; saveSettings();
      renderFormSwitch(); renderScan();
      if (curTab === "keys") renderKeys();
      toast("Đang chấm: " + f.label);
    },
  }, f.label))));
}

// ============================================================ xử lý ảnh -> chấm
async function prepImage(blob, maxSide = 2600) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), hh = Math.round(bmp.height * s);
  const c = new OffscreenCanvas(w, hh);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, hh);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, hh).data;
  const jpeg = s < 1 || blob.type !== "image/jpeg" ? await c.convertToBlob({ type: "image/jpeg", quality: 0.9 }) : blob;
  return { rgba, w, h: hh, jpeg };
}

const readCache = new Map();          // token -> kết quả đọc (giữ cùng lúc với bộ nhớ của bộ chấm)
let pipeline = Promise.resolve();     // các phiếu được chấm lần lượt
let jobs = 0;                         // số việc chấm đang chạy / đang chờ
function runJob(fn) { jobs++; const p = pipeline.then(fn, fn); pipeline = p.catch(() => {}).finally(() => { jobs--; }); return p; }

/** Đọc + chấm 1 ảnh. Trả {token, read, graded, p3final, image, key, src} hoặc {error, src}. */
async function ensureRead(src, token, form) {
  let read = readCache.get(token);
  let jpeg = src;
  if (!read || !(await engine.call("has", { token }))) {
    const prep = await prepImage(src);
    jpeg = prep.jpeg;
    read = await engine.call("process", { token, rgba: prep.rgba, w: prep.w, h: prep.h, delta: settings.delta, form }, [prep.rgba.buffer]);
    if (!read.error) {
      readCache.set(token, read);
      if (readCache.size > 5) readCache.delete(readCache.keys().next().value);
    }
  }
  return { read, jpeg };
}
function gradeSource(src, { token = uid(), override = {}, form = settings.form } = {}) {
  return runJob(async () => {
    const { read, jpeg } = await ensureRead(src, token, form);
    if (read.error) return { error: read.error, src: jpeg, token };
    const key = pickKey(read.made, form);
    const g = await engine.call("grade", { token, key: key && { made: key.made, p1: key.p1, p2: key.p2, p3: key.p3 }, settings: scoreSettings(form), override });
    return { token, form, read, graded: g.graded, p3final: g.p3final, image: g.image, key, override, src: jpeg };
  });
}

function toRecord(r, old) {
  const g = r.graded, rd = r.read;
  const p3 = rd.part3.map((it) => (r.p3final[it.q] || {}).value || "");
  const lowConf = rd.part3.some((it) => !(it.q in r.override) && it.note);
  return {
    id: r.token, form: r.form, time: old ? old.time : Date.now(), src: r.src,
    sbd: rd.sbd, made: rd.made, keyId: r.key && r.key.id, keyMade: r.key && r.key.made,
    score: g ? g.score : null, max: g ? g.max : null,
    s1: g && g.summary.p1, s2: g && g.summary.p2, s3: g && g.summary.p3,
    warns: [...(g ? g.warnings : []), ...rd.warnings], review: lowConf,
    p1: rd.part1.map((it) => it.answer || (it.multi ? "*" : "-")).join(""),
    p2: rd.part2.map((it) => it.items.map((s) => s.answer || (s.multi ? "*" : "_")).join("")).join(" "),
    p3: p3.join("; "), override: r.override,
  };
}

async function saveRecord(rec) {
  let replaced = null;
  if (rec.sbd && !rec.sbd.includes("?")) {
    const all = await db.all();
    replaced = all.find((o) => o.id !== rec.id && o.sbd === rec.sbd && normMade(o.made) === normMade(rec.made) && formOf(o) === formOf(rec));
    if (replaced) await db.del(replaced.id);
  }
  await db.put(rec);
  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  updateBadge();
  return replaced;
}

// ============================================================ màn KẾT QUẢ
let cur = null;      // phiếu đang xem
const urls = new Set();
function blobUrl(b) { const u = URL.createObjectURL(b); urls.add(u); return u; }
function freeUrls() { urls.forEach((u) => URL.revokeObjectURL(u)); urls.clear(); }

function showLoading(srcBlob, text = "Đang chấm phiếu…") {
  const v = $("#view-result");
  v.replaceChildren(h("div", { class: "card loading" },
    srcBlob ? h("img", { class: "preview-thumb", src: blobUrl(srcBlob) }) : null,
    h("div", { class: "spinner" }),
    h("b", {}, text),
    h("p", { class: "muted small", id: "load-sub" }, engine.ready ? "Thường mất vài giây" : "Đang khởi động bộ chấm (lần đầu cần mạng)…")));
  openResultView();
}

// ---- Nút Tiếp theo / Trước: danh sách phiếu đang duyệt và vị trí phiếu đang xem
const sessionIds = [];                // các phiếu chấm mới trong lần mở ứng dụng này, theo thứ tự chấm
let navCtx = { ids: [], i: -1 };
let navBusy = false;
function setNav(ids, id) { navCtx = { ids: [...ids], i: ids.indexOf(id) }; }
function trackNew(rec, replaced) {
  if (replaced) { const i = sessionIds.indexOf(replaced.id); if (i >= 0) sessionIds.splice(i, 1); }
  if (!sessionIds.includes(rec.id)) sessionIds.push(rec.id);
}
async function goNav(step) {
  if (navBusy) return;
  navBusy = true;
  try {
    let i = navCtx.i + step;
    while (i >= 0 && i < navCtx.ids.length) {
      const r = await db.get(navCtx.ids[i]);
      if (r && r.src) {
        navCtx.i = i;
        await gradeAndShow(r.src, { token: r.id, override: r.override || {}, form: formOf(r), quiet: true });
        return;
      }
      navCtx.ids.splice(i, 1);          // phiếu đã bị xoá / thay thế
      if (step < 0) i += step;
    }
    renderResult();
  } finally { navBusy = false; }
}
/** Đọc sẵn phiếu kế tiếp trong nền để bấm Tiếp theo là hiện ngay. */
async function prefetchNext() {
  const id = navCtx.ids[navCtx.i + 1];
  if (!id || readCache.has(id)) return;
  const r = await db.get(id);
  if (r && r.src) runJob(() => ensureRead(r.src, r.id, formOf(r))).catch(() => {});
}

async function gradeAndShow(blob, opts = {}) {
  const quiet = opts.quiet && topLayer() === "result";    // chuyển phiếu: giữ màn cũ tới khi có kết quả mới
  if (!quiet) { freeUrls(); showLoading(blob); }
  try {
    const r = await gradeSource(blob, opts);
    if (r.error) { if (quiet) freeUrls(); showError(r.error, blob); buzz([60, 60, 60]); return; }
    const old = opts.token ? await db.get(opts.token) : null;
    const rec = toRecord(r, old);
    const replaced = await saveRecord(rec);
    if (replaced) toast(`Đã thay kết quả cũ của SBD ${rec.sbd}`);
    if (!opts.token) { trackNew(rec, replaced); setNav(sessionIds, rec.id); }
    if (quiet) freeUrls();
    cur = { ...r, rec, edits: { ...r.override } };
    renderResult();
    if (quiet) window.scrollTo(0, 0);
    buzz(40);
  } catch (e) {
    showError(e.message, blob);
  }
}

function showError(msg, blob) {
  $("#view-result").replaceChildren(h("div", { class: "card errbox" },
    h("div", { class: "eico" }, "!"),
    h("h3", {}, "Chưa đọc được phiếu"),
    h("p", {}, msg),
    blob ? h("img", { class: "preview-thumb", src: blobUrl(blob), style: "width:180px" }) : null,
    h("p", { class: "muted small" }, "Hãy chụp lại: đủ sáng, thẳng từ trên xuống, lấy trọn cả tờ phiếu."),
    h("div", { class: "row2", style: "margin-top:16px" },
      h("button", { class: "btn soft", onclick: () => popLayer() }, "Quay lại"),
      h("button", { class: "btn primary", onclick: () => popLayer().then(startCapture) }, "Chụp lại"))));
}

function ratioClass(s, m) { if (s == null || !m) return "mid"; const r = s / m; return r >= 0.8 ? "hi" : r < 0.5 ? "lo" : "mid"; }

function renderResult() {
  const { read: rd, graded: g, key } = cur;
  const v = $("#view-result");
  const kids = [];

  // --- điểm tổng
  const hero = h("div", { class: "card hero " + (g ? ratioClass(g.score, g.max) : "") });
  if (g) {
    hero.append(h("div", { class: "big" }, fmt(g.score), h("small", {}, " / " + fmt(g.max))));
    if (Math.abs(g.max - 10) > 1e-9 && g.max > 0) hero.append(h("div", { class: "sub" }, `≈ ${fmt(g.score / g.max * 10)} / 10 điểm`));
  } else {
    hero.append(h("div", { class: "big", style: "font-size:30px" }, "Chưa có đáp án"),
      h("div", { class: "sub" }, "Đã đọc bài làm. Thêm đáp án để chấm điểm."),
      h("button", { class: "btn primary", style: "margin-top:12px", onclick: () => openKeyEditor(null, { made: rd.made && !rd.made.includes("?") ? rd.made : "", form: cur.form }) }, "+ Nhập đáp án"));
  }
  const madeMismatch = g && key && key.made && rd.made && normMade(key.made) !== normMade(rd.made);
  hero.append(h("div", { class: "idrow" },
    h("span", { class: "tag" + (rd.sbd.includes("?") ? " bad" : "") }, "SBD ", h("b", {}, rd.sbd)),
    h("span", { class: "tag" + (rd.made.includes("?") || madeMismatch ? " bad" : "") }, "Mã đề ", h("b", {}, rd.made)),
    key ? h("span", { class: "tag" }, "Đáp án: ", h("b", {}, key.made ? "mã " + key.made : "mặc định")) : null,
    FORMS[cur.form].parts ? null : h("span", { class: "tag" }, "Form tiếng Anh")));
  if (g && !FORMS[cur.form].parts) {
    const s = g.summary;
    hero.append(h("div", { class: "parts one" },
      h("div", {}, h("small", {}, "Số câu đúng"), h("b", {}, `${s.p1[0]}/${s.p1[1]}`), h("span", {}, fmt(s.p1[2]) + " điểm"))));
  } else if (g) {
    const s = g.summary;
    hero.append(h("div", { class: "parts" },
      h("div", {}, h("small", {}, "Phần I"), h("b", {}, `${s.p1[0]}/${s.p1[1]}`), h("span", {}, fmt(s.p1[2]) + " điểm")),
      h("div", {}, h("small", {}, "Phần II (ý)"), h("b", {}, `${s.p2[0]}/${s.p2[1]}`), h("span", {}, fmt(s.p2[2]) + " điểm")),
      h("div", {}, h("small", {}, "Phần III"), h("b", {}, `${s.p3[0]}/${s.p3[1]}`), h("span", {}, fmt(s.p3[2]) + " điểm"))));
  }
  kids.push(hero);

  // --- cảnh báo
  const warns = [...(g ? g.warnings : []), ...rd.warnings];
  if (warns.length) {   // chỉ hiện 3 dòng đầu, còn lại thu gọn cho đỡ phải cuộn
    const li = (w) => h("li", {}, w);
    kids.push(h("div", { class: "card warns" }, h("h3", {}, `⚠ Cần kiểm tra (${warns.length})`),
      h("ul", {}, warns.slice(0, 3).map(li)),
      warns.length > 3 ? h("details", {}, h("summary", {}, `Xem thêm ${warns.length - 3} lưu ý`), h("ul", {}, warns.slice(3).map(li))) : null));
  }

  // --- Phần III: bài tô ô tròn (ảnh chữ viết chỉ để đối chiếu), sửa được
  if (rd.part3.length) {
    const card = h("div", { class: "card" }, h("h3", {}, "🔢 Phần III – bài tô"),
      h("p", { class: "muted small", style: "margin:-6px 0 4px" }, "Đọc theo ô tròn đã tô. Cần thì sửa vào ô, điểm tự tính lại."));
    const g3 = g ? Object.fromEntries(g.p3.map((x) => [x.q, x])) : {};
    rd.part3.forEach((it, i) => {
      const fin = cur.p3final[it.q] || { value: it.value };
      const gi = g3[it.q];
      const edited = it.q in cur.edits;
      const input = h("input", {
        type: "text", inputmode: "decimal", value: fin.value, placeholder: "(trống)", autocomplete: "off",
        class: (edited ? "edited" : "") + (fin.value && !validP3(fin.value) ? " invalid" : ""),
        onchange: (e) => editP3(it.q, e.target.value),
        onkeydown: (e) => { if (e.key === "Enter") e.target.blur(); },
      });
      const mark = gi ? h("span", { class: "mark " + (gi.ok ? "ok" : "bad") }, gi.ok ? "✓" : "✗") : h("span", { class: "mark na" }, "–");
      const note = edited ? h("div", { class: "note edited" }, `✏️ Đã sửa (máy đọc: ${it.value || "trống"})`)
        : it.note ? h("div", { class: "note" }, "⚠ " + it.note + " – kiểm tra lại") : null;
      card.append(h("div", { class: "p3row" },
        h("div", { class: "qn" }, "Câu " + it.q, gi ? h("small", {}, "ĐA " + gi.key) : null),
        h("img", { src: blobUrl(rd.crops[i]), alt: "Chữ viết câu " + it.q }),
        input, mark, note));
    });
    kids.push(card);
  }

  // --- ảnh kết quả
  const imgUrl = blobUrl(cur.image);
  kids.push(h("div", { class: "card shot" },
    h("img", { src: imgUrl, alt: "Ảnh phiếu đã chấm", onclick: () => openViewer(imgUrl) }),
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: "border-color:#28aa28" }), "Đáp án đúng"),
      h("span", {}, h("i", { style: "border-color:#dc2828" }), "Tô sai"),
      h("span", {}, h("i", { style: "border-color:#ff8c00" }), "Bỏ trống"),
      h("span", {}, h("i", { style: "border-color:#1e5adc" }), "Không chấm")),
    h("div", { class: "mini-actions" },
      h("button", { class: "btn soft", onclick: () => openViewer(imgUrl) }, "🔍 Phóng to"),
      h("button", { class: "btn soft", onclick: shareImage }, "📤 Gửi ảnh"))));

  // --- chi tiết
  if (g) kids.push(detailP1(g), detailP2(g), detailP3(g));
  else kids.push(readOnlyDetail(rd));

  kids.push(h("button", { class: "btn soft danger wide", style: "margin-top:4px", onclick: deleteCurrent }, "Xoá kết quả này"));
  // Thanh dưới: ‹ phiếu trước · Xong · 📷 chụp thêm · Tiếp theo › (hết danh sách thì nút chính là chấm bài mới)
  const total = navCtx.ids.length, hasPrev = navCtx.i > 0, hasNext = navCtx.i >= 0 && navCtx.i < total - 1;
  const capture = () => popLayer().then(startCapture);
  $("#title").textContent = total > 1 ? `Kết quả · Tờ ${navCtx.i + 1}/${total}` : TITLES.result;
  kids.push(h("div", { class: "actions nav" + (hasNext ? "" : " last") },
    h("button", { class: "btn soft arrow", "aria-label": "Phiếu trước", disabled: !hasPrev, onclick: () => goNav(-1) }, "‹"),
    h("button", { class: "btn soft", onclick: () => popLayer() }, "Xong"),
    hasNext ? h("button", { class: "btn soft", "aria-label": "Chấm bài mới", onclick: capture }, "📷") : null,
    hasNext ? h("button", { class: "btn primary", id: "btn-next", onclick: () => goNav(1) }, "Tiếp theo ›")
      : h("button", { class: "btn primary", onclick: capture }, "📷 Chấm bài tiếp")));
  v.replaceChildren(...kids.filter(Boolean));
  prefetchNext();
}

function detailP1(g) {
  const s = g.summary.p1;
  return h("details", { class: "card acc" }, h("summary", {}, FORMS[cur.form].parts ? "Phần I" : "Chi tiết từng câu", h("small", {}, `${s[0]}/${s[1]} câu đúng`)),
    h("div", { class: "grid1" }, g.p1.map((x) => {
      const cls = x.ok ? "ok" : x.student ? "bad" : "blank";
      return h("div", { class: "q1 " + cls }, h("small", {}, x.q), h("b", {}, x.student ? x.student.replace(" (nhiều)", "*") : "–"),
        x.ok ? null : h("s", {}, x.key));
    })));
}
function detailP2(g) {
  if (!g.p2.length) return null;
  const s = g.summary.p2;
  return h("details", { class: "card acc" }, h("summary", {}, "Phần II", h("small", {}, `${s[0]}/${s[1]} ý đúng`)),
    h("table", { class: "t2" }, h("tr", {}, h("th", {}, "Câu"), ..."abcd".split("").map((c) => h("th", {}, c)), h("th", {}, "Điểm")),
      g.p2.map((x) => h("tr", {}, h("td", {}, h("b", {}, x.q)),
        ...[0, 1, 2, 3].map((i) => {
          const st = (x.student || "____")[i];
          const cls = x.ok_items[i] ? "ok" : st === "_" ? "blank" : "bad";
          return h("td", { class: cls }, st === "_" ? "–" : st, x.ok_items[i] ? null : h("small", {}, "ĐA " + x.key[i]));
        }),
        h("td", {}, fmt(x.points))))));
}
function detailP3(g) {
  if (!g.p3.length) return null;
  const s = g.summary.p3;
  return h("details", { class: "card acc" }, h("summary", {}, "Phần III", h("small", {}, `${s[0]}/${s[1]} câu đúng`)),
    h("table", { class: "t2" }, h("tr", {}, h("th", {}, "Câu"), h("th", {}, "Bài làm"), h("th", {}, "Đáp án"), h("th", {}, "Điểm")),
      g.p3.map((x) => h("tr", {}, h("td", {}, h("b", {}, x.q)),
        h("td", { class: x.ok ? "ok" : x.student ? "bad" : "blank" }, x.student || "–", x.edited ? h("small", {}, "đã sửa") : null),
        h("td", {}, x.key), h("td", {}, fmt(x.points))))));
}
function readOnlyDetail(rd) {
  return h("details", { class: "card acc", open: true }, h("summary", {}, "Bài làm đọc được"),
    h("p", { class: "small" }, h("b", {}, "Phần I: "), rd.part1.map((it) => `${it.q}${it.answer || (it.multi ? "*" : "-")}`).join(" ")),
    rd.part2.length ? h("p", { class: "small" }, h("b", {}, "Phần II: "), rd.part2.map((it) => `${it.q}:${it.items.map((s) => s.answer || "_").join("")}`).join("  ")) : null,
    rd.part3.length ? h("p", { class: "small" }, h("b", {}, "Phần III: "), rd.part3.map((it) => `${it.q}: ${(cur.p3final[it.q] || {}).value || "–"}`).join("  ")) : null);
}

async function editP3(q, value) {
  const v = value.trim().replace(/\./g, ",").replace(/\s/g, "");
  const it = cur.read.part3.find((x) => x.q === q);
  if (v === it.value) delete cur.edits[q]; else cur.edits[q] = v;
  await regradeCurrent();
}
async function regradeCurrent(msg) {
  const y = window.scrollY;
  try {
    const me = cur;
    const r = await gradeSource(cur.src, { token: cur.token, override: { ...cur.edits }, form: cur.form });
    if (r.error) { toast(r.error); return; }
    const rec = toRecord(r, me.rec);
    await saveRecord(rec);
    if (cur !== me) return;             // đã chuyển sang phiếu khác trong lúc chấm lại
    freeUrls();
    cur = { ...r, rec, edits: { ...r.override } };
    renderResult();
    window.scrollTo(0, y);
    if (msg) toast(msg);
  } catch (e) { toast(e.message); }
}
async function shareImage() {
  const name = `phieu-${cur.read.sbd}.jpg`.replace(/\?/g, "x");
  const file = new File([cur.image], name, { type: "image/jpeg" });
  const score = cur.graded ? ` – ${fmt(cur.graded.score)}/${fmt(cur.graded.max)} điểm` : "";
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    navigator.share({ files: [file], title: `SBD ${cur.read.sbd}${score}`, text: `SBD ${cur.read.sbd} · Mã đề ${cur.read.made}${score}` }).catch(() => {});
  } else download(file, name);
}
async function deleteCurrent() {
  if (!confirm(`Xoá kết quả của SBD ${cur.read.sbd}?`)) return;
  await db.del(cur.rec.id);
  updateBadge();
  await popLayer();
  toast("Đã xoá");
}
function download(blob, name) {
  const a = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

// ============================================================ CAMERA
const cam = { stream: null, track: null, ic: null, torch: false, count: 0, lock: null, busy: false };
function startCapture() {
  if (settings.continuous || (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.isSecureContext)) openCamera();
  else $("#file-native").click();
}
async function openCamera() {
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) || !window.isSecureContext) {
    toast("Trình duyệt chưa cho mở camera trực tiếp – dùng camera của máy");
    $("#file-native").click(); return;
  }
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 } },
    });
  } catch (e) {
    toast("Không mở được camera (" + (e.name === "NotAllowedError" ? "chưa cấp quyền" : e.name) + ") – dùng camera của máy", 3500);
    $("#file-native").click(); return;
  }
  const ov = $("#camera");
  ov.hidden = false; document.body.classList.add("overlay-open");
  pushLayer("camera", closeCamera);
  const vid = $("#cam-video");
  vid.srcObject = cam.stream;
  vid.onloadedmetadata = vid.onresize = positionGuide;
  vid.play().catch(() => {});
  cam.track = cam.stream.getVideoTracks()[0];
  const caps = cam.track.getCapabilities ? cam.track.getCapabilities() : {};
  $("#cam-torch").hidden = !caps.torch;
  $("#cam-torch").classList.toggle("on", false); cam.torch = false;
  if (caps.focusMode && caps.focusMode.includes("continuous"))
    cam.track.applyConstraints({ advanced: [{ focusMode: "continuous" }] }).catch(() => {});
  cam.ic = "ImageCapture" in window ? new ImageCapture(cam.track) : null;
  cam.count = 0; $("#cam-count").textContent = ""; $("#cam-results").replaceChildren();
  $("#cam-hint").textContent = `${FORMS[settings.form].label}${settings.continuous ? " · chụp liên tục" : ""} – đặt trọn tờ phiếu vào khung`;
  try { cam.lock = await navigator.wakeLock.request("screen"); } catch { cam.lock = null; }
}
/** Khung ngắm khổ A4 nằm trọn trong vùng hình thật của camera (thấy gì chụp nấy). */
function positionGuide() {
  const v = $("#cam-video"), g = $(".cam-guide");
  if (!v.videoWidth) return;
  const W = innerWidth, H = innerHeight;
  const s = Math.min(W / v.videoWidth, H / v.videoHeight);
  const dw = v.videoWidth * s, dh = v.videoHeight * s, dx = (W - dw) / 2, dy = (H - dh) / 2;
  const top = Math.max(dy, 64), bottom = Math.min(dy + dh, H - 130);
  const availW = dw * 0.94, availH = (bottom - top) * 0.96;
  const gw = Math.min(availW, availH / 1.414), gh = gw * 1.414;
  Object.assign(g.style, { left: (W - gw) / 2 + "px", top: top + (bottom - top - gh) / 2 + "px", width: gw + "px", height: gh + "px" });
}
function closeCamera() {
  if (cam.stream) cam.stream.getTracks().forEach((t) => t.stop());
  cam.stream = cam.track = cam.ic = null;
  if (cam.lock) { cam.lock.release().catch(() => {}); cam.lock = null; }
  $("#camera").hidden = true; document.body.classList.remove("overlay-open");
  if (curTab === "scan" && topLayer() !== "result") renderScan();
}
async function snapPhoto() {
  let blob = null;
  if (cam.ic) {
    try {
      const pc = await cam.ic.getPhotoCapabilities();
      const opt = {};
      if (pc.imageWidth && pc.imageWidth.max) opt.imageWidth = Math.min(pc.imageWidth.max, 4032);
      blob = await cam.ic.takePhoto(opt);
      if (cam.torch) cam.track.applyConstraints({ advanced: [{ torch: true }] }).catch(() => {});
    } catch { blob = null; }
  }
  if (!blob) {
    const vid = $("#cam-video");
    const c = new OffscreenCanvas(vid.videoWidth, vid.videoHeight);
    c.getContext("2d").drawImage(vid, 0, 0);
    blob = await c.convertToBlob({ type: "image/jpeg", quality: 0.93 });
  }
  return blob;
}
async function onShutter() {
  if (cam.busy || !cam.stream) return;
  cam.busy = true; $("#cam-shutter").classList.add("busy");
  const fl = $("#cam-flash"); fl.classList.add("on"); setTimeout(() => fl.classList.remove("on"), 60);
  buzz(15);
  let blob;
  try { blob = await snapPhoto(); } catch (e) { toast("Lỗi chụp: " + e.message); }
  cam.busy = false; $("#cam-shutter").classList.remove("busy");
  if (!blob) return;
  if (!settings.continuous) { await popLayer(); gradeAndShow(blob); return; }
  cam.count++; $("#cam-count").textContent = cam.count + " tờ";
  const chip = h("div", { class: "cam-res" }, h("b", { class: "sc" }, "…"), h("span", {}, `Tờ ${cam.count}: đang chấm…`));
  const list = $("#cam-results");
  list.prepend(chip);
  while (list.children.length > 3) list.lastChild.remove();
  gradeBatchItem(blob).then((res) => {
    if (res.error) { chip.classList.add("err"); chip.firstChild.textContent = "✗"; chip.children[1].textContent = res.error.slice(0, 90) + " – chụp lại"; buzz([60, 60, 60]); return; }
    const rec = res.rec;
    chip.firstChild.textContent = rec.score == null ? "✓" : fmt(rec.score);
    chip.children[1].textContent = `SBD ${rec.sbd} · Mã ${rec.made}${rec.review || rec.warns.length ? " · ⚠ cần xem" : ""}`;
    chip.onclick = () => popLayer().then(() => openRecord(rec.id, sessionIds));
    buzz(40);
  });
}
async function gradeBatchItem(blob) {
  try {
    const r = await gradeSource(blob);
    if (r.error) return r;
    const rec = toRecord(r, null);
    trackNew(rec, await saveRecord(rec));
    return { rec };
  } catch (e) { return { error: e.message }; }
}

// ============================================================ XEM ẢNH phóng to (chụm 2 ngón)
const vw = { s: 1, x: 0, y: 0, min: 1, pts: new Map(), base: null, lastTap: 0 };
function openViewer(url) {
  const ov = $("#viewer"), img = $("#viewer-img");
  ov.hidden = false; document.body.classList.add("overlay-open");
  pushLayer("viewer", () => { ov.hidden = true; document.body.classList.remove("overlay-open"); });
  img.onload = () => {
    vw.min = vw.s = Math.min(innerWidth / img.naturalWidth, innerHeight / img.naturalHeight);
    vw.x = (innerWidth - img.naturalWidth * vw.s) / 2; vw.y = (innerHeight - img.naturalHeight * vw.s) / 2;
    applyView();
  };
  img.src = url;
}
function applyView() { $("#viewer-img").style.transform = `translate(${vw.x}px,${vw.y}px) scale(${vw.s})`; }
function zoomAt(px, py, ns) {
  ns = Math.max(vw.min, Math.min(vw.min * 10, ns));
  vw.x = px - (px - vw.x) * ns / vw.s; vw.y = py - (py - vw.y) * ns / vw.s; vw.s = ns;
  if (ns === vw.min) { const img = $("#viewer-img"); vw.x = (innerWidth - img.naturalWidth * ns) / 2; vw.y = (innerHeight - img.naturalHeight * ns) / 2; }
  applyView();
}
function viewerBase() {
  const p = [...vw.pts.values()];
  if (p.length >= 2) {
    vw.base = { two: true, d: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), mx: (p[0].x + p[1].x) / 2, my: (p[0].y + p[1].y) / 2, s: vw.s, x: vw.x, y: vw.y };
  } else if (p.length === 1) vw.base = { two: false, px: p[0].x, py: p[0].y, x: vw.x, y: vw.y };
  else vw.base = null;
}
function initViewer() {
  const ov = $("#viewer");
  ov.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".viewer-close")) return;
    ov.setPointerCapture(e.pointerId);
    vw.pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    viewerBase();
    if (vw.pts.size === 1) {
      const now = Date.now();
      if (now - vw.lastTap < 300) zoomAt(e.clientX, e.clientY, vw.s > vw.min * 1.5 ? vw.min : vw.min * 3);
      vw.lastTap = now;
    }
  });
  ov.addEventListener("pointermove", (e) => {
    if (!vw.pts.has(e.pointerId) || !vw.base) return;
    vw.pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const p = [...vw.pts.values()], b = vw.base;
    if (b.two && p.length >= 2) {
      const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;
      const ns = Math.max(vw.min, Math.min(vw.min * 10, b.s * d / b.d));
      vw.x = mx - (b.mx - b.x) * ns / b.s; vw.y = my - (b.my - b.y) * ns / b.s; vw.s = ns;
    } else if (!b.two) { vw.x = b.x + e.clientX - b.px; vw.y = b.y + e.clientY - b.py; }
    applyView();
  });
  const up = (e) => { vw.pts.delete(e.pointerId); viewerBase(); };
  ov.addEventListener("pointerup", up); ov.addEventListener("pointercancel", up);
  $("#viewer-close").onclick = () => popLayer();
}

// ============================================================ màn CHẤM
function renderScan() {
  const b = $("#key-banner");
  const fkeys = formKeys();
  b.className = "card key-banner" + (fkeys.length ? "" : " none");
  const ico = h("div", { class: "kb-ico", html: fkeys.length ? '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M12 8v5m0 3v.5"/><circle cx="12" cy="12" r="9"/></svg>' });
  let txt;
  if (!fkeys.length) txt = h("div", { class: "kb-txt" }, h("b", {}, "Chưa có đáp án"), h("small", {}, "Vẫn đọc được bài làm; thêm đáp án để chấm điểm"));
  else if (settings.autoKey && fkeys.length > 1) txt = h("div", { class: "kb-txt" }, h("b", {}, `${fkeys.length} bộ đáp án · tự chọn theo mã đề`), h("small", {}, fkeys.map((k) => k.made || "mặc định").join(", ")));
  else { const k = pickKey(null); txt = h("div", { class: "kb-txt" }, h("b", {}, "Đáp án: " + (k.made ? "mã đề " + k.made : "mặc định")), h("small", {}, keyDesc(k))); }
  b.replaceChildren(ico, txt, h("button", { class: "btn soft sm", style: "background:var(--primary-soft)", onclick: () => fkeys.length ? switchTab("keys") : openKeyEditor(null) }, fkeys.length ? "Đổi" : "+ Thêm"));
  $("#sw-continuous").checked = settings.continuous;
  db.all().then((all) => {
    $("#recent-title").hidden = !all.length;
    const ids = all.map((r) => r.id);
    $("#recent").replaceChildren(...all.slice(0, 4).map((r) => histItem(r, ids)));
  });
}

// ============================================================ màn KẾT QUẢ ĐÃ CHẤM
function histItem(r, ids) {
  const sc = r.score == null ? h("div", { class: "score na" }, "chưa\nchấm") : h("div", { class: "score " + ratioClass(r.score, r.max) }, fmt(r.score));
  const flag = r.review || (r.warns && r.warns.length) ? h("span", { class: "flag" }, "⚠ xem lại") : null;
  return h("button", { class: "item", onclick: () => openRecord(r.id, ids) }, sc,
    h("div", { class: "meta" }, h("b", {}, "SBD " + r.sbd), h("small", {}, `Mã đề ${r.made} · ${timeStr(r.time)}${r.max != null ? " · /" + fmt(r.max) : ""}${formOf(r) === "english" ? " · Tiếng Anh" : ""}`)), flag);
}
async function renderHistory() {
  const all = await db.all();
  const q = $("#hist-search").value.trim();
  const list = q ? all.filter((r) => r.sbd.includes(q)) : all;
  const graded = all.filter((r) => r.score != null);
  const avg = graded.length ? graded.reduce((a, r) => a + r.score / r.max * 10, 0) / graded.length : null;
  $("#hist-stats").replaceChildren(
    h("div", {}, h("b", {}, all.length), h("small", {}, "phiếu đã chấm")),
    h("div", {}, h("b", {}, avg == null ? "–" : fmt(avg)), h("small", {}, "điểm TB (thang 10)")),
    h("div", {}, h("b", {}, all.filter((r) => r.review || (r.warns && r.warns.length)).length), h("small", {}, "cần xem lại")));
  $("#hist-list").replaceChildren(...(list.length ? list.map((r) => histItem(r, list.map((x) => x.id))) : [h("div", { class: "empty", html: '<svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg><p>' + (q ? "Không có SBD phù hợp" : "Chưa có phiếu nào. Vào tab <b>Chấm</b> để bắt đầu.") + "</p>" })]));
  ["btn-export", "btn-regrade-all", "btn-clear"].forEach((id) => ($("#" + id).disabled = !all.length));
}
async function updateBadge() {
  const all = await db.all();
  const n = all.filter((r) => r.review || (r.warns && r.warns.length)).length;
  const b = $("#hist-badge"); b.hidden = !n; b.textContent = n;
}
async function openRecord(id, ids = [id]) {
  const r = await db.get(id);
  if (!r) return;
  if (!r.src) { toast("Phiếu này không còn ảnh gốc"); return; }
  setNav(ids, id);
  gradeAndShow(r.src, { token: r.id, override: r.override || {}, form: formOf(r) });
}
async function exportExcel() {
  const all = (await db.all()).sort((a, b) => a.sbd.localeCompare(b.sbd) || a.time - b.time);
  const rows = [["STT", "SBD", "Mã đề", "Tổng điểm", "Thang 10", "P.I câu đúng", "Điểm P.I", "P.II ý đúng", "Điểm P.II", "P.III câu đúng", "Điểm P.III",
    "Bài làm Phần I", "Bài làm Phần II", "Bài làm Phần III", "Ghi chú", "Thời gian chấm", "Kiểu phiếu"]];
  const frac = (s) => (s && s[1] ? `${s[0]}/${s[1]}` : ""), pts = (s) => (s && s[1] ? s[2] : null);
  all.forEach((r, i) => rows.push([i + 1, r.sbd, r.made, r.score, r.max ? Math.round(r.score / r.max * 1000) / 100 : null,
    frac(r.s1), pts(r.s1), frac(r.s2), pts(r.s2), frac(r.s3), pts(r.s3), r.p1, r.p2, r.p3,
    [...(r.warns || []), r.review ? "Phần III cần kiểm tra bài tô" : ""].filter(Boolean).join(" | "), new Date(r.time).toLocaleString("vi-VN"), FORMS[formOf(r)].label]));
  const blob = makeXlsx(rows, [5, 9, 8, 10, 9, 11, 9, 11, 9, 12, 10, 44, 40, 30, 40, 18, 15]);
  const d = new Date();
  download(blob, `ket-qua-cham-${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}-${d.getHours()}h${String(d.getMinutes()).padStart(2, "0")}.xlsx`);
  toast(`Đã xuất ${all.length} phiếu ra file Excel (thư mục Tải xuống)`, 3500);
}
async function regradeRecords(filter, label) {
  const all = (await db.all()).filter(filter);
  let n = 0, err = 0;
  for (const r of all) {
    toast(`${label} ${++n}/${all.length}…`, 60000);
    try {
      const res = await gradeSource(r.src, { token: r.id, override: r.override || {}, form: formOf(r) });
      if (res.error) { err++; continue; }
      await db.put(toRecord(res, r));
    } catch { err++; }
  }
  toast(`Đã chấm lại ${all.length - err} phiếu` + (err ? `, ${err} phiếu lỗi` : ""), 3000);
  updateBadge();
  if (curTab === "history") renderHistory();
}
async function handleFiles(files) {
  files = [...files].filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|heic|webp)$/i.test(f.name));
  if (!files.length) return;
  if (topLayer() === "camera") await popLayer();
  if (files.length === 1) { gradeAndShow(files[0]); return; }
  if (topLayer() === "result") await popLayer();
  switchTab("history");
  let ok = 0, bad = 0;
  const batch = [];
  for (let i = 0; i < files.length; i++) {
    toast(`Đang chấm ${i + 1}/${files.length}…`, 60000);
    const r = await gradeBatchItem(files[i]);
    r.error ? bad++ : ok++;
    if (r.rec) batch.push(r.rec.id);
    if (curTab === "history" && !topLayer()) renderHistory();
  }
  buzz(60);
  toast(`Xong: ${ok} phiếu đã chấm` + (bad ? `, ${bad} ảnh không đọc được` : "") + (batch.length ? " – chạm vào đây để xem từng tờ" : ""),
    8000, batch.length ? () => { $("#toast").classList.remove("show"); openRecord(batch[0], batch); } : undefined);
}

// ============================================================ ĐÁP ÁN
function renderKeys() {
  const list = $("#key-list");
  const fkeys = formKeys();
  if (!fkeys.length) {
    list.replaceChildren(h("div", { class: "empty", html: '<svg viewBox="0 0 24 24"><path d="M9 11l3 3 8-8"/><path d="M20 12v7H4V5h11"/></svg><p>Chưa có bộ đáp án nào</p>' }));
    return;
  }
  const using = pickKey(null);
  list.replaceChildren(...fkeys.map((k) => h("div", { class: "card keycard", onclick: () => openKeyEditor(k) },
    h("div", { class: "kc-made" }, h("small", {}, "MÃ ĐỀ"), k.made || "–"),
    h("div", { class: "kc-info" }, h("b", {}, k.made ? "Mã đề " + k.made : "Đáp án mặc định"), h("small", {}, keyDesc(k)), h("small", {}, "Sửa " + timeStr(k.updated))),
    h("button", {
      class: "use" + (k.id === using.id ? " on" : ""), onclick: (e) => {
        e.stopPropagation(); activeKey[settings.form] = k.id; saveKeys(); renderKeys();
        toast(settings.autoKey ? "Dùng khi mã đề trên phiếu không khớp bộ nào" : "Đã chọn bộ đáp án này");
      },
    }, k.id === using.id ? "✓ Mặc định" : "Đặt mặc định"))));
}

let draft = null, keTab = "p1", k1cur = 1;
const KEY_SAMPLE = "MÃ ĐỀ: 101\n\nPHẦN I\n1D 2B 3A 4C 5B 6A 7C 8B 9D 10B\n11C 12B\n\nPHẦN II\n1: ĐSĐS\n2: SSĐĐ\n\nPHẦN III\n1: 12,5\n2: -3\n";
function openKeyEditor(k, preset = {}) {
  draft = k ? JSON.parse(JSON.stringify(k)) : { id: null, made: preset.made || "", p1: {}, p2: {}, p3: {} };
  draft.form = k ? formOf(k) : preset.form || settings.form;
  const full = FORMS[draft.form].parts;
  $("#ke-tabs").classList.toggle("two", !full);
  $$("#ke-tabs button").forEach((b) => { b.hidden = !full && (b.dataset.tab === "p2" || b.dataset.tab === "p3"); });
  $("#ke-text").value = "";
  $("#ke-title").textContent = (k ? "Sửa đáp án" : "Thêm đáp án") + (full ? "" : " · Tiếng Anh");
  $("#ke-made").value = draft.made || "";
  $("#ke-delete").hidden = !k;
  $("#ke-err").textContent = "";
  k1cur = 1;
  const ov = $("#key-editor");
  ov.hidden = false; ov.scrollTop = 0; document.body.classList.add("overlay-open");
  pushLayer("keys", () => { ov.hidden = true; document.body.classList.remove("overlay-open"); });
  setKeTab(preset.text != null ? "txt" : "p1");
  if (preset.text != null) { $("#ke-text").value = preset.text; parseKeyText(); }
  renderKeyPanes();
}
function setKeTab(t) {
  keTab = t;
  $$("#ke-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  $$(".ke-pane").forEach((p) => p.classList.toggle("active", p.id === "ke-" + t));
  if (t === "txt" && !$("#ke-text").value.trim()) $("#ke-text").value = keyCounts(draft).p1 + keyCounts(draft).p2 + keyCounts(draft).p3 ? keyToText(draft) : "";
  if (t === "txt") $("#ke-text").placeholder = FORMS[draft.form].parts ? KEY_SAMPLE : KEY_SAMPLE_EN;
}
const KEY_SAMPLE_EN = "MÃ ĐỀ: 101\n\n1D 2B 3A 4C 5B 6A 7C 8B 9D 10B\n11C 12B\n";
function keyToText(k) {
  const lines = [];
  if (k.made) lines.push("MÃ ĐỀ: " + k.made, "");
  const p1 = Object.entries(k.p1).sort((a, b) => a[0] - b[0]).map(([q, a]) => q + a);
  if (p1.length) { if (FORMS[formOf(k)].parts) lines.push("PHẦN I"); for (let i = 0; i < p1.length; i += 10) lines.push(p1.slice(i, i + 10).join(" ")); lines.push(""); }
  const p2 = Object.entries(k.p2).sort((a, b) => a[0] - b[0]);
  if (p2.length) { lines.push("PHẦN II", ...p2.map(([q, a]) => `${q}: ${a}`), ""); }
  const p3 = Object.entries(k.p3).sort((a, b) => a[0] - b[0]);
  if (p3.length) lines.push("PHẦN III", ...p3.map(([q, a]) => `${q}: ${a}`));
  return lines.join("\n");
}
function renderKeyPanes() {
  const c = keyCounts(draft);
  const n1 = FORMS[draft.form].p1, half = Math.ceil(n1 / 2);
  $("#ke-c1").textContent = `${c.p1}/${n1}`; $("#ke-c2").textContent = c.p2 + "/8"; $("#ke-c3").textContent = c.p3 + "/6";
  // Phần I: lưới câu x A B C D – chạm là chọn và tự nhảy sang câu sau
  const rows = [];
  for (let q = 1; q <= n1; q++) {
    rows.push(h("div", { class: "k1row" + (q === k1cur ? " cur" : "") }, h("span", { class: "n" }, q),
      ...LETTERS.split("").map((L) => h("button", {
        class: draft.p1[q] === L ? "on" : "", onclick: () => {
          if (draft.p1[q] === L) delete draft.p1[q]; else { draft.p1[q] = L; k1cur = q + 1; }
          renderKeyPanes();
        },
      }, L))));
  }
  // chia 2 cột: nửa đầu bên trái, nửa sau bên phải
  const order = [];
  for (let i = 0; i < half; i++) order.push(rows[i], rows[i + half]);
  $("#ke-p1").replaceChildren(
    h("div", { class: "k1tools" }, h("span", { class: "muted" }, `Đã nhập ${c.p1} câu · câu để trống không chấm`),
      h("button", { onclick: () => { if (confirm("Xoá toàn bộ đáp án Phần I?")) { draft.p1 = {}; k1cur = 1; renderKeyPanes(); } } }, "Xoá hết")),
    h("div", { class: "k1" }, order));
  // Phần II: 8 câu x 4 ý, mỗi ý chọn Đ hoặc S
  $("#ke-p2").replaceChildren(...[1, 2, 3, 4, 5, 6, 7, 8].map((q) => {
    const cur = (draft.p2[q] || "____").split("");
    return h("div", { class: "k2row" }, h("b", {}, `Câu ${q}`, cur.includes("_") && draft.p2[q] ? h("span", { class: "err" }, " – chưa đủ 4 ý") : null),
      h("div", { class: "k2" }, ..."abcd".split("").map((s, i) => h("div", {}, s + ")",
        h("div", { class: "pair" }, ..."ĐS".split("").map((v) => h("button", {
          class: (cur[i] === v ? "on" : "") + (v === "S" ? " s" : ""), onclick: () => {
            cur[i] = cur[i] === v ? "_" : v;
            const s2 = cur.join("");
            if (s2 === "____") delete draft.p2[q]; else draft.p2[q] = s2;
            renderKeyPanes();
          },
        }, v)))))));
  }));
  // Phần III: 6 ô nhập số
  $("#ke-p3").replaceChildren(h("p", { class: "muted small" }, "Ghi số như thí sinh tô: dấu phẩy cho số thập phân (vd 2,75), dấu trừ cho số âm."),
    ...[1, 2, 3, 4, 5, 6].map((q) => h("div", { class: "k3row" }, h("b", {}, "Câu " + q),
      h("input", {
        type: "text", inputmode: "decimal", value: draft.p3[q] || "", placeholder: "(không chấm)", maxlength: 6,
        class: draft.p3[q] && !validP3(draft.p3[q]) ? "invalid" : "",
        onchange: (e) => {
          const v = e.target.value.trim().replace(/\./g, ",").replace(/\s/g, "");
          if (v) draft.p3[q] = v; else delete draft.p3[q];
          e.target.value = v; e.target.classList.toggle("invalid", !!v && !validP3(v));
          const cc = keyCounts(draft); $("#ke-c3").textContent = cc.p3 + "/6";
        },
      }))));
}
async function parseKeyText() {
  const text = $("#ke-text").value;
  $("#ke-err").textContent = "";
  if (!text.trim()) { $("#ke-err").textContent = "Chưa có nội dung."; return; }
  $("#ke-parse").disabled = true; $("#ke-parse").textContent = engine.ready ? "Đang đọc…" : "Chờ bộ chấm khởi động…";
  try {
    const k = await engine.call("parseKey", { text, form: draft.form });
    if (k.error) { $("#ke-err").textContent = k.error; return; }
    draft.p1 = k.p1; draft.p2 = k.p2; draft.p3 = k.p3;
    if (k.made) { draft.made = k.made; $("#ke-made").value = k.made; }
    renderKeyPanes();
    const c = keyCounts(draft);
    toast(`Đã đọc: ${c.p1} câu P.I, ${c.p2} câu P.II, ${c.p3} câu P.III – kiểm tra rồi bấm Lưu`, 3500);
    setKeTab("p1");
  } catch (e) { $("#ke-err").textContent = e.message; }
  finally { $("#ke-parse").disabled = false; $("#ke-parse").textContent = "Đọc đáp án"; }
}
async function saveKey() {
  document.activeElement && document.activeElement.blur();
  draft.made = $("#ke-made").value.trim();
  const errs = [];
  if (draft.made && !/^\d{1,4}$/.test(draft.made)) errs.push("Mã đề chỉ gồm chữ số.");
  const bad2 = Object.entries(draft.p2).filter(([, v]) => v.includes("_")).map(([q]) => q);
  if (bad2.length) errs.push(`Phần II câu ${bad2.join(", ")} chưa đủ 4 ý.`);
  const bad3 = Object.entries(draft.p3).filter(([, v]) => !validP3(v)).map(([q]) => q);
  if (bad3.length) errs.push(`Phần III câu ${bad3.join(", ")} không phải số hợp lệ.`);
  const c = keyCounts(draft);
  if (!c.p1 && !c.p2 && !c.p3) errs.push("Chưa nhập câu nào.");
  const dup = formKeys(draft.form).find((k) => k.id !== draft.id && normMade(k.made) === normMade(draft.made));
  if (dup) errs.push(draft.made ? `Đã có bộ đáp án mã đề ${draft.made}.` : "Đã có một bộ đáp án không ghi mã đề.");
  if (errs.length) { alert(errs.join("\n")); return; }
  const isNew = !draft.id;
  draft.updated = Date.now();
  if (isNew) { draft.id = uid(); keys.push(draft); } else keys[keys.findIndex((k) => k.id === draft.id)] = draft;
  keys.sort((a, b) => (a.made || "").localeCompare(b.made || "", undefined, { numeric: true }));
  if (!formKeys(draft.form).some((k) => k.id === activeKey[draft.form])) activeKey[draft.form] = draft.id;
  saveKeys();
  const saved = draft;
  await popLayer();
  toast(isNew ? "Đã lưu đáp án" : "Đã cập nhật đáp án");
  if (curTab === "keys") renderKeys();
  renderScan();
  // chấm lại các phiếu đã chấm của mã đề này cho đúng đáp án mới
  if (!isNew) {
    const all = await db.all();
    const affected = all.filter((r) => formOf(r) === saved.form && (r.keyId === saved.id || (saved.made && normMade(r.made) === normMade(saved.made))));
    if (affected.length && confirm(`Có ${affected.length} phiếu đã chấm theo đáp án này. Chấm lại theo đáp án mới?`))
      regradeRecords((r) => affected.some((a) => a.id === r.id), "Đang chấm lại");
  }
  if (topLayer() === "result" && cur) regradeCurrent("Đã chấm lại theo đáp án mới");
}
async function deleteKey() {
  if (!confirm(`Xoá bộ đáp án ${draft.made ? "mã đề " + draft.made : "này"}?`)) return;
  keys = keys.filter((k) => k.id !== draft.id);
  if (activeKey[draft.form] === draft.id) delete activeKey[draft.form];
  saveKeys();
  await popLayer();
  renderKeys(); renderScan();
}

// ============================================================ CÀI ĐẶT
function renderSettings() {
  $("#set-p1").value = fmt(settings.p1);
  $("#set-p1en").value = fmt(settings.p1En);
  $("#set-p3").value = String(settings.p3);
  $("#set-p2").replaceChildren(...[1, 2, 3, 4].map((n) => h("label", {}, `${n} ý đúng`,
    h("input", {
      type: "text", inputmode: "decimal", value: fmt(settings.p2[n]), onchange: (e) => {
        const v = num(e.target.value); if (v != null) { settings.p2[n] = v; saveSettings(); } e.target.value = fmt(settings.p2[n]);
      },
    }))));
  $("#set-autokey").checked = settings.autoKey;
  $("#set-delta").value = settings.delta; $("#set-delta-v").textContent = settings.delta;
  $("#set-vibrate").checked = settings.vibrate;
  $("#app-version").textContent = "Phiên bản " + APP_VERSION + (navigator.serviceWorker && navigator.serviceWorker.controller ? " · dùng được khi không có mạng" : "");
  Promise.all([db.all(), navigator.storage && navigator.storage.estimate ? navigator.storage.estimate() : null]).then(([all, est]) => {
    $("#storage-info").textContent = `Đã lưu ${all.length} phiếu` + (est ? ` · bộ nhớ dùng ${(est.usage / 1048576).toFixed(0)} MB` : "");
  });
}
function initSettings() {
  $("#set-p1").onchange = (e) => { const v = num(e.target.value); if (v != null && v > 0) { settings.p1 = v; saveSettings(); } e.target.value = fmt(settings.p1); };
  $("#set-p1en").onchange = (e) => { const v = num(e.target.value); if (v != null && v > 0) { settings.p1En = v; saveSettings(); } e.target.value = fmt(settings.p1En); };
  $("#set-p3").onchange = (e) => { settings.p3 = parseFloat(e.target.value); saveSettings(); };
  $("#set-autokey").onchange = (e) => { settings.autoKey = e.target.checked; saveSettings(); };
  $("#set-delta").oninput = (e) => { $("#set-delta-v").textContent = e.target.value; };
  $("#set-delta").onchange = (e) => { settings.delta = parseInt(e.target.value, 10); saveSettings(); readCache.clear(); };
  $("#set-vibrate").onchange = (e) => { settings.vibrate = e.target.checked; saveSettings(); };
  $("#btn-update").onclick = async () => {
    if (!navigator.serviceWorker) { location.reload(); return; }
    const reg = await navigator.serviceWorker.getRegistration();
    if (reg) { await reg.update(); toast(reg.installing || reg.waiting ? "Đang tải bản mới…" : "Đang dùng bản mới nhất"); }
    else location.reload();
  };
}

// ============================================================ cài đặt ứng dụng + chạy offline
let installEvt = null;
window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); installEvt = e; $("#btn-install").hidden = false; });
window.addEventListener("appinstalled", () => { $("#btn-install").hidden = true; toast("Đã cài ứng dụng vào màn hình chính"); });
function initSW() {
  if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
  const hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register("sw.js").then((reg) => {
    // ứng dụng đã cài có thể mở rất lâu: mỗi lần quay lại thì kiểm tra bản mới
    document.addEventListener("visibilitychange", () => { if (!document.hidden) reg.update().catch(() => {}); });
  }).catch(() => {});
  // Có bản mới -> tự tải lại, nhưng đợi lúc rảnh để không làm mất bài đang xem / đang chấm
  let waitingReload = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!hadController || waitingReload) return;       // lần cài đầu tiên thì không cần tải lại
    waitingReload = true;
    setInterval(() => { if (!layers.length && !jobs) location.reload(); }, 500);
  });
}

// ============================================================ khởi động
function init() {
  $$(".tabbar button").forEach((b) => (b.onclick = () => switchTab(b.dataset.view)));
  $("#btn-back").onclick = () => popLayer();
  $("#btn-camera").onclick = startCapture;
  $("#btn-gallery").onclick = () => $("#file-gallery").click();
  $("#btn-native").onclick = () => $("#file-native").click();
  $("#file-gallery").onchange = (e) => { handleFiles(e.target.files); e.target.value = ""; };
  $("#file-native").onchange = (e) => { handleFiles(e.target.files); e.target.value = ""; };
  $("#sw-continuous").onchange = (e) => { settings.continuous = e.target.checked; saveSettings(); };
  $("#cam-close").onclick = () => popLayer();
  $("#cam-shutter").onclick = onShutter;
  $("#cam-gallery").onclick = () => $("#file-gallery").click();
  $("#cam-torch").onclick = () => {
    cam.torch = !cam.torch;
    cam.track && cam.track.applyConstraints({ advanced: [{ torch: cam.torch }] }).catch(() => toast("Không bật được đèn"));
    $("#cam-torch").classList.toggle("on", cam.torch);
  };
  $("#engine-chip").onclick = () => {
    if (engine.fatal) engine.restart();
    else toast(engine.ready ? "Bộ chấm sẵn sàng – chạy hoàn toàn trên điện thoại, không gửi ảnh đi đâu" : "Đang khởi động bộ chấm, vui lòng chờ…", 3500);
  };
  $("#btn-install").onclick = async () => { if (installEvt) { installEvt.prompt(); await installEvt.userChoice; installEvt = null; $("#btn-install").hidden = true; } };
  // đáp án
  $("#btn-key-new").onclick = () => openKeyEditor(null);
  $("#btn-key-file").onclick = () => $("#file-key").click();
  $("#file-key").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (f) openKeyEditor(null, { text: await f.text() });
  };
  $("#ke-close").onclick = () => popLayer();
  $("#ke-save").onclick = saveKey;
  $("#ke-delete").onclick = deleteKey;
  $("#ke-parse").onclick = parseKeyText;
  $("#ke-paste").onclick = async () => {
    try { $("#ke-text").value = await navigator.clipboard.readText(); parseKeyText(); }
    catch { toast("Không đọc được bộ nhớ tạm – hãy nhấn giữ vào ô rồi chọn Dán"); }
  };
  $$("#ke-tabs button").forEach((b) => (b.onclick = () => setKeTab(b.dataset.tab)));
  $("#ke-made").oninput = (e) => { draft.made = e.target.value.trim(); };
  // lịch sử
  $("#hist-search").oninput = renderHistory;
  $("#btn-export").onclick = exportExcel;
  $("#btn-regrade-all").onclick = () => { if (confirm("Chấm lại tất cả phiếu theo đáp án và thang điểm hiện tại?")) regradeRecords(() => true, "Đang chấm lại"); };
  $("#btn-clear").onclick = async () => {
    if (!confirm("Xoá TẤT CẢ kết quả đã chấm? Nên Xuất Excel trước.")) return;
    await db.clear(); updateBadge(); renderHistory(); toast("Đã xoá toàn bộ kết quả");
  };
  window.addEventListener("resize", () => { if (cam.stream) positionGuide(); });
  initViewer(); initSettings(); initSW();
  engine.start();
  renderFormSwitch(); renderScan(); updateBadge();
  // mở bằng lối tắt "Chụp phiếu" từ biểu tượng ứng dụng
  if (new URLSearchParams(location.search).get("action") === "scan") setTimeout(startCapture, 300);
}
init();
