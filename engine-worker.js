/* Bộ chấm chạy nền: Python (Pyodide) + OpenCV + omr.py, hoàn toàn trên điện thoại. */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
let bridge = null;
const say = (stage, pct) => postMessage({ type: "progress", stage, pct });

async function boot() {
  say("Đang tải Python…", 5);
  importScripts(PYODIDE_URL + "pyodide.js");
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  say("Đang tải OpenCV (chỉ lần đầu)…", 35);
  await py.loadPackage(["numpy", "opencv-python"]);
  say("Đang nạp bộ chấm…", 85);
  py.FS.mkdirTree("/app/models");
  for (const f of ["omr.py", "bridge.py", "models/mnist12.npz"]) {
    const r = await fetch(f);
    if (!r.ok) throw new Error("Không tải được " + f);
    py.FS.writeFile("/app/" + f, new Uint8Array(await r.arrayBuffer()));
  }
  py.runPython("import sys; sys.path.insert(0, '/app'); import bridge");
  bridge = py.pyimport("bridge");
  say("Sẵn sàng", 100);
}
const ready = boot();
ready.catch((err) => postMessage({ type: "fatal", error: String((err && err.message) || err) }));

function toImageData(t) {          // (w, h, bytes) từ Python -> ImageData
  const [w, h, b] = t.toJs();
  t.destroy();
  return new ImageData(new Uint8ClampedArray(b.buffer, b.byteOffset, b.byteLength), w, h);
}
function toBlob(img, q) {
  const c = new OffscreenCanvas(img.width, img.height);
  c.getContext("2d").putImageData(img, 0, 0);
  return c.convertToBlob({ type: "image/jpeg", quality: q });
}

const handlers = {
  has: ({ token }) => bridge.has(token),
  async process({ token, rgba, w, h, delta }) {
    const t0 = performance.now();
    const out = JSON.parse(bridge.process(token, rgba, w, h, delta));
    if (out.error) return out;
    out.crops = [];
    for (let i = 0; i < out.part3.length; i++) out.crops.push(await toBlob(toImageData(bridge.crop(token, i)), 0.9));
    out.ms = Math.round(performance.now() - t0);
    return out;
  },
  async grade({ token, key, settings, override }) {
    const out = JSON.parse(bridge.grade(token, key ? JSON.stringify(key) : "",
      JSON.stringify(settings || {}), JSON.stringify(override || {})));
    out.image = await toBlob(toImageData(bridge.annotated(token, 1800)), 0.85);
    return out;
  },
  parseKey: ({ text }) => JSON.parse(bridge.parse_key(text)),
};

// Các lệnh được xử lý lần lượt (bộ chấm dùng chung trạng thái)
let chain = Promise.resolve();
onmessage = (e) => {
  const { id, cmd, args } = e.data;
  chain = chain.then(async () => {
    try {
      await ready;
      postMessage({ type: "reply", id, ok: true, data: await handlers[cmd](args || {}) });
    } catch (err) {
      postMessage({ type: "reply", id, ok: false, error: String((err && err.message) || err) });
    }
  });
};
