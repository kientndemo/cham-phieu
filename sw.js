/* Service worker: lưu sẵn ứng dụng + Python/OpenCV để dùng được cả khi không có mạng. */
const VERSION = "72e908b331"; // build_mobile.py tự cập nhật mỗi lần đóng gói
const APP_CACHE = "cham-app-" + VERSION;
const CDN = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
const CDN_CACHE = "cham-pyodide-v0.29.3";
const APP_FILES = ["./", "index.html", "app.css", "app.js", "xlsx.js", "engine-worker.js", "bridge.py", "omr.py",
  "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "icons/maskable-512.png"];
const CDN_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json",
  "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl", "opencv_python-4.11.0.86-cp313-cp313-pyodide_2025_0_wasm32.whl"].map((f) => CDN + f);

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(APP_CACHE);
    await c.addAll(APP_FILES.map((f) => new Request(f, { cache: "reload" })));
    // Python + OpenCV (~25 MB): tải 1 lần, không bắt buộc thành công ngay
    caches.open(CDN_CACHE).then(async (cc) => {
      for (const u of CDN_FILES) if (!(await cc.match(u))) await cc.add(new Request(u, { mode: "cors" })).catch(() => {});
    });
    self.skipWaiting();   // luôn dùng ngay bản mới, không chờ người dùng đóng hết cửa sổ
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys())
      if ((k.startsWith("cham-app-") && k !== APP_CACHE) || (k.startsWith("cham-pyodide-") && k !== CDN_CACHE)) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (req.url.startsWith(CDN)) {
    e.respondWith(caches.open(CDN_CACHE).then(async (c) => {
      const hit = await c.match(req.url);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req.url, res.clone());
      return res;
    }));
    return;
  }
  if (url.origin !== location.origin) return;
  e.respondWith((async () => {
    const c = await caches.open(APP_CACHE);
    const hit = await c.match(req, { ignoreSearch: true });
    if (hit) return hit;
    try { return await fetch(req); }
    catch (err) {
      if (req.mode === "navigate") return (await c.match("index.html")) || Response.error();
      throw err;
    }
  })());
});
