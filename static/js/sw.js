const CACHE = "ac-monitor-v1";
const ASSETS = ["/", "/static/icons/icon-192.png"];

// Disable service worker caching on localhost
if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
  self.addEventListener("fetch", (e) => {
    e.respondWith(fetch(e.request));
  });
} else {
  // Normal PWA caching for production (Render)
  self.addEventListener("install", (e) => {
    e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
    self.skipWaiting();
  });

  self.addEventListener("activate", (e) => {
    e.waitUntil(
      caches
        .keys()
        .then((keys) =>
          Promise.all(
            keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)),
          ),
        ),
    );
    self.clients.claim();
  });

  self.addEventListener("fetch", (e) => {
    if (
      e.request.url.includes("/predict") ||
      e.request.url.includes("/history")
    )
      return;
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
  });
}
