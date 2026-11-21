(() => {
  const ext = globalThis.browser ?? globalThis.chrome;
  const CHANNEL = '__TRUEHDR_BRIDGE_V1__';

  const DEFAULT_SETTINGS = {
    enabled: true,
    hdrEnabled: true,
    upscaleEnabled: true,
    preset: 'vivid',
    peakBoost: 6.0,
    threshold: 0.46,
    localPunch: 0.70,
    gamutBoost: 0.45,
    shadowDepth: 0.14,
    aplGuard: 0.45,
    upscaleQuality: 'quality',
    upscaleContent: 'live',
    upscaleSharpness: 0.50,
    bypassNativeHdr: true,
    imagesEnabled: false,
    splitEnabled: false,
    splitPos: 0.5,
    disabledSites: {},
    settingsVersion: 2
  };

  function migrateSettings(raw) {
    const merged = { ...DEFAULT_SETTINGS, ...(raw || {}) };
    const legacy = !raw || raw.settingsVersion !== DEFAULT_SETTINGS.settingsVersion;
    // Legacy 'upscaleMode': only 'off' carries meaning (auto/4k targets are gone).
    if (merged.upscaleMode === 'off' && (!raw || typeof raw.upscaleEnabled !== 'boolean')) {
      merged.upscaleEnabled = false;
    }
    delete merged.upscaleMode;
    // Sharpness is now an optional post-sharpen on top of AI SR: old default 0.55 -> 0.25.
    if (legacy && raw && raw.upscaleSharpness === 0.55) merged.upscaleSharpness = 0.25;
    if (merged.upscaleQuality !== 'fast' && merged.upscaleQuality !== 'quality') merged.upscaleQuality = DEFAULT_SETTINGS.upscaleQuality;
    if (merged.upscaleContent !== 'live' && merged.upscaleContent !== 'anime') merged.upscaleContent = DEFAULT_SETTINGS.upscaleContent;
    merged.settingsVersion = DEFAULT_SETTINGS.settingsVersion;
    return merged;
  }

  let cachedSettings = migrateSettings(null);
  let lastStatus = null;
  let engineReady = false;

  function computeEffectiveSettings(raw) {
    const merged = migrateSettings(raw);
    const host = location.hostname || '';
    const siteDisabled = Boolean(merged.disabledSites && merged.disabledSites[host]);
    const siteActive = Boolean(merged.enabled && !siteDisabled);
    const effectiveHdr = siteActive && Boolean(merged.hdrEnabled);
    const effectiveUpscale = siteActive && Boolean(merged.upscaleEnabled);
    return {
      ...merged,
      host,
      siteDisabled,
      effectiveHdr,
      effectiveUpscale,
      effectiveEnabled: effectiveHdr || effectiveUpscale,
      effectiveImages: effectiveHdr && Boolean(merged.imagesEnabled)
    };
  }

  function isDisplayedImageUrl(url) {
    if (!/^https?:/i.test(url)) return false;
    for (const img of document.images) {
      if (img.currentSrc === url || img.src === url) return true;
    }
    return false;
  }

  function postToMain(kind, payload) {
    window.postMessage({ channel: CHANNEL, from: 'bridge', kind, payload }, '*');
  }

  async function loadAndPushSettings() {
    try {
      const res = await ext.storage.local.get('truehdrSettings');
      cachedSettings = migrateSettings(res?.truehdrSettings);
    } catch (_) {
      cachedSettings = migrateSettings(null);
    }
    postToMain('SETTINGS', computeEffectiveSettings(cachedSettings));
  }

  async function updateStoredSettings(mutate) {
    try {
      const res = await ext.storage.local.get('truehdrSettings');
      const cur = migrateSettings(res?.truehdrSettings);
      mutate(cur);
      await ext.storage.local.set({ truehdrSettings: cur });
    } catch (_) {}
  }

  window.addEventListener('message', async (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.channel !== CHANNEL || data.from !== 'engine') return;

    if (data.kind === 'READY') {
      engineReady = true;
      await loadAndPushSettings();
    } else if (data.kind === 'STATUS') {
      lastStatus = data.payload;
      try {
        ext.runtime.sendMessage({ type: 'TRUEHDR_STATUS_UPDATE', payload: lastStatus });
      } catch (_) {}
    } else if (data.kind === 'TOGGLE_SITE') {
      const host = location.hostname || '';
      if (!host) return;
      await updateStoredSettings((cur) => {
        const disabledSites = { ...(cur.disabledSites || {}) };
        if (!cur.enabled) {
          cur.enabled = true;
          delete disabledSites[host];
        } else if (disabledSites[host]) {
          delete disabledSites[host];
        } else {
          disabledSites[host] = true;
        }
        cur.disabledSites = disabledSites;
      });
    } else if (data.kind === 'TOGGLE_MASTER') {
      await updateStoredSettings((cur) => { cur.enabled = !cur.enabled; });
    } else if (data.kind === 'TOGGLE_HDR') {
      await updateStoredSettings((cur) => { cur.hdrEnabled = !cur.hdrEnabled; });
    } else if (data.kind === 'TOGGLE_UPSCALE') {
      await updateStoredSettings((cur) => { cur.upscaleEnabled = !cur.upscaleEnabled; });
    } else if (data.kind === 'TOGGLE_IMAGES') {
      await updateStoredSettings((cur) => { cur.imagesEnabled = !cur.imagesEnabled; });
    } else if (data.kind === 'FETCH_IMAGE') {
      // Cross-origin <img> without CORS headers: the engine cannot read its pixels, so the
      // background fetches the bytes (no cookies). Only for URLs this page actually displays.
      const id = data.payload && data.payload.id;
      const url = data.payload && typeof data.payload.url === 'string' ? data.payload.url : '';
      let resp = null;
      if (url && isDisplayedImageUrl(url)) {
        try {
          resp = await ext.runtime.sendMessage({ type: 'TRUEHDR_FETCH_IMAGE', url });
        } catch (err) {
          resp = { ok: false, error: String(err) };
        }
      } else {
        resp = { ok: false, error: 'not a displayed image' };
      }
      let blob = null;
      if (resp && resp.ok && typeof resp.b64 === 'string') {
        try {
          const bin = atob(resp.b64);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          blob = new Blob([bytes], { type: resp.mime || '' });
        } catch (_) {
          blob = null;
        }
      }
      postToMain('IMAGE_DATA', { id, ok: Boolean(blob), error: blob ? null : (resp && resp.error) || 'fetch failed', blob });
    } else if (data.kind === 'TOGGLE_SPLIT') {
      await updateStoredSettings((cur) => { cur.splitEnabled = !cur.splitEnabled; });
    } else if (data.kind === 'SET_SPLIT_POS') {
      const pos = data.payload && typeof data.payload.splitPos === 'number' ? data.payload.splitPos : null;
      if (pos === null || !isFinite(pos)) return;
      await updateStoredSettings((cur) => {
        cur.splitPos = Math.min(0.95, Math.max(0.05, Math.round(pos * 100) / 100));
        cachedSettings = cur;
      });
    }
  });

  ext.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes.truehdrSettings) return;
    cachedSettings = migrateSettings(changes.truehdrSettings.newValue);
    postToMain('SETTINGS', computeEffectiveSettings(cachedSettings));
  });

  ext.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || typeof message !== 'object') return false;
    if (message.type === 'TRUEHDR_GET_FRAME_STATUS') {
      sendResponse({ status: lastStatus });
      return false;
    }
    if (message.type === 'TRUEHDR_RETRY_CORS') {
      postToMain('RETRY_CORS', {});
      sendResponse({ ok: true });
      return false;
    }
    if (message.type === 'TRUEHDR_SET_SPLIT_POS') {
      postToMain('SETTINGS', { ...computeEffectiveSettings(cachedSettings), splitPos: message.splitPos });
      return false;
    }
    return false;
  });

  loadAndPushSettings();

  setTimeout(() => {
    if (!engineReady) {
      try {
        ext.runtime.sendMessage({ type: 'TRUEHDR_INJECT_MAIN' });
      } catch (_) {}
    }
  }, 150);
})();
