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
    upscaleMode: 'auto',
    upscaleSharpness: 0.55,
    bypassNativeHdr: true,
    splitEnabled: false,
    splitPos: 0.5,
    disabledSites: {}
  };

  let cachedSettings = { ...DEFAULT_SETTINGS };
  let lastStatus = null;
  let engineReady = false;

  function computeEffectiveSettings(raw) {
    const merged = { ...DEFAULT_SETTINGS, ...(raw || {}) };
    if (merged.upscaleMode === 'off') {
      if (!raw || typeof raw.upscaleEnabled !== 'boolean') {
        merged.upscaleEnabled = false;
      }
      merged.upscaleMode = 'auto';
    }
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
      effectiveEnabled: effectiveHdr || effectiveUpscale
    };
  }

  function postToMain(kind, payload) {
    window.postMessage({ channel: CHANNEL, from: 'bridge', kind, payload }, '*');
  }

  async function loadAndPushSettings() {
    try {
      const res = await ext.storage.local.get('truehdrSettings');
      cachedSettings = { ...DEFAULT_SETTINGS, ...(res?.truehdrSettings || {}) };
    } catch (_) {
      cachedSettings = { ...DEFAULT_SETTINGS };
    }
    postToMain('SETTINGS', computeEffectiveSettings(cachedSettings));
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
      try {
        const res = await ext.storage.local.get('truehdrSettings');
        const cur = { ...DEFAULT_SETTINGS, ...(res?.truehdrSettings || {}) };
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
        await ext.storage.local.set({ truehdrSettings: cur });
      } catch (_) {}
    } else if (data.kind === 'TOGGLE_SPLIT') {
      try {
        const res = await ext.storage.local.get('truehdrSettings');
        const cur = { ...DEFAULT_SETTINGS, ...(res?.truehdrSettings || {}) };
        cur.splitEnabled = !cur.splitEnabled;
        await ext.storage.local.set({ truehdrSettings: cur });
      } catch (_) {}
    }
  });

  ext.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes.truehdrSettings) return;
    cachedSettings = { ...DEFAULT_SETTINGS, ...(changes.truehdrSettings.newValue || {}) };
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
