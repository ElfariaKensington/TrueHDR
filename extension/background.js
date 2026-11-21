const ext = globalThis.browser ?? globalThis.chrome;

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

const tabStates = new Map();

// MAIN-world engine: model weights, then SR runtime, then the engine (order matters).
const MAIN_WORLD_FILES = ['sr-models.js', 'sr-runtime.js', 'hdr-engine.js'];
const MAIN_SCRIPT = {
  id: 'truehdr-main',
  matches: ['<all_urls>'],
  js: MAIN_WORLD_FILES,
  runAt: 'document_start',
  allFrames: true,
  world: 'MAIN'
};

async function ensureMainWorldScriptRegistered() {
  if (!ext.scripting || !ext.scripting.registerContentScripts) return;
  try {
    const existing = await ext.scripting.getRegisteredContentScripts({ ids: [MAIN_SCRIPT.id] });
    if (existing && existing.length > 0) {
      // Registrations persist across updates: replace one with an outdated file list.
      const js = existing[0].js || [];
      if (js.length === MAIN_WORLD_FILES.length && js.every((f, i) => f === MAIN_WORLD_FILES[i])) return;
      await ext.scripting.unregisterContentScripts({ ids: [MAIN_SCRIPT.id] });
    }
    await ext.scripting.registerContentScripts([MAIN_SCRIPT]);
  } catch (err) {
    console.warn('[TrueHDR] registerContentScripts fallback:', err);
  }
}

ext.runtime.onInstalled.addListener(async () => {
  await ensureMainWorldScriptRegistered();
  try {
    const existing = await ext.storage.local.get('truehdrSettings');
    const stored = existing && existing.truehdrSettings;
    if (!stored || stored.settingsVersion !== DEFAULT_SETTINGS.settingsVersion) {
      await ext.storage.local.set({ truehdrSettings: migrateSettings(stored) });
    }
  } catch (err) {
    console.warn('[TrueHDR] Failed to initialize settings:', err);
  }
});

if (ext.runtime.onStartup) {
  ext.runtime.onStartup.addListener(() => {
    ensureMainWorldScriptRegistered();
  });
}

ensureMainWorldScriptRegistered();

ext.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || typeof message !== 'object') return false;

  if (message.type === 'TRUEHDR_INJECT_MAIN' && sender.tab && sender.tab.id >= 0) {
    const target = { tabId: sender.tab.id };
    if (typeof sender.frameId === 'number') {
      target.frameIds = [sender.frameId];
    }
    ext.scripting
      .executeScript({
        target,
        world: 'MAIN',
        files: MAIN_WORLD_FILES
      })
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === 'TRUEHDR_STATUS_UPDATE' && sender.tab && sender.tab.id >= 0) {
    const tabId = sender.tab.id;
    tabStates.set(tabId, {
      ...message.payload,
      updatedAt: Date.now()
    });
    updateBadge(tabId, message.payload);
    return false;
  }

  if (message.type === 'TRUEHDR_FETCH_IMAGE' && sender.tab && sender.tab.id >= 0) {
    fetchImageBytes(String(message.url || ''))
      .then((r) => sendResponse({ ok: true, ...r }))
      .catch((err) => sendResponse({ ok: false, error: String(err && err.message ? err.message : err) }));
    return true;
  }

  if (message.type === 'TRUEHDR_GET_TAB_STATUS') {
    const state = tabStates.get(message.tabId) || null;
    sendResponse({ state });
    return false;
  }

  return false;
});

// Image bytes for cross-origin <img> elements that lack CORS headers (HDR-for-images option).
// The page can read whatever this returns, so it must never act as a credentialed or
// intranet proxy: no cookies, public http(s) hosts only, image/* responses only, size-capped.
const IMAGE_FETCH_MAX_BYTES = 24 * 1024 * 1024;

function isPrivateHost(host) {
  const h = host.toLowerCase().replace(/^\[|\]$/g, '');
  if (h === 'localhost' || h.endsWith('.localhost') || h.endsWith('.local') || h.endsWith('.internal')) return true;
  if (!h.includes('.') && !h.includes(':')) return true; // single-label intranet names
  const v4 = h.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return a === 0 || a === 10 || a === 127 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
  }
  if (h.includes(':')) {
    return h === '::1' || h === '::' || h.startsWith('fc') || h.startsWith('fd') || h.startsWith('fe80') || h.startsWith('::ffff:');
  }
  return false;
}

function bytesToBase64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

async function fetchImageBytes(url) {
  const u = new URL(url);
  if (u.protocol !== 'https:' && u.protocol !== 'http:') throw new Error('unsupported scheme');
  if (isPrivateHost(u.hostname)) throw new Error('private host');
  const resp = await fetch(u.href, { credentials: 'omit', cache: 'force-cache', redirect: 'follow', referrerPolicy: 'no-referrer' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  if (resp.url && isPrivateHost(new URL(resp.url).hostname)) throw new Error('private host');
  const mime = (resp.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  if (!mime.startsWith('image/')) throw new Error(`not an image (${mime || 'unknown'})`);
  const declared = Number(resp.headers.get('content-length') || 0);
  if (declared > IMAGE_FETCH_MAX_BYTES) throw new Error('image too large');
  const buf = new Uint8Array(await resp.arrayBuffer());
  if (buf.length > IMAGE_FETCH_MAX_BYTES) throw new Error('image too large');
  return { mime, b64: bytesToBase64(buf) };
}

ext.tabs.onRemoved.addListener((tabId) => {
  tabStates.delete(tabId);
});

async function updateBadge(tabId, status) {
  if (!ext.action || !ext.action.setBadgeText) return;
  try {
    let text = '';
    let color = '#1e293b';
    if (status && (status.activeCount > 0 || status.imageActiveCount > 0)) {
      if (status.effectiveHdr && !status.bypassedNativeHdr) {
        text = status.upscalingActive ? 'H+SR' : (status.isEdrDisplay ? 'HDR' : 'ON');
        color = status.isEdrDisplay ? '#f59e0b' : '#3b82f6';
      } else if (status.upscalingActive) {
        text = 'SR';
        color = '#10b981';
      }
    } else if (status && status.bypassedNativeHdr) {
      text = 'NAT';
      color = '#10b981';
    } else if (status && status.blockedReason) {
      text = '!';
      color = '#ef4444';
    }
    await ext.action.setBadgeText({ tabId, text });
    if (text) {
      await ext.action.setBadgeBackgroundColor({ tabId, color });
    }
  } catch (_) {
    // Ignore badge errors on closed/restricted tabs
  }
}
