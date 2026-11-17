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
  upscaleSharpness: 0.25,
  bypassNativeHdr: true,
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

  if (message.type === 'TRUEHDR_GET_TAB_STATUS') {
    const state = tabStates.get(message.tabId) || null;
    sendResponse({ state });
    return false;
  }

  return false;
});

ext.tabs.onRemoved.addListener((tabId) => {
  tabStates.delete(tabId);
});

async function updateBadge(tabId, status) {
  if (!ext.action || !ext.action.setBadgeText) return;
  try {
    let text = '';
    let color = '#1e293b';
    if (status && status.activeCount > 0) {
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
