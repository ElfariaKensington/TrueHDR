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
  upscaleMode: 'auto',
  upscaleSharpness: 0.55,
  bypassNativeHdr: true,
  splitEnabled: false,
  splitPos: 0.5,
  disabledSites: {}
};

const tabStates = new Map();

async function ensureMainWorldScriptRegistered() {
  if (!ext.scripting || !ext.scripting.registerContentScripts) return;
  try {
    const existing = await ext.scripting.getRegisteredContentScripts({ ids: ['truehdr-main'] });
    if (existing && existing.length > 0) return;
    await ext.scripting.registerContentScripts([
      {
        id: 'truehdr-main',
        matches: ['<all_urls>'],
        js: ['hdr-engine.js'],
        runAt: 'document_start',
        allFrames: true,
        world: 'MAIN'
      }
    ]);
  } catch (err) {
    console.warn('[TrueHDR] registerContentScripts fallback:', err);
  }
}

ext.runtime.onInstalled.addListener(async () => {
  await ensureMainWorldScriptRegistered();
  try {
    const existing = await ext.storage.local.get('truehdrSettings');
    if (!existing || !existing.truehdrSettings) {
      await ext.storage.local.set({ truehdrSettings: DEFAULT_SETTINGS });
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
        files: ['hdr-engine.js']
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
        text = status.upscalingActive ? 'H+4K' : (status.isEdrDisplay ? 'HDR' : 'ON');
        color = status.isEdrDisplay ? '#f59e0b' : '#3b82f6';
      } else if (status.upscalingActive) {
        text = '4K';
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
