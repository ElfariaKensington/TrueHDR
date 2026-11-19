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

function normalizeSettings(raw) {
  return migrateSettings(raw);
}

const PRESETS = {
  cinema: {
    peakBoost: 4.0,
    threshold: 0.52,
    localPunch: 0.50,
    gamutBoost: 0.30,
    shadowDepth: 0.15,
    aplGuard: 0.60
  },
  vivid: {
    peakBoost: 6.0,
    threshold: 0.46,
    localPunch: 0.70,
    gamutBoost: 0.45,
    shadowDepth: 0.14,
    aplGuard: 0.45
  },
  neon: {
    peakBoost: 10.0,
    threshold: 0.38,
    localPunch: 0.95,
    gamutBoost: 0.70,
    shadowDepth: 0.18,
    aplGuard: 0.30
  },
  gentle: {
    peakBoost: 2.6,
    threshold: 0.58,
    localPunch: 0.35,
    gamutBoost: 0.20,
    shadowDepth: 0.08,
    aplGuard: 0.70
  }
};

let settings = { ...DEFAULT_SETTINGS };
let currentHost = '';
let currentTabId = null;

const el = (id) => document.getElementById(id);

function renderUI() {
  el('masterToggle').checked = Boolean(settings.enabled);
  el('upscaleToggle').checked = Boolean(settings.upscaleEnabled);
  el('hdrToggle').checked = Boolean(settings.hdrEnabled);

  el('upscaleBody').classList.toggle('is-dimmed', !settings.upscaleEnabled);
  el('hdrBody').classList.toggle('is-dimmed', !settings.hdrEnabled);

  const siteDisabled = Boolean(currentHost && settings.disabledSites && settings.disabledSites[currentHost]);
  el('siteHost').textContent = currentHost || 'No active site';
  const siteBtn = el('siteToggleBtn');
  siteBtn.disabled = !currentHost;
  if (!currentHost) {
    siteBtn.textContent = 'Unavailable';
    siteBtn.classList.remove('is-off');
  } else if (siteDisabled) {
    siteBtn.textContent = 'Paused on site';
    siteBtn.classList.add('is-off');
  } else {
    siteBtn.textContent = 'Enabled on site';
    siteBtn.classList.remove('is-off');
  }

  for (const btn of document.querySelectorAll('.preset-btn')) {
    btn.classList.toggle('active', btn.dataset.preset === settings.preset);
  }

  for (const btn of document.querySelectorAll('.upscale-btn[data-quality]')) {
    btn.classList.toggle('active', btn.dataset.quality === settings.upscaleQuality);
  }
  for (const btn of document.querySelectorAll('.upscale-btn[data-content]')) {
    btn.classList.toggle('active', btn.dataset.content === settings.upscaleContent);
  }

  const sharp = settings.upscaleSharpness ?? DEFAULT_SETTINGS.upscaleSharpness;
  el('upscaleSharpness').value = sharp;
  el('upscaleSharpnessVal').textContent = `${Math.round(sharp * 100)}%`;

  el('peakBoost').value = settings.peakBoost;
  el('peakBoostVal').textContent = `${Number(settings.peakBoost).toFixed(1)}×`;

  el('threshold').value = settings.threshold;
  el('thresholdVal').textContent = Number(settings.threshold).toFixed(2);

  el('localPunch').value = settings.localPunch;
  el('localPunchVal').textContent = `${Math.round(settings.localPunch * 100)}%`;

  el('gamutBoost').value = settings.gamutBoost;
  el('gamutBoostVal').textContent = `${Math.round(settings.gamutBoost * 100)}%`;

  el('shadowDepth').value = settings.shadowDepth;
  el('shadowDepthVal').textContent = `${Math.round(settings.shadowDepth * 100)}%`;

  el('aplGuard').value = settings.aplGuard;
  el('aplGuardVal').textContent = `${Math.round(settings.aplGuard * 100)}%`;

  el('splitEnabled').checked = Boolean(settings.splitEnabled);
  el('splitPos').value = settings.splitPos;
  el('splitPosVal').textContent = `${Math.round(settings.splitPos * 100)}%`;
  el('splitPosRow').classList.toggle('hidden', !settings.splitEnabled);

  el('bypassNativeHdr').checked = Boolean(settings.bypassNativeHdr);
  el('imagesEnabled').checked = Boolean(settings.imagesEnabled);
}

function fmtRes(res) {
  return String(res).replace(/(\d)\s*[x×]\s*(\d)/, '$1×$2');
}

// e.g. "1280×720 → 2880×1620 · AI 2× Quality · 3.1 ms · fallback: <reason>"
function srDetail(status) {
  const parts = [];
  if (status.sourceRes && status.outputRes) parts.push(`${fmtRes(status.sourceRes)} → ${fmtRes(status.outputRes)}`);
  const passes = Number(status.srPasses);
  const scale = Number.isFinite(passes) && passes > 0 ? `${2 ** passes}× ` : '';
  const model = typeof status.srModel === 'string' ? status.srModel : '';
  const tier = /fast/i.test(model) ? 'Fast'
    : /quality/i.test(model) ? 'Quality'
    : (settings.upscaleQuality === 'fast' ? 'Fast' : 'Quality');
  parts.push(`AI ${scale}${tier}`);
  const gpuMs = Number(status.gpuMs);
  if (status.gpuMs != null && Number.isFinite(gpuMs)) parts.push(`${gpuMs.toFixed(1)} ms`);
  if (status.srFallback) parts.push(`fallback: ${status.srFallback}`);
  return parts.join(' · ');
}

function renderStatus(status) {
  const badge = el('statusBadge');
  const detail = el('statusDetail');
  const corsBanner = el('corsBanner');
  const retryBtn = el('retryCorsBtn');
  const resBadge = el('resBadge');

  badge.className = 'badge';
  corsBanner.classList.add('hidden');
  if (!settings.upscaleEnabled) {
    resBadge.textContent = status && status.sourceRes ? `${status.sourceRes} (Off)` : 'Off';
  } else {
    resBadge.textContent = status && status.sourceRes && status.outputRes
      ? `${status.sourceRes} → ${status.outputRes}`
      : 'Native';
  }

  if (!settings.enabled) {
    badge.classList.add('badge-idle');
    badge.textContent = 'Off';
    detail.textContent = 'Extension is globally paused';
    return;
  }

  if (!settings.hdrEnabled && !settings.upscaleEnabled) {
    badge.classList.add('badge-idle');
    badge.textContent = 'Paused';
    detail.textContent = 'Enable Upscaling or HDR below';
    return;
  }

  if (currentHost && settings.disabledSites && settings.disabledSites[currentHost]) {
    badge.classList.add('badge-idle');
    badge.textContent = 'Paused';
    detail.textContent = `Disabled for ${currentHost}`;
    return;
  }

  if (!status) {
    const isEdr = window.matchMedia('(dynamic-range: high)').matches;
    badge.classList.add('badge-idle');
    badge.textContent = isEdr ? 'Ready' : 'Standby';
    detail.textContent = 'Waiting for video on this tab';
    return;
  }

  if (!status.hasWebGPU) {
    badge.classList.add('badge-warn');
    badge.textContent = 'No WebGPU';
    detail.textContent = 'Enable WebGPU in Safari settings';
    return;
  }

  if (status.blockedReason) {
    badge.classList.add('badge-warn');
    badge.textContent = 'Blocked';
    detail.textContent = status.blockedReason;
    if (status.blockedReason.includes('CORS')) {
      el('corsBannerText').textContent = 'Cross-origin video needs CORS permission.';
      retryBtn.classList.remove('hidden');
      corsBanner.classList.remove('hidden');
    } else if (status.blockedReason.includes('DRM')) {
      el('corsBannerText').textContent = 'DRM-protected video cannot be read by GPU shaders.';
      retryBtn.classList.add('hidden');
      corsBanner.classList.remove('hidden');
    }
    return;
  }

  if (status.bypassedNativeHdr && !status.upscalingActive) {
    badge.classList.add('badge-native');
    badge.textContent = 'Native HDR';
    detail.textContent = 'Source is already HDR — passing through untouched';
    return;
  }

  if (status.activeCount > 0) {
    const hdrActive = settings.hdrEnabled && !status.bypassedNativeHdr;
    const upActive = Boolean(status.upscalingActive);
    if (hdrActive && status.isEdrDisplay) {
      badge.classList.add('badge-edr');
      badge.textContent = upActive ? 'EDR + AI SR' : 'EDR Active';
      detail.textContent = upActive
        ? srDetail(status)
        : `Float16 Display-P3 HDR (${status.activeCount} video${status.activeCount > 1 ? 's' : ''})`;
    } else if (hdrActive && !status.isEdrDisplay) {
      badge.classList.add('badge-sdr');
      badge.textContent = upActive ? 'SDR + AI SR' : 'SDR Display';
      detail.textContent = 'Move window to an HDR/XDR screen for full peak brightness';
    } else if (upActive) {
      badge.classList.add('badge-native');
      badge.textContent = 'AI SR';
      detail.textContent = srDetail(status);
    } else {
      badge.classList.add('badge-idle');
      badge.textContent = 'Passthrough';
      detail.textContent = 'Source resolution >= display (upscale standby)';
    }
    return;
  }

  if (status.imageActiveCount > 0 && settings.hdrEnabled) {
    const n = status.imageActiveCount;
    const anim = status.imageAnimatedCount > 0 ? ` (${status.imageAnimatedCount} animated)` : '';
    badge.classList.add(status.isEdrDisplay ? 'badge-edr' : 'badge-sdr');
    badge.textContent = status.isEdrDisplay ? 'EDR Images' : 'SDR Display';
    detail.textContent = `${n} image${n > 1 ? 's' : ''} in HDR${anim}`;
    return;
  }

  badge.classList.add('badge-idle');
  badge.textContent = status.isEdrDisplay ? 'EDR Ready' : 'Standby';
  detail.textContent = status.videoCount > 0
    ? 'Video paused or initializing…'
    : (settings.imagesEnabled ? 'No video; no large images in view' : 'No active video on this page');
}

let saveTimer = null;
function saveSettings(immediate = false) {
  const write = () => ext.storage.local.set({ truehdrSettings: settings }).catch(() => {});
  if (immediate) {
    clearTimeout(saveTimer);
    write();
  } else {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(write, 35);
  }
}

async function refreshTabStatus() {
  if (!currentTabId) {
    renderStatus(null);
    return;
  }
  let status = null;
  try {
    const resp = await ext.tabs.sendMessage(currentTabId, { type: 'TRUEHDR_GET_FRAME_STATUS' });
    if (resp && resp.status) status = resp.status;
  } catch (_) {}

  if (!status) {
    try {
      const bgResp = await ext.runtime.sendMessage({ type: 'TRUEHDR_GET_TAB_STATUS', tabId: currentTabId });
      if (bgResp && bgResp.state) status = bgResp.state;
    } catch (_) {}
  }

  renderStatus(status);
}

async function init() {
  try {
    const stored = await ext.storage.local.get('truehdrSettings');
    if (stored && stored.truehdrSettings) {
      settings = normalizeSettings(stored.truehdrSettings);
    }
  } catch (_) {}

  try {
    const tabs = await ext.tabs.query({ active: true, currentWindow: true });
    const tab = tabs && tabs[0];
    if (tab) {
      currentTabId = tab.id;
      if (tab.url && (tab.url.startsWith('http://') || tab.url.startsWith('https://'))) {
        currentHost = new URL(tab.url).hostname;
      }
    }
  } catch (_) {}

  renderUI();
  await refreshTabStatus();

  el('masterToggle').addEventListener('change', (e) => {
    settings.enabled = e.target.checked;
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });

  el('upscaleToggle').addEventListener('change', (e) => {
    settings.upscaleEnabled = e.target.checked;
    renderUI();
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });

  el('hdrToggle').addEventListener('change', (e) => {
    settings.hdrEnabled = e.target.checked;
    renderUI();
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });

  el('siteToggleBtn').addEventListener('click', () => {
    if (!currentHost) return;
    const disabledSites = { ...(settings.disabledSites || {}) };
    if (disabledSites[currentHost]) {
      delete disabledSites[currentHost];
    } else {
      disabledSites[currentHost] = true;
    }
    settings.disabledSites = disabledSites;
    renderUI();
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });

  for (const btn of document.querySelectorAll('.preset-btn')) {
    btn.addEventListener('click', () => {
      const key = btn.dataset.preset;
      const p = PRESETS[key];
      if (!p) return;
      settings = { ...settings, ...p, preset: key, hdrEnabled: true };
      renderUI();
      saveSettings(true);
      setTimeout(refreshTabStatus, 140);
    });
  }

  for (const btn of document.querySelectorAll('.upscale-btn')) {
    btn.addEventListener('click', () => {
      if (btn.dataset.quality) settings.upscaleQuality = btn.dataset.quality;
      if (btn.dataset.content) settings.upscaleContent = btn.dataset.content;
      settings.upscaleEnabled = true;
      renderUI();
      saveSettings(true);
      setTimeout(refreshTabStatus, 160);
    });
  }

  const sliderKeys = ['peakBoost', 'threshold', 'localPunch', 'gamutBoost', 'shadowDepth', 'aplGuard', 'upscaleSharpness'];
  for (const key of sliderKeys) {
    el(key).addEventListener('input', (e) => {
      settings[key] = parseFloat(e.target.value);
      if (key !== 'upscaleSharpness') settings.preset = 'custom';
      renderUI();
      saveSettings(false);
    });
  }

  el('splitEnabled').addEventListener('change', (e) => {
    settings.splitEnabled = e.target.checked;
    renderUI();
    saveSettings(true);
  });

  el('splitPos').addEventListener('input', (e) => {
    settings.splitPos = parseFloat(e.target.value);
    renderUI();
    saveSettings(false);
  });

  el('bypassNativeHdr').addEventListener('change', (e) => {
    settings.bypassNativeHdr = e.target.checked;
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });

  el('imagesEnabled').addEventListener('change', (e) => {
    settings.imagesEnabled = e.target.checked;
    if (settings.imagesEnabled) settings.hdrEnabled = true; // image HDR rides on the HDR switch
    renderUI();
    saveSettings(true);
    setTimeout(refreshTabStatus, 300);
  });

  el('retryCorsBtn').addEventListener('click', async () => {
    if (!currentTabId) return;
    try {
      await ext.tabs.sendMessage(currentTabId, { type: 'TRUEHDR_RETRY_CORS' });
      setTimeout(refreshTabStatus, 400);
    } catch (_) {}
  });

  el('resetBtn').addEventListener('click', () => {
    settings = { ...DEFAULT_SETTINGS, disabledSites: settings.disabledSites || {} };
    renderUI();
    saveSettings(true);
    setTimeout(refreshTabStatus, 140);
  });
}

init();
