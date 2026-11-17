#!/usr/bin/env python3
"""Drives the EDR focus experiment end to end (macOS, built-in XDR display).

  python3 dev/focus-test/run_experiment.py [variant-query ...]

For each variant (URL query for index.html, e.g. "" or "reconfigure=1"):
  1. opens the test page in a dedicated window of com.apple.Safari (closed at the end),
  2. cycles focus: Safari active -> EDRApp active (Safari inactive, window still visible)
     -> Safari active again [-> another Safari window key while Safari stays active],
  3. in each phase takes an HDR screenshot (`screencapture -x -H -m`) and measures every
     patch's displayed linear value relative to SDR white with `hdrprobe measure`,
     plus NSScreen EDR headroom, the frontmost app, and the page's own state.
Results: dev/focus-test/logs/results.json and a printed table.

Needs: tools/build.sh first, Screen Recording permission for the terminal host, Automation
permission for Safari. macOS cooperative activation only lets the *active* app hand
activation over, so app switches go through EDRApp's command file (EDRApp activates Safari
with NSRunningApplication.activate(from:)), and every phase records the real frontmost app.
Phases whose frontmost app is not the intended one are flagged INVALID (!) in the table.
Env: SETTLE=<s> wait per phase (3), OTHER_WINDOW=0 skip the second-Safari-window phase,
APPEND=1 append to results.json, PHASES=a,b only run the named phases,
CONTENT_FOCUS=1 load the page so web content has focus (then blur/focus events fire).
"""
import json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, '.build')
LOGS = os.path.join(HERE, 'logs')
os.makedirs(LOGS, exist_ok=True)
PORT = 8765
BASE = f'http://localhost:{PORT}/dev/focus-test/index.html'
PROBE = os.path.join(BUILD, 'hdrprobe')
EDRAPP = os.path.join(BUILD, 'EDRApp.app')
EDRLOG = os.path.join(LOGS, 'edrapp.log')
SETTLE = float(os.environ.get('SETTLE', '3'))
SAFARI = 'com.apple.Safari'
EDR_ID = 'dev.truehdr.focustest.edrapp'


def sh(*cmd, check=True, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout).stdout.strip()


def osa(script):
    return sh('osascript', '-e', script)


# Safari 27's terminology for tab/URL fails to compile ("A property can't go after this
# identifier"), so raw Apple-event codes are used: cTab = current tab, pURL = URL.
def set_url(win_id, url):
    osa(f'tell application id "{SAFARI}" to set «class pURL» of «class cTab» of window id {win_id} to "{url}"')


def new_window(url, bounds):
    wid = osa(f'tell application id "{SAFARI}"\nmake new document\ndelay 0.6\nreturn id of window 1\nend tell')
    set_url(wid, url)
    osa(f'tell application id "{SAFARI}" to set bounds of window id {wid} to {{{", ".join(map(str, bounds))}}}')
    return wid


def front():
    return sh(PROBE, 'front')


def edr_cmd(c):
    with open(EDRLOG + '.cmd', 'w') as f:
        f.write(c)


def activate(bundle):
    for _ in range(4):
        if bundle == EDR_ID:
            sh('open', '-a', EDRAPP)
            edr_cmd('self')
        else:
            if front() != EDR_ID:
                sh('open', '-a', EDRAPP); edr_cmd('self'); time.sleep(0.6)
            edr_cmd('safari')
        time.sleep(0.8)
        if front() == bundle:
            return True
    return False


def page_state():
    with urllib.request.urlopen(f'http://localhost:{PORT}/focus-log/state', timeout=3) as r:
        return json.loads(r.read())


def measure(tag, extra_rects, expect_front):
    st = page_state()
    fr = front()
    path = os.path.join(LOGS, f'cap-{tag}.heic')
    sh('screencapture', '-x', '-H', '-m', path)
    fr2 = front()
    rects = [f'{k}:{",".join(map(str, v))}' for k, v in st.get('probes', {}).items()] + extra_rects
    out = sh(PROBE, 'measure', path, *rects)
    vals = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1].startswith('mean='):
            vals[parts[0]] = float(parts[1][5:])
    return {
        'tag': tag, 'front': fr, 'valid': fr == fr2 == expect_front,
        'screens': sh(PROBE, 'screens'), 'values': vals,
        'page': {k: st.get(k) for k in ('hasFocus', 'visibility', 'dynamicRangeHigh', 'videoDynamicRangeHigh', 'rafPerSec', 'rvfcPerSec', 'videoPaused')},
    }


def main():
    variants = sys.argv[1:] or ['']
    server = subprocess.Popen([sys.executable, os.path.join(HERE, 'server.py'), str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.8)
    m = re.search(r'frame=\{\{[-\d.]+, [-\d.]+\}, \{([\d.]+), ([\d.]+)\}\}', sh(PROBE, 'screens'))
    screen_w, screen_h = float(m.group(1)), float(m.group(2))
    subprocess.run(['pkill', '-f', 'EDRApp.app/Contents/MacOS/EDRApp']); time.sleep(0.3)
    # Floating EDRApp window over the empty lower part of the test window (no page patches there)
    sh('open', '-n', '-a', EDRAPP, '--args', '-logPath', EDRLOG, '-winX', '900', '-winY', str(screen_h - 900))
    time.sleep(1.5)
    edr_rects = open(EDRLOG + '.rects').read().split()
    focus_content = os.environ.get('CONTENT_FOCUS') == '1'
    win_id = new_window('about:blank' if focus_content else BASE, (0, 30, int(screen_w * 0.62), int(screen_h - 40)))
    results = []
    extra_windows = []
    try:
        for v in variants:
            name = v or 'base'
            url = BASE + ('?' + v if v else '')
            osa(f'tell application id "{SAFARI}" to set index of window id {win_id} to 1')
            activate(SAFARI)
            if focus_content:
                # A URL opened through LaunchServices gets web-content focus (document.hasFocus()
                # true); AppleScript `set URL` leaves the address bar focused. It opens as a new
                # tab of the front window, which is the dedicated test window.
                sh('open', '-a', 'Safari', url + ('&' if v else '?') + 'tab=' + str(time.time()))
                time.sleep(1.5)
                n = osa(f'tell application id "{SAFARI}" to count «class bTab» of window id {win_id}')
                if n == '1':  # opened elsewhere (new window): remember it so it gets closed too
                    extra_windows.append(osa(f'tell application id "{SAFARI}"\nrepeat with w in windows\nif («class pURL» of «class cTab» of w) contains "focus-test/index.html" then return id of w\nend repeat\nend tell'))
                print(f'[{v or "base"}] content-focus tab opened; test window tabs={n} extra={extra_windows}', flush=True)
            else:
                set_url(win_id, url)
            time.sleep(4)
            phases = [('safari_active', SAFARI), ('safari_inactive', EDR_ID), ('safari_inactive_8s', None), ('safari_reactivated', SAFARI)]
            if os.environ.get('OTHER_WINDOW', '1') == '1' and not v:
                phases += [('safari_other_window_key', 'other'), ('safari_test_window_key', 'back')]
            if os.environ.get('PHASES'):
                want = os.environ['PHASES'].split(',')
                phases = [ph for ph in phases if ph[0] in want]
            other = None
            expect = SAFARI
            for tag, act in phases:
                if act == 'other':
                    other = new_window('about:blank', (int(screen_w * 0.66), 30, int(screen_w - 10), 400))
                    expect = SAFARI
                elif act == 'back':
                    osa(f'tell application id "{SAFARI}" to close window id {other}')
                    osa(f'tell application id "{SAFARI}" to set index of window id {win_id} to 1')
                    expect = SAFARI
                elif act:
                    activate(act); expect = act
                time.sleep(SETTLE if tag != 'safari_inactive_8s' else 5)
                r = measure(f'{name}-{tag}', edr_rects, expect)
                r['variant'] = name
                results.append(r)
                cur = r['screens'].split('current=')[1].split(' ')[0]
                print(f'[{name}] {tag}: front={r["front"]} valid={r["valid"]} edrCurrent={cur} page={r["page"]}', flush=True)
    finally:
        for w in [win_id] + extra_windows:
            try:
                osa(f'tell application id "{SAFARI}" to close window id {w}')
            except Exception:
                pass
        server.terminate()
        subprocess.run(['pkill', '-f', 'EDRApp.app/Contents/MacOS/EDRApp'])
    out = os.path.join(LOGS, 'results.json')
    prev = json.load(open(out)) if os.path.exists(out) and os.environ.get('APPEND') == '1' else []
    json.dump(prev + results, open(out, 'w'), indent=1)
    keys = sorted({k for r in results for k in r['values']})
    print('\n' + 'patch'.ljust(20) + ''.join((r['tag'].split('-', 1)[1][:21] + ('' if r['valid'] else '!')).rjust(23) for r in results))
    for k in keys:
        print(k.ljust(20) + ''.join(f'{r["values"].get(k, float("nan")):23.3f}' for r in results))
    print('(! = INVALID phase: frontmost app was not the intended one)')


if __name__ == '__main__':
    main()
