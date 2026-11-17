#!/usr/bin/env python3
"""Measures how long after an app switch Safari's EDR clamp on a WebGPU canvas lands / lifts.

  python3 dev/focus-test/tools/latency.py [query]

Opens the test page (address-bar focus, so no page workaround fires), then repeatedly
captures the gpuMain@4 patch with `screencapture -x -H -R` right after deactivating Safari
(EDRApp activated) and again after re-activating it. Prints (seconds since switch, value).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_experiment as rx  # noqa: E402


def series(rect, label, sw, seconds=2.5):
    t0 = time.time()
    out = []
    i = 0
    while time.time() - t0 < seconds:
        path = os.path.join(rx.LOGS, f'lat-{label}-{i}.heic')
        t = time.time() - t0
        rx.sh('screencapture', '-x', '-H', '-R' + ','.join(str(int(v)) for v in rect), path)
        # hdrprobe maps points with scale = imageWidth / screenWidth; for a rect-sized capture
        # this rect covers the whole image (its central 50% is sampled).
        line = rx.sh(rx.PROBE, 'measure', path, f'p:0,0,{sw},{rect[3] * sw / rect[2]:.1f}')
        val = line.splitlines()[-1].split('mean=')[1].split()[0] if 'mean=' in line else 'nan'
        out.append((round(t, 3), val))
        os.remove(path)
        i += 1
    print(label, out, flush=True)


def main():
    import subprocess, re
    q = sys.argv[1] if len(sys.argv) > 1 else ''
    server = subprocess.Popen([sys.executable, os.path.join(rx.HERE, 'server.py'), str(rx.PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.8)
    m = re.search(r'frame=\{\{[-\d.]+, [-\d.]+\}, \{([\d.]+), ([\d.]+)\}\}', rx.sh(rx.PROBE, 'screens'))
    sw, shh = float(m.group(1)), float(m.group(2))
    subprocess.run(['pkill', '-f', 'EDRApp.app/Contents/MacOS/EDRApp']); time.sleep(0.3)
    rx.sh('open', '-n', '-a', rx.EDRAPP, '--args', '-logPath', rx.EDRLOG)
    time.sleep(1.5)
    wid = rx.new_window(rx.BASE + ('?' + q if q else ''), (0, 30, int(sw * 0.62), int(shh - 40)))
    try:
        rx.osa(f'tell application id "{rx.SAFARI}" to set index of window id {wid} to 1')
        rx.activate(rx.SAFARI)
        time.sleep(4)
        r = rx.page_state()['probes']['gpuMain@4']
        rect = [int(r[0] + r[2] * 0.3), int(r[1] + r[3] * 0.3), int(r[2] * 0.4), int(r[3] * 0.4)]
        for _ in range(2):
            rx.sh('open', '-a', rx.EDRAPP); rx.edr_cmd('self')
            series(rect, 'deactivate', sw)
            time.sleep(1)
            rx.edr_cmd('safari')
            series(rect, 'reactivate', sw)
            time.sleep(1)
    finally:
        rx.osa(f'tell application id "{rx.SAFARI}" to close window id {wid}')
        server.terminate()
        subprocess.run(['pkill', '-f', 'EDRApp.app/Contents/MacOS/EDRApp'])


if __name__ == '__main__':
    main()
