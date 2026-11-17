// EDRApp: a native control app for the focus test.
// - Shows CAMetalLayer rgba16Float / extendedLinearDisplayP3 / wantsExtendedDynamicRangeContent
//   patches at linear 1, 2, 4 (it deliberately IGNORES the AppKit HDR-suppression notification),
//   so measuring it while inactive shows whether macOS itself dims EDR of background apps.
// - Logs NSApplication active/inactive + ShouldBegin/EndSuppressingHighDynamicRangeContent
//   notifications + applicationShouldSuppressHighDynamicRangeContent + screen headroom.
// Defaults (pass via `open -a EDRApp.app --args -logPath /x.log -winX 0 -winY 0`).
import AppKit
import Metal
import QuartzCore

let defaults = UserDefaults.standard
let logPath = defaults.string(forKey: "logPath") ?? "/tmp/edrapp.log"
let t0 = Date()
FileManager.default.createFile(atPath: logPath, contents: nil)
let logHandle = FileHandle(forWritingAtPath: logPath)!
func log(_ s: String) {
    let line = String(format: "%.3f ", Date().timeIntervalSince(t0)) + s + "\n"
    logHandle.seekToEndOfFile(); logHandle.write(line.data(using: .utf8)!)
}
func headroom() -> String {
    guard let s = NSScreen.main else { return "noscreen" }
    return String(format: "edr cur=%.3f pot=%.3f", s.maximumExtendedDynamicRangeColorComponentValue,
                  s.maximumPotentialExtendedDynamicRangeColorComponentValue)
}

// Resolve the exported NSString constant (its Swift spelling differs between SDKs).
func appKitNotificationName(_ symbol: String) -> Notification.Name {
    if let p = dlsym(UnsafeMutableRawPointer(bitPattern: -2), symbol) {
        let str = p.assumingMemoryBound(to: NSString.self).pointee
        return Notification.Name(str as String)
    }
    return Notification.Name(symbol)
}

final class Delegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var layers: [CAMetalLayer] = []
    let device = MTLCreateSystemDefaultDevice()!
    lazy var queue = device.makeCommandQueue()!
    let values: [Double] = [1, 2, 4]

    func applicationDidFinishLaunching(_ n: Notification) {
        let x = defaults.object(forKey: "winX") as? Double ?? 0
        let y = defaults.object(forKey: "winY") as? Double ?? 0
        let pw = 90.0
        window = NSWindow(contentRect: NSRect(x: x, y: y, width: pw * Double(values.count), height: pw),
                          styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "EDRApp"
        window.level = .floating          // stay visible (and measurable) while inactive
        window.hidesOnDeactivate = false
        let view = NSView(frame: window.contentLayoutRect)
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        window.contentView = view
        for (i, _) in values.enumerated() {
            let l = CAMetalLayer()
            l.device = device
            l.pixelFormat = .rgba16Float
            l.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)
            l.wantsExtendedDynamicRangeContent = true
            l.frame = CGRect(x: Double(i) * pw, y: 0, width: pw, height: pw)
            l.contentsScale = window.backingScaleFactor
            l.drawableSize = CGSize(width: pw * l.contentsScale, height: pw * l.contentsScale)
            view.layer?.addSublayer(l)
            layers.append(l)
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        // Patch rects in global top-left points, for hdrprobe measure.
        let cr = window.convertToScreen(view.frame)
        let screenH = NSScreen.screens.first?.frame.height ?? 0
        let rects = values.enumerated().map { i, v in
            String(format: "edrapp@%g:%.1f,%.1f,%.1f,%.1f", v, cr.minX + Double(i) * pw, screenH - cr.maxY, pw, pw)
        }
        try? rects.joined(separator: " ").write(toFile: logPath + ".rects", atomically: true, encoding: .utf8)
        let nc = NotificationCenter.default
        let names: [(Notification.Name, String)] = [
            (NSApplication.didBecomeActiveNotification, "didBecomeActive"),
            (NSApplication.didResignActiveNotification, "didResignActive"),
            (appKitNotificationName("NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification"), "SHOULD_BEGIN_SUPPRESSING_HDR"),
            (appKitNotificationName("NSApplicationShouldEndSuppressingHighDynamicRangeContentNotification"), "SHOULD_END_SUPPRESSING_HDR"),
            (NSWindow.didBecomeKeyNotification, "windowDidBecomeKey"),
            (NSWindow.didResignKeyNotification, "windowDidResignKey"),
        ]
        for (name, label) in names {
            nc.addObserver(forName: name, object: nil, queue: .main) { _ in
                log("\(label) suppress=\(NSApp.applicationShouldSuppressHighDynamicRangeContent) active=\(NSApp.isActive) \(headroom())")
            }
        }
        log("launched pid=\(getpid()) suppress=\(NSApp.applicationShouldSuppressHighDynamicRangeContent) args=\(CommandLine.arguments.dropFirst().joined(separator: " "))")
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { _ in self.draw() }
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            log("tick suppress=\(NSApp.applicationShouldSuppressHighDynamicRangeContent) active=\(NSApp.isActive) front=\(NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "-") \(headroom())")
        }
        // Command channel (cooperative activation: only the active app can hand activation over).
        //   echo safari > <logPath>.cmd   -> activate com.apple.Safari from this app
        //   echo self   > <logPath>.cmd   -> activate this app
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            let cmdPath = logPath + ".cmd"
            guard let cmd = try? String(contentsOfFile: cmdPath, encoding: .utf8) else { return }
            try? FileManager.default.removeItem(atPath: cmdPath)
            let c = cmd.trimmingCharacters(in: .whitespacesAndNewlines)
            if c == "self" {
                NSApp.activate(ignoringOtherApps: true)
            } else if let target = NSRunningApplication.runningApplications(withBundleIdentifier: c == "safari" ? "com.apple.Safari" : c).first {
                if !NSApp.isActive { NSApp.activate(ignoringOtherApps: true) }
                NSApp.yieldActivation(to: target)
                let ok = target.activate(from: NSRunningApplication.current, options: [])
                log("cmd \(c) activate(from:) -> \(ok)")
            } else {
                log("cmd \(c): no such app")
            }
        }
    }

    func draw() {
        for (i, l) in layers.enumerated() {
            guard let d = l.nextDrawable(), let cb = queue.makeCommandBuffer() else { continue }
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = d.texture
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            let v = values[i]
            rp.colorAttachments[0].clearColor = MTLClearColor(red: v, green: v, blue: v, alpha: 1)
            cb.makeRenderCommandEncoder(descriptor: rp)?.endEncoding()
            cb.present(d)
            cb.commit()
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = Delegate()
app.delegate = delegate
app.run()
