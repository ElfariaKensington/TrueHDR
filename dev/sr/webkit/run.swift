// Minimal WKWebView host: loads a URL (the SR harness page) in real WebKit so the runtime's WGSL is
// compiled by WebKit's WGSL compiler and runs on Safari's WebGPU/Metal backend, isolated from the
// user's Safari. The page reports results to the driver server itself; this process only hosts it.
//   swiftc -O dev/sr/webkit/run.swift -o dev/sr/out/wkrun && dev/sr/out/wkrun http://127.0.0.1:PORT/...
import Cocoa
import WebKit

final class Host: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    let url: URL
    var window: NSWindow!
    var web: WKWebView!
    init(url: URL) { self.url = url }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .nonPersistent()
        let frame = NSRect(x: 0, y: 0, width: 640, height: 360)
        web = WKWebView(frame: frame, configuration: cfg)
        web.navigationDelegate = self
        window = NSWindow(contentRect: frame, styleMask: [.titled, .miniaturizable], backing: .buffered, defer: false)
        window.title = "TrueHDR SR harness"
        window.contentView = web
        window.isReleasedWhenClosed = false
        window.orderFrontRegardless()  // visible so WebKit does not throttle the page
        web.load(URLRequest(url: url))
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        FileHandle.standardError.write("navigation failed: \(error)\n".data(using: .utf8)!)
        exit(2)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        FileHandle.standardError.write("navigation failed: \(error)\n".data(using: .utf8)!)
        exit(2)
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        FileHandle.standardError.write("web content process terminated\n".data(using: .utf8)!)
        exit(3)
    }
}

guard CommandLine.arguments.count > 1, let url = URL(string: CommandLine.arguments[1]) else {
    FileHandle.standardError.write("usage: wkrun <url>\n".data(using: .utf8)!)
    exit(64)
}
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let host = Host(url: url)
app.delegate = host
app.run()
