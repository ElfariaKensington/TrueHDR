//
//  SafariWebExtensionHandler.swift
//  TrueHDR Extension
//
//  Created by Elfaria on 15.11.2026.
//

import AppKit
import SafariServices
import os.log

class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {

    func beginRequest(with context: NSExtensionContext) {
        let request = context.inputItems.first as? NSExtensionItem

        let profile: UUID?
        if #available(iOS 17.0, macOS 14.0, *) {
            profile = request?.userInfo?[SFExtensionProfileKey] as? UUID
        } else {
            profile = request?.userInfo?["profile"] as? UUID
        }

        let message: Any?
        if #available(iOS 15.0, macOS 11.0, *) {
            message = request?.userInfo?[SFExtensionMessageKey]
        } else {
            message = request?.userInfo?["message"]
        }

        let reply: [String: Any]
        if let dict = message as? [String: Any], dict["type"] as? String == "APP_STATE" {
            reply = Self.appState()
        } else {
            os_log(.default, "Received message from browser.runtime.sendNativeMessage: %@ (profile: %@)", String(describing: message), profile?.uuidString ?? "none")
            reply = [ "echo": message as Any ]
        }

        let response = NSExtensionItem()
        if #available(iOS 15.0, macOS 11.0, *) {
            response.userInfo = [ SFExtensionMessageKey: reply ]
        } else {
            response.userInfo = [ "message": reply ]
        }

        context.completeRequest(returningItems: [ response ], completionHandler: nil)
    }

    // Stock WebKit clamps page EDR to 1.6x while Safari is not the frontmost app; the engine then
    // refits its tone curve. Safari web apps (com.apple.Safari.WebApp.*) count as Safari. The clamp
    // is lifted by a patched WebKit (SafariForWebKitDevelopment / the PatchedSafari launcher) or by
    // tools/edr-unsuppress, which can only be injected into Safari with SIP off.
    private static func appState() -> [String: Any] {
        let front = NSWorkspace.shared.frontmostApplication
        let bid = front?.bundleIdentifier ?? ""
        let patched = NSWorkspace.shared.runningApplications.contains { app in
            app.bundleIdentifier == "com.elfaria.PatchedSafari"
                || app.executableURL?.lastPathComponent == "SafariForWebKitDevelopment"
        }
        return [
            "frontmost": bid,
            "safariFrontmost": bid.hasPrefix("com.apple.Safari"),
            "sipOff": sipAllowsDyldInjection(),
            "unsuppressedWebKit": patched || sipAllowsDyldInjection()
        ]
    }

    // csr_check(CSR_ALLOW_UNRESTRICTED_FS) == 0 when SIP is disabled ("csrutil disable").
    private static func sipAllowsDyldInjection() -> Bool {
        typealias CsrCheck = @convention(c) (UInt32) -> Int32
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "csr_check") else { return false }
        return unsafeBitCast(sym, to: CsrCheck.self)(1 << 1) == 0
    }

}
