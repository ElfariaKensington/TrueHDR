// hdrprobe: measure EDR state from outside Safari.
//
//   hdrprobe screens
//       Print current / potential / reference EDR headroom of every NSScreen.
//   hdrprobe measure <capture.heic> <name:x,y,w,h> [...]
//       Decode an HDR screenshot (`screencapture -x -H -m file`) with expandToHDR
//       and print the mean linear (extended linear Display P3) value of each rect.
//       Rects are in global screen points, top-left origin (CSS screen coords).
//       The central 50% of each rect is sampled.
//   hdrprobe make-hdr-image <out.heic> <v1,v2,...>
//       Write a gain-map HEIC with vertical patches of the given linear values
//       (1.0 = SDR white), for the <img> test.
import AppKit
import CoreImage

let args = CommandLine.arguments

func screens() {
    for s in NSScreen.screens {
        let id = (s.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.intValue ?? -1
        print(String(format: "screen id=%d name=\"%@\" current=%.3f potential=%.3f reference=%.3f frame=%@",
                     id, s.localizedName,
                     s.maximumExtendedDynamicRangeColorComponentValue,
                     s.maximumPotentialExtendedDynamicRangeColorComponentValue,
                     s.maximumReferenceExtendedDynamicRangeColorComponentValue,
                     NSStringFromRect(s.frame)))
    }
}

func measure(_ path: String, _ specs: [String]) {
    let url = URL(fileURLWithPath: path)
    guard let img = CIImage(contentsOf: url, options: [.expandToHDR: true, .applyOrientationProperty: true]) else {
        print("error: cannot decode \(path)"); exit(2)
    }
    let ext = img.extent
    let screenW = NSScreen.screens.first?.frame.width ?? ext.width
    let scale = ext.width / screenW
    let cs = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!
    let ctx = CIContext(options: [.workingColorSpace: cs, .workingFormat: CIFormat.RGBAh])
    print(String(format: "image %.0fx%.0f scale=%.3f", ext.width, ext.height, scale))
    for spec in specs {
        let parts = spec.split(separator: ":", maxSplits: 1)
        guard parts.count == 2 else { continue }
        let nums = parts[1].split(separator: ",").compactMap { Double($0) }
        guard nums.count == 4 else { continue }
        // central 50%
        let x = nums[0] + nums[2] * 0.25, y = nums[1] + nums[3] * 0.25
        let w = max(1, nums[2] * 0.5), h = max(1, nums[3] * 0.5)
        // CI origin is bottom-left
        let px = CGRect(x: x * scale, y: ext.height - (y + h) * scale, width: w * scale, height: h * scale).integral
        let wi = Int(px.width), hi = Int(px.height)
        var buf = [Float](repeating: 0, count: wi * hi * 4)
        ctx.render(img, toBitmap: &buf, rowBytes: wi * 16, bounds: px, format: .RGBAf, colorSpace: cs)
        var r = 0.0, g = 0.0, b = 0.0, mx = 0.0
        for i in 0..<(wi * hi) {
            r += Double(buf[i*4]); g += Double(buf[i*4+1]); b += Double(buf[i*4+2])
            mx = max(mx, Double(max(buf[i*4], buf[i*4+1], buf[i*4+2])))
        }
        let n = Double(wi * hi)
        print(String(format: "%@ mean=%.3f r=%.3f g=%.3f b=%.3f max=%.3f", String(parts[0]), (r+g+b)/(3*n), r/n, g/n, b/n, mx))
    }
}

func makeHDRImage(_ path: String, _ values: [Double]) {
    let w = 160 * values.count, h = 160
    let cs = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!
    var data = [Float](repeating: 1, count: w * h * 4)
    for y in 0..<h { for x in 0..<w {
        let v = Float(values[x / 160])
        let i = (y * w + x) * 4
        data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 1
    } }
    let d = Data(bytes: &data, count: data.count * 4)
    let img = CIImage(bitmapData: d, bytesPerRow: w * 16, size: CGSize(width: w, height: h), format: .RGBAf, colorSpace: cs)
    let ctx = CIContext()
    let sdr = img.applyingFilter("CIToneMapHeadroom", parameters: ["inputSourceHeadroom": Float(values.max() ?? 1), "inputTargetHeadroom": 1.0])
    do {
        try ctx.writeHEIFRepresentation(of: sdr, to: URL(fileURLWithPath: path), format: .RGBA8,
                                        colorSpace: CGColorSpace(name: CGColorSpace.displayP3)!,
                                        options: [.hdrImage: img])
        print("wrote \(path)")
    } catch { print("error: \(error)"); exit(3) }
}

switch args.count > 1 ? args[1] : "" {
case "screens": screens()
case "front": print(NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "-")
case "measure" where args.count > 2: measure(args[2], Array(args.dropFirst(3)))
case "make-hdr-image" where args.count > 3: makeHDRImage(args[2], args[3].split(separator: ",").compactMap { Double($0) })
default:
    print("usage: hdrprobe screens | measure <file> name:x,y,w,h ... | make-hdr-image <out.heic> v1,v2,..."); exit(1)
}
