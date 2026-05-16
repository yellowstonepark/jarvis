import AppKit
import ApplicationServices
import Foundation

struct WindowSnapshot: Encodable {
    let app_name: String
    let window_title: String?
    let observed_at: String
    let source: String
}

let logDirectory = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library")
    .appendingPathComponent("Logs")
    .appendingPathComponent("Jarvis")
let logFile = logDirectory.appendingPathComponent("jarvis.log")
let configDirectory = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".jarvis")
let receiverURLFile = configDirectory.appendingPathComponent("receiver-url")
let encoder = JSONEncoder()
let dateFormatter = ISO8601DateFormatter()

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var timer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        requestAccessibilityIfNeeded()
        log("Jarvis native app started")

        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            recordActiveWindow()
        }
        timer?.fire()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        timer?.invalidate()
        log("Jarvis native app stopping")
        return .terminateNow
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()

func requestAccessibilityIfNeeded() {
    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
    _ = AXIsProcessTrustedWithOptions(options as CFDictionary)
}

func recordActiveWindow() {
    do {
        let snapshot = try activeWindowSnapshot()
        let data = try encoder.encode(snapshot)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        log("active_window \(json)")
        sendSnapshot(data)
    } catch {
        log("active_window_error \(error)")
    }
}

func activeWindowSnapshot() throws -> WindowSnapshot {
    guard let app = NSWorkspace.shared.frontmostApplication else {
        throw JarvisError.noFrontmostApplication
    }

    let appName = app.localizedName ?? app.bundleIdentifier ?? "Unknown"
    let windowTitle = focusedWindowTitle(pid: app.processIdentifier)

    return WindowSnapshot(
        app_name: appName,
        window_title: windowTitle,
        observed_at: dateFormatter.string(from: Date()),
        source: "jarvis-app"
    )
}

func focusedWindowTitle(pid: pid_t) -> String? {
    let appElement = AXUIElementCreateApplication(pid)
    var focusedWindow: CFTypeRef?

    let focusedResult = AXUIElementCopyAttributeValue(
        appElement,
        kAXFocusedWindowAttribute as CFString,
        &focusedWindow
    )

    guard focusedResult == .success, let focusedWindow else {
        return nil
    }

    var title: CFTypeRef?
    let titleResult = AXUIElementCopyAttributeValue(
        focusedWindow as! AXUIElement,
        kAXTitleAttribute as CFString,
        &title
    )

    guard titleResult == .success else {
        return nil
    }

    return title as? String
}

func sendSnapshot(_ data: Data) {
    guard let receiverURL = configuredReceiverURL() else {
        return
    }

    var request = URLRequest(url: receiverURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.httpBody = data

    URLSession.shared.dataTask(with: request) { _, response, error in
        if let error {
            log("send_error \(error)")
            return
        }

        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
            log("send_error HTTP \(httpResponse.statusCode)")
        }
    }.resume()
}

func configuredReceiverURL() -> URL? {
    guard let rawValue = try? String(contentsOf: receiverURLFile, encoding: .utf8) else {
        return nil
    }

    let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmedValue.isEmpty {
        return nil
    }

    return URL(string: trimmedValue)
}

func log(_ message: String) {
    do {
        try FileManager.default.createDirectory(
            at: logDirectory,
            withIntermediateDirectories: true
        )

        let line = "\(Date()) \(message)\n"
        let data = Data(line.utf8)

        if FileManager.default.fileExists(atPath: logFile.path) {
            let handle = try FileHandle(forWritingTo: logFile)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } else {
            try data.write(to: logFile)
        }
    } catch {
        NSLog("Jarvis log error: \(String(describing: error))")
    }
}

enum JarvisError: Error {
    case noFrontmostApplication
}
