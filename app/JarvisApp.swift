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
let outboxFile = configDirectory.appendingPathComponent("window-outbox.jsonl")
let encoder = JSONEncoder()
let dateFormatter = ISO8601DateFormatter()

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var timer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard terminateIfAnotherJarvisIsRunning() == false else {
            return
        }

        let trusted = requestAccessibilityIfNeeded()
        log("Jarvis native app started accessibility_trusted=\(trusted)")

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

func terminateIfAnotherJarvisIsRunning() -> Bool {
    let currentPID = ProcessInfo.processInfo.processIdentifier
    let bundleIdentifier = Bundle.main.bundleIdentifier

    let alreadyRunning = NSWorkspace.shared.runningApplications.contains { application in
        application.processIdentifier != currentPID
            && application.bundleIdentifier == bundleIdentifier
    }

    if alreadyRunning {
        log("Jarvis native app duplicate launch ignored")
        NSApp.terminate(nil)
        return true
    }

    return false
}

func requestAccessibilityIfNeeded() -> Bool {
    if AXIsProcessTrusted() {
        return true
    }

    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
    return AXIsProcessTrustedWithOptions(options as CFDictionary)
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

    flushOutbox(to: receiverURL) { flushed, completed in
        if completed == false {
            appendToOutbox(data)
            return
        }

        postSnapshot(data, to: receiverURL) { success in
            if success {
                if flushed > 0 {
                    log("send_flushed \(flushed)")
                }
            } else {
                appendToOutbox(data)
            }
        }
    }
}

func flushOutbox(to receiverURL: URL, completion: @escaping (Int, Bool) -> Void) {
    let queuedEvents = readOutbox()
    if queuedEvents.isEmpty {
        completion(0, true)
        return
    }

    sendQueuedEvents(queuedEvents, index: 0, sent: 0, to: receiverURL, completion: completion)
}

func sendQueuedEvents(
    _ events: [Data],
    index: Int,
    sent: Int,
    to receiverURL: URL,
    completion: @escaping (Int, Bool) -> Void
) {
    if index >= events.count {
        clearOutbox()
        completion(sent, true)
        return
    }

    postSnapshot(events[index], to: receiverURL) { success in
        if success {
            sendQueuedEvents(events, index: index + 1, sent: sent + 1, to: receiverURL, completion: completion)
            return
        }

        replaceOutbox(Array(events[index...]))
        completion(sent, false)
    }
}

func postSnapshot(_ data: Data, to receiverURL: URL, completion: @escaping (Bool) -> Void) {
    var request = URLRequest(url: receiverURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.httpBody = data
    request.timeoutInterval = 3

    URLSession.shared.dataTask(with: request) { _, response, error in
        if let error {
            log("send_error \(error)")
            completion(false)
            return
        }

        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
            log("send_error HTTP \(httpResponse.statusCode)")
            completion(false)
            return
        }

        completion(true)
    }.resume()
}

func appendToOutbox(_ data: Data) {
    do {
        try FileManager.default.createDirectory(
            at: configDirectory,
            withIntermediateDirectories: true
        )

        var line = data
        line.append(Data("\n".utf8))

        if FileManager.default.fileExists(atPath: outboxFile.path) {
            let handle = try FileHandle(forWritingTo: outboxFile)
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: outboxFile)
        }
    } catch {
        log("outbox_append_error \(error)")
    }
}

func readOutbox() -> [Data] {
    guard let contents = try? String(contentsOf: outboxFile, encoding: .utf8) else {
        return []
    }

    return contents
        .split(separator: "\n")
        .compactMap { String($0).data(using: .utf8) }
}

func replaceOutbox(_ events: [Data]) {
    if events.isEmpty {
        clearOutbox()
        return
    }

    do {
        try FileManager.default.createDirectory(
            at: configDirectory,
            withIntermediateDirectories: true
        )

        var data = Data()
        for event in events {
            data.append(event)
            data.append(Data("\n".utf8))
        }
        try data.write(to: outboxFile)
    } catch {
        log("outbox_replace_error \(error)")
    }
}

func clearOutbox() {
    do {
        if FileManager.default.fileExists(atPath: outboxFile.path) {
            try FileManager.default.removeItem(at: outboxFile)
        }
    } catch {
        log("outbox_clear_error \(error)")
    }
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
