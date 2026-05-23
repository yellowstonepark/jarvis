import AppKit
import ApplicationServices
import AVFoundation
import Foundation
import Speech
import UserNotifications

let appName = "JarvisHotkey"
let capsLockKeyCode: CGKeyCode = 57
let logDirectory = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library")
    .appendingPathComponent("Logs")
    .appendingPathComponent("Jarvis")
let logFile = logDirectory.appendingPathComponent("jarvis-hotkey.log")
let configDirectory = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".jarvis")
let receiverURLFile = configDirectory.appendingPathComponent("receiver-url")
let ttsURLFile = configDirectory.appendingPathComponent("tts-url")
let ttsVoiceFile = configDirectory.appendingPathComponent("tts-voice")
let ttsVolumeFile = configDirectory.appendingPathComponent("tts-volume")
let defaultTTSURL = "http://127.0.0.1:28766/v1/speak"
let defaultTTSVoice = "am_adam"
// afplay volume 0–255; ~52 ≈ 20% system-scale playback.
let defaultAfplayVolume = 52

final class HotkeyAppDelegate: NSObject, NSApplicationDelegate {
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var globalMonitor: Any?
    private let voiceController = VoiceController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard terminateIfAnotherHotkeyAppIsRunning() == false else {
            return
        }

        log("JarvisHotkey started; shortcut Caps Lock hold")
        requestNotificationPermission()
        requestSpeechPermission()
        requestMicrophonePermission()
        startEventListener()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        stopEventTap()
        voiceController.cancel()
        log("JarvisHotkey stopping")
        return .terminateNow
    }

    private func startEventListener() {
        if startEventTap() {
            return
        }

        _ = CGRequestListenEventAccess()
        startGlobalMonitorFallback()
    }

    @discardableResult
    private func startEventTap() -> Bool {
        let mask = (1 << CGEventType.keyDown.rawValue)
            | (1 << CGEventType.keyUp.rawValue)
            | (1 << CGEventType.flagsChanged.rawValue)

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(mask),
            callback: eventTapCallback,
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        ) else {
            log("failed to create listen-only CGEventTap; toggle Input Monitoring off/on and relaunch")
            notify(title: "Jarvis Hotkey", body: "Trying fallback keyboard monitor.")
            return false
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        if let runLoopSource {
            CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        }
        CGEvent.tapEnable(tap: tap, enable: true)
        log("event tap started")
        notify(title: "Jarvis Hotkey", body: "Hold Caps Lock to talk to Jarvis.")
        return true
    }

    private func startGlobalMonitorFallback() {
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(keyCode: CGKeyCode(event.keyCode), capsLockIsDown: event.modifierFlags.contains(.capsLock))
        }

        if globalMonitor == nil {
            log("failed to create global monitor fallback")
            notify(title: "Jarvis Hotkey", body: "Could not monitor Caps Lock. Toggle Input Monitoring off/on and relaunch.")
            return
        }

        log("global monitor fallback started")
        notify(title: "Jarvis Hotkey", body: "Hold Caps Lock to talk to Jarvis.")
    }

    private func stopEventTap() {
        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
        }
        if let runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        }
        runLoopSource = nil
        eventTap = nil
        if let globalMonitor {
            NSEvent.removeMonitor(globalMonitor)
        }
        globalMonitor = nil
    }

    fileprivate func handle(eventType: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if eventType == .tapDisabledByTimeout || eventType == .tapDisabledByUserInput {
            if let eventTap {
                CGEvent.tapEnable(tap: eventTap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        let keyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
        guard eventType == .flagsChanged, keyCode == capsLockKeyCode else {
            return Unmanaged.passUnretained(event)
        }

        handleFlagsChanged(keyCode: keyCode, capsLockIsDown: event.flags.contains(.maskAlphaShift))
        return Unmanaged.passUnretained(event)
    }

    private func handleFlagsChanged(keyCode: CGKeyCode, capsLockIsDown: Bool) {
        guard keyCode == capsLockKeyCode else {
            return
        }

        if capsLockIsDown, voiceController.isRecording == false {
            voiceController.startRecording()
        }

        if capsLockIsDown == false, voiceController.isRecording {
            voiceController.stopRecordingAndAskJarvis()
        }
    }
}

private let eventTapCallback: CGEventTapCallBack = { _, type, event, userInfo in
    guard let userInfo else {
        return Unmanaged.passUnretained(event)
    }
    let delegate = Unmanaged<HotkeyAppDelegate>.fromOpaque(userInfo).takeUnretainedValue()
    return delegate.handle(eventType: type, event: event)
}

final class VoiceController: NSObject {
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var audioRecorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var recognitionTask: SFSpeechRecognitionTask?
    private(set) var isRecording = false

    func startRecording() {
        guard isRecording == false else {
            return
        }

        AfplaySpeechPlayer.shared.stopPlayback()

        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            notify(title: "Jarvis Hotkey", body: "Speech Recognition permission is required.")
            log("speech permission not authorized")
            return
        }

        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            break
        case .notDetermined:
            requestMicrophonePermission()
            return
        default:
            notify(title: "Jarvis Hotkey", body: "Microphone permission is required.")
            log("microphone permission not authorized")
            return
        }

        recognitionTask?.cancel()
        recognitionTask = nil
        stopRecorder(deleteFile: true)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("jarvis-recording-\(UUID().uuidString).wav")
        recordingURL = url

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.prepareToRecord()
            guard recorder.record() else {
                log("audio_recorder_start_failed")
                notify(title: "Jarvis Hotkey", body: "Could not start microphone recording.")
                stopRecorder(deleteFile: true)
                return
            }
            audioRecorder = recorder
            isRecording = true
            notify(title: "Jarvis", body: "Listening...")
            log("recording started file=\(url.lastPathComponent)")
        } catch {
            log("audio_recorder_start_error \(error)")
            notify(title: "Jarvis Hotkey", body: "Could not start microphone recording.")
            stopRecorder(deleteFile: true)
        }
    }

    func stopRecordingAndAskJarvis() {
        guard isRecording else {
            return
        }

        isRecording = false
        audioRecorder?.stop()
        audioRecorder = nil
        log("microphone released")

        guard let url = recordingURL else {
            notify(title: "Jarvis", body: "I didn't catch anything.")
            return
        }

        log("recording stopped file=\(url.lastPathComponent)")
        transcribeRecording(at: url)
    }

    func cancel() {
        recognitionTask?.cancel()
        recognitionTask = nil
        isRecording = false
        stopRecorder(deleteFile: true)
        log("microphone released")
    }

    private func stopRecorder(deleteFile: Bool) {
        audioRecorder?.stop()
        audioRecorder = nil
        if deleteFile, let url = recordingURL {
            try? FileManager.default.removeItem(at: url)
        }
        recordingURL = nil
    }

    private func transcribeRecording(at url: URL) {
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.shouldReportPartialResults = false
        request.requiresOnDeviceRecognition = false

        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else {
                return
            }

            if let error {
                log("speech_recognition_error \(error)")
            }

            guard let result, result.isFinal else {
                return
            }

            let transcript = result.bestTranscription.formattedString
                .trimmingCharacters(in: .whitespacesAndNewlines)
            self.recognitionTask = nil
            try? FileManager.default.removeItem(at: url)
            self.recordingURL = nil

            DispatchQueue.main.async {
                self.finishAsk(transcript: transcript)
            }
        }
    }

    private func finishAsk(transcript: String) {
        guard transcript.isEmpty == false else {
            notify(title: "Jarvis", body: "I didn't catch anything.")
            return
        }

        notify(title: "Jarvis", body: "Thinking: \(transcript)")
        askJarvis(prompt: transcript) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let answer):
                    speakAnswer(answer)
                    notify(title: "Jarvis", body: answer.truncatedForNotification)
                case .failure(let error):
                    notify(title: "Jarvis error", body: String(describing: error).truncatedForNotification)
                }
            }
        }
    }
}

func askJarvis(prompt: String, completion: @escaping (Result<String, Error>) -> Void) {
    guard let askURL = configuredAskURL() else {
        completion(.failure(JarvisHotkeyError.missingReceiverURL))
        return
    }

    let payload: [String: Any] = [
        "prompt": prompt,
        "with_window_history": true,
        "history_minutes": 30,
        "max_history_events": 80,
        "timezone": TimeZone.current.identifier,
    ]

    do {
        var request = URLRequest(url: askURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        request.timeoutInterval = 90

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                log("ask_error \(error)")
                completion(.failure(error))
                return
            }

            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
                let detail = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                log("ask_http_error \(httpResponse.statusCode) \(detail)")
                completion(.failure(JarvisHotkeyError.httpStatus(httpResponse.statusCode)))
                return
            }

            let answer = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            log("ask_answer \(answer)")
            completion(.success(answer.isEmpty ? "Jarvis returned an empty response." : answer))
        }.resume()
    } catch {
        completion(.failure(error))
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndianValue = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndianValue) { buffer in
            append(contentsOf: buffer)
        }
    }
}

func wrapPCMInWAV(_ pcm: Data, sampleRate: UInt32 = 24_000) -> Data {
    var wav = Data()
    let channels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
    let blockAlign = channels * (bitsPerSample / 8)
    let dataSize = UInt32(pcm.count)
    let riffSize = 36 + dataSize

    wav.append(contentsOf: Array("RIFF".utf8))
    wav.appendLittleEndian(riffSize)
    wav.append(contentsOf: Array("WAVE".utf8))
    wav.append(contentsOf: Array("fmt ".utf8))
    wav.appendLittleEndian(UInt32(16))
    wav.appendLittleEndian(UInt16(1))
    wav.appendLittleEndian(channels)
    wav.appendLittleEndian(sampleRate)
    wav.appendLittleEndian(byteRate)
    wav.appendLittleEndian(blockAlign)
    wav.appendLittleEndian(bitsPerSample)
    wav.append(contentsOf: Array("data".utf8))
    wav.appendLittleEndian(dataSize)
    wav.append(pcm)
    return wav
}

/// Plays TTS via /usr/bin/afplay so Jarvis never keeps an AVAudioEngine output graph alive.
final class AfplaySpeechPlayer {
    static let shared = AfplaySpeechPlayer()

    private var pcmData = Data()
    private var playerProcess: Process?
    private var tempWavURL: URL?

    func begin() {
        reset()
    }

    func append(pcm data: Data) {
        guard data.isEmpty == false else {
            return
        }
        pcmData.append(data)
    }

    func endStream(error: Error?) {
        if let error {
            log("tts_stream_error \(error)")
            reset()
            return
        }

        guard pcmData.isEmpty == false else {
            log("tts_empty_audio")
            reset()
            return
        }

        let wav = wrapPCMInWAV(pcmData)
        pcmData = Data()
        playWavData(wav)
    }

    func playWavData(_ wav: Data) {
        guard wav.isEmpty == false else {
            log("tts_empty_audio")
            reset()
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("jarvis-tts-\(UUID().uuidString).wav")
        do {
            try wav.write(to: url)
            tempWavURL = url
            playWithAfplay(url: url)
        } catch {
            log("tts_wav_write_error \(error)")
            reset()
        }
    }

    func stopPlayback() {
        if let process = playerProcess, process.isRunning {
            process.terminate()
        }
        playerProcess = nil
    }

    func reset() {
        stopPlayback()
        pcmData = Data()
        if let url = tempWavURL {
            try? FileManager.default.removeItem(at: url)
            tempWavURL = nil
        }
    }

    private func playWithAfplay(url: URL) {
        stopPlayback()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        process.arguments = ["-v", String(configuredAfplayVolume()), url.path]
        process.terminationHandler = { _ in
            DispatchQueue.main.async {
                AfplaySpeechPlayer.shared.playbackDidFinish()
            }
        }
        do {
            try process.run()
            playerProcess = process
            log("tts_playback_started path=\(url.path)")
        } catch {
            log("tts_afplay_error \(error)")
            try? FileManager.default.removeItem(at: url)
            tempWavURL = nil
        }
    }

    fileprivate func playbackDidFinish() {
        if let url = tempWavURL {
            try? FileManager.default.removeItem(at: url)
            tempWavURL = nil
        }
        playerProcess = nil
        log("tts_playback_finished")
    }
}

private final class TTSStreamSessionDelegate: NSObject, URLSessionDataDelegate {
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        DispatchQueue.main.async {
            AfplaySpeechPlayer.shared.append(pcm: data)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        DispatchQueue.main.async {
            AfplaySpeechPlayer.shared.endStream(error: error)
        }
    }
}

private var ttsStreamSession: URLSession?
private var ttsStreamDelegate: TTSStreamSessionDelegate?

func speakAnswer(_ text: String) {
    guard let speakURL = configuredTTSURL() else {
        log("tts_skipped_no_url")
        return
    }

    let payload: [String: Any] = [
        "text": text,
        "voice": configuredTTSVoice(),
    ]

    do {
        AfplaySpeechPlayer.shared.begin()
        var request = URLRequest(url: speakURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        if speakURL.path.hasSuffix("/speak/stream") {
            let delegate = TTSStreamSessionDelegate()
            ttsStreamDelegate = delegate
            let configuration = URLSessionConfiguration.default
            configuration.timeoutIntervalForRequest = 120
            let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
            ttsStreamSession = session
            session.dataTask(with: request).resume()
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error {
                    log("tts_request_error \(error)")
                    AfplaySpeechPlayer.shared.reset()
                    return
                }
                guard let data, data.isEmpty == false else {
                    log("tts_empty_audio")
                    AfplaySpeechPlayer.shared.reset()
                    return
                }
                if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                    log("tts_http_error status=\(http.statusCode)")
                    AfplaySpeechPlayer.shared.reset()
                    return
                }
                AfplaySpeechPlayer.shared.playWavData(data)
            }
        }.resume()
    } catch {
        log("tts_request_error \(error)")
    }
}

func configuredAfplayVolume() -> Int {
    if let rawValue = try? String(contentsOf: ttsVolumeFile, encoding: .utf8) {
        let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if let parsed = Int(trimmedValue) {
            return min(255, max(0, parsed))
        }
        if let fraction = Double(trimmedValue), fraction >= 0, fraction <= 1 {
            return Int((fraction * 255).rounded())
        }
    }
    return defaultAfplayVolume
}

func configuredTTSVoice() -> String {
    if let rawValue = try? String(contentsOf: ttsVoiceFile, encoding: .utf8) {
        let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedValue.isEmpty == false {
            return trimmedValue
        }
    }

    return defaultTTSVoice
}

func configuredTTSURL() -> URL? {
    if let rawValue = try? String(contentsOf: ttsURLFile, encoding: .utf8) {
        let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedValue.isEmpty == false, let url = URL(string: trimmedValue) {
            return url
        }
    }

    return URL(string: defaultTTSURL)
}

func configuredTTSStreamURL() -> URL? {
    guard let url = configuredTTSURL() else {
        return nil
    }

    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    let path = components?.path ?? ""
    if path.hasSuffix("/speak/stream") {
        return url
    }
    if path.hasSuffix("/speak") {
        components?.path = String(path.dropLast("/speak".count)) + "/speak/stream"
        return components?.url
    }
    return url
}

func configuredAskURL() -> URL? {
    guard let rawValue = try? String(contentsOf: receiverURLFile, encoding: .utf8) else {
        return nil
    }

    let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmedValue.isEmpty {
        return nil
    }

    if trimmedValue.hasSuffix("/v1/window/events") {
        return URL(string: String(trimmedValue.dropLast("/v1/window/events".count)) + "/v1/ask")
    }

    if trimmedValue.hasSuffix("/v1/ask") {
        return URL(string: trimmedValue)
    }

    return URL(string: trimmedValue.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1/ask")
}

func requestAccessibilityIfNeeded() {
    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
    _ = AXIsProcessTrustedWithOptions(options as CFDictionary)
}

func requestSpeechPermission() {
    SFSpeechRecognizer.requestAuthorization { status in
        log("speech_permission \(status.rawValue)")
    }
}

func requestMicrophonePermission() {
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        log("microphone_permission \(granted)")
    }
}

func requestNotificationPermission() {
    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, error in
        if let error {
            log("notification_permission_error \(error)")
            return
        }
        log("notification_permission \(granted)")
    }
}

func notify(title: String, body: String) {
    let content = UNMutableNotificationContent()
    content.title = title
    content.body = body
    content.sound = .default

    let request = UNNotificationRequest(
        identifier: UUID().uuidString,
        content: content,
        trigger: nil
    )
    UNUserNotificationCenter.current().add(request) { error in
        if let error {
            log("notification_error \(error)")
        }
    }
}

func terminateIfAnotherHotkeyAppIsRunning() -> Bool {
    let currentPID = ProcessInfo.processInfo.processIdentifier
    let bundleIdentifier = Bundle.main.bundleIdentifier

    let alreadyRunning = NSWorkspace.shared.runningApplications.contains { application in
        application.processIdentifier != currentPID
            && application.bundleIdentifier == bundleIdentifier
    }

    if alreadyRunning {
        log("duplicate launch ignored")
        NSApp.terminate(nil)
        return true
    }

    return false
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
        NSLog("JarvisHotkey log error: \(String(describing: error))")
    }
}

extension String {
    var truncatedForNotification: String {
        if count <= 180 {
            return self
        }
        let index = self.index(startIndex, offsetBy: 177)
        return String(self[..<index]) + "..."
    }
}

enum JarvisHotkeyError: Error {
    case missingReceiverURL
    case httpStatus(Int)
}

let app = NSApplication.shared
let delegate = HotkeyAppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
