import { useEffect, useRef, useState } from "react"
import { ActivityIndicator, Linking, Platform, StyleSheet, Text, View } from "react-native"
import { StatusBar } from "expo-status-bar"
import Constants from "expo-constants"
import { useCameraPermissions } from "expo-camera"
import { WebView } from "react-native-webview"

/**
 * Magic Wand — the native shell.
 *
 * The product itself is the mobile-first /wand web app (realtime touch-to-
 * transform over Decart Lucy 2.5, first minute free then Studio Pro). This
 * shell wraps it in a camera-enabled WebView: WKWebView/Chromium support
 * getUserMedia + WebRTC in-page (iOS 14.3+/Android), so the whole realtime
 * pipeline runs unchanged. Native WebRTC (à la opentxt's LiveKit dev build)
 * is the later upgrade path if we outgrow the WebView.
 *
 * Point the shell at a deployment with EXPO_PUBLIC_WAND_URL (dev: an https
 * tunnel or your Mac's localhost from the iOS simulator) or app.json's
 * `extra.wandUrl` (production). getUserMedia requires a secure context —
 * https or localhost; a bare LAN IP over http will NOT get a camera.
 */
const WAND_URL: string =
  process.env.EXPO_PUBLIC_WAND_URL ??
  (Constants.expoConfig?.extra?.["wandUrl"] as string | undefined) ??
  "http://localhost:3000/wand"

export default function App() {
  const webref = useRef<WebView>(null)
  const [permission, requestPermission] = useCameraPermissions()
  const [failed, setFailed] = useState<string | null>(null)

  // Ask for the native camera permission up front so the in-page getUserMedia
  // prompt is the only prompt the user sees inside the experience.
  useEffect(() => {
    if (permission && !permission.granted && permission.canAskAgain) {
      void requestPermission()
    }
  }, [permission, requestPermission])

  return (
    // Edge-to-edge on purpose: the wand page is a full-screen camera stage and
    // insets its own floating controls with env(safe-area-inset-*), so a
    // SafeAreaView here would letterbox the video instead of protecting it.
    <View style={styles.root}>
      <StatusBar hidden />
      {failed ? (
        <View style={styles.center}>
          <Text style={styles.title}>magic wand ✨</Text>
          <Text style={styles.err}>{failed}</Text>
          <Text style={styles.hint} onPress={() => setFailed(null)}>
            tap to retry
          </Text>
        </View>
      ) : (
        <WebView
          ref={webref}
          source={{ uri: WAND_URL }}
          style={styles.web}
          // Camera/mic inside the page: grant without a second native prompt.
          mediaCapturePermissionGrantType="grant"
          allowsInlineMediaPlayback
          mediaPlaybackRequiresUserAction={false}
          javaScriptEnabled
          domStorageEnabled
          // Entitlement lives in HttpOnly cookies. WKWebView's default jar is
          // per-session, so without these the free-minute meter (and an owner
          // grant) would reset on every app launch — a paywall leak, not just
          // an inconvenience.
          sharedCookiesEnabled
          thirdPartyCookiesEnabled
          // Stripe Checkout must open in the real browser (Apple 3.1.1-safe:
          // the purchase is a web flow outside the app binary).
          onShouldStartLoadWithRequest={(req) => {
            const external =
              req.url.startsWith("https://checkout.stripe.com") ||
              req.url.startsWith("https://billing.stripe.com")
            if (external) {
              void Linking.openURL(req.url)
              return false
            }
            return true
          }}
          onError={(e) => setFailed(`could not reach the studio (${e.nativeEvent.description ?? "network error"})`)}
          startInLoadingState
          renderLoading={() => (
            <View style={[styles.center, StyleSheet.absoluteFill]}>
              <ActivityIndicator color="#c9a0ff" size="large" />
              <Text style={styles.hint}>summoning…</Text>
            </View>
          )}
          {...(Platform.OS === "android" ? { onPermissionRequest: undefined } : {})}
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#07070d" },
  web: { flex: 1, backgroundColor: "#07070d" },
  center: { flex: 1, alignItems: "center", justifyContent: "center", gap: 12, backgroundColor: "#07070d" },
  title: { color: "#f0ecff", fontSize: 28, fontWeight: "700" },
  err: { color: "#ff6b8a", fontSize: 14, textAlign: "center", paddingHorizontal: 30 },
  hint: { color: "#9a92b8", fontSize: 13 },
})
