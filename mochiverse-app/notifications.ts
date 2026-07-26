import * as Notifications from "expo-notifications"
import * as Device from "expo-device"
import { Platform } from "react-native"

/**
 * Notification permission, owned by the native shell.
 *
 * The experience lives in a WebView, and a web page cannot raise the OS
 * permission prompt — so the page decides WHEN to ask and the shell does the
 * asking. That split matters more than it looks: iOS grants exactly one
 * system prompt per install, and once it is dismissed the only way back is
 * the Settings app. So the page is expected to show its own explanation
 * first and only call this after the user has said yes to that.
 */

export type PermissionStatus = "granted" | "denied" | "undetermined"

function normalize(status: Notifications.PermissionStatus, canAskAgain: boolean): PermissionStatus {
  if (status === "granted") return "granted"
  // "undetermined" is the only state where the OS prompt is still available.
  return canAskAgain && status === "undetermined" ? "undetermined" : "denied"
}

export async function getStatus(): Promise<PermissionStatus> {
  const p = await Notifications.getPermissionsAsync()
  return normalize(p.status, p.canAskAgain)
}

/**
 * Raises the OS prompt if it is still available. Returns the resulting state
 * so the page can stop asking once the answer is final.
 */
export async function request(): Promise<PermissionStatus> {
  const current = await Notifications.getPermissionsAsync()
  if (current.status === "granted") return "granted"
  if (!current.canAskAgain) return "denied"

  const next = await Notifications.requestPermissionsAsync({
    ios: { allowAlert: true, allowBadge: true, allowSound: true },
  })
  return normalize(next.status, next.canAskAgain)
}

/**
 * The device's push token, or null when one cannot exist.
 *
 * Simulators have no APNs registration, so this throws there rather than
 * returning empty — a notification feature that only breaks on real hardware
 * is worse than one that reports honestly during development.
 */
export async function getPushToken(): Promise<string | null> {
  if (!Device.isDevice) return null
  try {
    const token = await Notifications.getDevicePushTokenAsync()
    return typeof token.data === "string" ? token.data : String(token.data)
  } catch {
    return null
  }
}

/**
 * Hands the token to the server so it can be addressed later. Cookies carry
 * the identity, so this deliberately sends no identifier of its own.
 */
export async function registerToken(appUrl: string, token: string): Promise<boolean> {
  try {
    const res = await fetch(new URL("/api/push/register", appUrl).toString(), {
      method: "POST",
      headers: { "content-type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ token, platform: Platform.OS }),
    })
    return res.ok
  } catch {
    return false
  }
}
