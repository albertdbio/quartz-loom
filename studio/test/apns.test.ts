import { describe, expect, it } from "vitest"
import { apnsConfigFromEnv, apnsPayload, classifyApnsAnswer } from "../src/server/apns"

describe("classifyApnsAnswer", () => {
  it("delivers on 200 from either gateway", () => {
    expect(classifyApnsAnswer(200, undefined, "production")).toBe("delivered")
    expect(classifyApnsAnswer(200, undefined, "sandbox")).toBe("delivered")
  })

  it("treats 410 Unregistered as a token to forget — pushing to ghosts forever is the alternative", () => {
    expect(classifyApnsAnswer(410, "Unregistered", "production")).toBe("forget-token")
    expect(classifyApnsAnswer(410, "Unregistered", "sandbox")).toBe("forget-token")
  })

  it("routes production BadDeviceToken to the sandbox (dev-build tokens live there)", () => {
    expect(classifyApnsAnswer(400, "BadDeviceToken", "production")).toBe("retry-sandbox")
  })

  it("gives up on sandbox BadDeviceToken — the token is garbage, not misrouted", () => {
    expect(classifyApnsAnswer(400, "BadDeviceToken", "sandbox")).toBe("forget-token")
  })

  it("does NOT forget tokens on auth/infra failures — a bad key must not wipe the audience", () => {
    expect(classifyApnsAnswer(403, "InvalidProviderToken", "production")).toBe("failed")
    expect(classifyApnsAnswer(500, undefined, "production")).toBe("failed")
    expect(classifyApnsAnswer(0, "timeout", "sandbox")).toBe("failed")
  })
})

describe("apnsPayload", () => {
  it("builds the alert shape Apple documents", () => {
    expect(JSON.parse(apnsPayload("Hi", "there"))).toEqual({
      aps: { alert: { title: "Hi", body: "there" }, sound: "default" },
    })
  })
})

describe("apnsConfigFromEnv", () => {
  it("is null until both key path and key id exist — half a config must not half-work", () => {
    expect(apnsConfigFromEnv({} as NodeJS.ProcessEnv)).toBeNull()
    expect(apnsConfigFromEnv({ APNS_KEY_PATH: "/k.p8" } as NodeJS.ProcessEnv)).toBeNull()
    expect(apnsConfigFromEnv({ APNS_KEY_ID: "ABC123DEFG" } as NodeJS.ProcessEnv)).toBeNull()
  })

  it("defaults team and topic to this app's identity", () => {
    const cfg = apnsConfigFromEnv({ APNS_KEY_PATH: "/k.p8", APNS_KEY_ID: "ABC123DEFG" } as NodeJS.ProcessEnv)
    expect(cfg).toEqual({
      keyPath: "/k.p8",
      keyId: "ABC123DEFG",
      teamId: "THRN2A465Z",
      topic: "io.mochiverse.app",
    })
  })
})
