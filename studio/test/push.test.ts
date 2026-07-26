import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { resetSubscribersDbForTests, subscribersDb } from "../src/server/subscribers"
import { forgetPushToken, recordPushToken, resetPushTableForTests, tokensFor } from "../src/server/push"

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "mochiverse-push-"))
  resetSubscribersDbForTests()
  resetPushTableForTests()
  subscribersDb(join(dir, "test.db"))
})

afterEach(() => {
  resetSubscribersDbForTests()
  resetPushTableForTests()
  rmSync(dir, { recursive: true, force: true })
})

describe("push token store", () => {
  it("records a device once, then treats repeats as re-registration", () => {
    expect(recordPushToken({ token: "tok-aaaaaaa", platform: "ios", phoneId: "pid1" })).toBe(true)
    expect(recordPushToken({ token: "tok-aaaaaaa", platform: "ios", phoneId: "pid1" })).toBe(false)
    expect(tokensFor("pid1")).toEqual([{ token: "tok-aaaaaaa", platform: "ios" }])
  })

  it("re-points a device at whoever is signed in now", () => {
    recordPushToken({ token: "tok-shared", platform: "ios", phoneId: "pid1" })
    recordPushToken({ token: "tok-shared", platform: "ios", phoneId: "pid2" })
    // the previous owner must stop being notified on a device they gave up
    expect(tokensFor("pid1")).toEqual([])
    expect(tokensFor("pid2")).toEqual([{ token: "tok-shared", platform: "ios" }])
  })

  it("does not orphan a linked device when it re-registers anonymously", () => {
    // the shell registers on app launch, before any cookie is read
    recordPushToken({ token: "tok-linked", platform: "ios", phoneId: "pid1" })
    recordPushToken({ token: "tok-linked", platform: "ios", phoneId: null })
    expect(tokensFor("pid1")).toEqual([{ token: "tok-linked", platform: "ios" }])
  })

  it("keeps an unlinked device until someone claims it", () => {
    expect(recordPushToken({ token: "tok-anon12", platform: "android", phoneId: null })).toBe(true)
    expect(tokensFor("pid1")).toEqual([])
    recordPushToken({ token: "tok-anon12", platform: "android", phoneId: "pid1" })
    expect(tokensFor("pid1")).toEqual([{ token: "tok-anon12", platform: "android" }])
  })

  it("collects every device a person signed in on", () => {
    recordPushToken({ token: "tok-phone1", platform: "ios", phoneId: "pid1" })
    recordPushToken({ token: "tok-tablet", platform: "android", phoneId: "pid1" })
    expect(tokensFor("pid1")).toHaveLength(2)
  })

  it("forgets a token the push service rejected", () => {
    recordPushToken({ token: "tok-deadxx", platform: "ios", phoneId: "pid1" })
    forgetPushToken("tok-deadxx")
    expect(tokensFor("pid1")).toEqual([])
  })
})
