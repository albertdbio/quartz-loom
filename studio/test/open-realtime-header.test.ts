import { describe, expect, it } from "vitest"
import {
  decodeOutputHeader,
  encodeFrameHeader,
  HDR_IN_SIZE,
  HDR_OUT_SIZE,
} from "~/lib/open-realtime"

/**
 * Wire-contract v1 header codecs, pinned against Python's struct — the
 * server's exact packing (openstudio-server/server.py + contract_test.py):
 *
 *   struct.Struct("<BId").pack(0x01, 7, 12345.678).hex()
 *     == "01070000005839b4c8d61cc840"
 *   struct.Struct("<BId").pack(0x01, 0xFFFFFFFF, 0.5).hex()
 *     == "01ffffffff000000000000e03f"
 *   struct.Struct("<BIdf").pack(0x02, 9, 98765.4321, 42.5).hex()
 *     == "02090000008ab0e1e9d61cf84000002a42"
 */

function hex(buf: ArrayBuffer): string {
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("")
}

function fromHex(s: string): ArrayBuffer {
  const bytes = new Uint8Array(s.length / 2)
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16)
  return bytes.buffer
}

describe("encodeFrameHeader (client → server, <BId>, 13 bytes)", () => {
  it("matches Python struct.pack('<BId', 0x01, 7, 12345.678)", () => {
    expect(hex(encodeFrameHeader(7, 12345.678))).toBe("01070000005839b4c8d61cc840")
  })

  it("is exactly 13 bytes", () => {
    expect(encodeFrameHeader(0, 0).byteLength).toBe(HDR_IN_SIZE)
    expect(HDR_IN_SIZE).toBe(13)
  })

  it("packs seq little-endian at max u32 (wrap boundary)", () => {
    expect(hex(encodeFrameHeader(0xffffffff, 0.5))).toBe("01ffffffff000000000000e03f")
  })

  it("wraps seq mod 2^32", () => {
    expect(hex(encodeFrameHeader(0x1_0000_0002, 0.5))).toBe(hex(encodeFrameHeader(2, 0.5)))
  })
})

describe("decodeOutputHeader (server → client, <BIdf>, 17 bytes)", () => {
  it("parses Python struct.pack('<BIdf', 0x02, 9, 98765.4321, 42.5) + payload", () => {
    const payload = fromHex("02090000008ab0e1e9d61cf84000002a42" + "ffd8") // + 2 JPEG bytes
    const hdr = decodeOutputHeader(payload)
    expect(hdr).not.toBeNull()
    expect(hdr?.seq).toBe(9)
    expect(hdr?.captureTsMs).toBeCloseTo(98765.4321, 6)
    expect(hdr?.inferMs).toBe(42.5) // exactly representable in f32
  })

  it("rejects the wrong magic (0x01 is the client→server direction)", () => {
    const payload = fromHex("01090000008ab0e1e9d61cf84000002a42" + "ffd8")
    expect(decodeOutputHeader(payload)).toBeNull()
  })

  it("rejects header-only / short messages (no JPEG payload)", () => {
    expect(decodeOutputHeader(fromHex("0200000000000000000000000000000000"))).toBeNull() // exactly 17
    expect(decodeOutputHeader(new ArrayBuffer(0))).toBeNull()
    expect(decodeOutputHeader(new ArrayBuffer(5))).toBeNull()
    expect(HDR_OUT_SIZE).toBe(17)
  })
})
