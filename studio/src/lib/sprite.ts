/**
 * Sprite post-processing, client-side.
 *
 * Generated characters arrive as square PNGs whose actual content occupies
 * roughly half the canvas width (a standing chibi is tall, and the generator
 * centers it). Rendering the raw square makes every character look undersized
 * no matter what the layout does — measured on a real sprite: content bbox
 * was 49% of width. So the sprite is cropped to its opaque bounds ONCE when
 * it arrives, and the scaffold renders honest pixels.
 */

/** Opaque bounding box of RGBA pixel data. Pure — the testable core. */
export function alphaBBox(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  threshold = 8,
): { left: number; top: number; right: number; bottom: number } | null {
  let left = width
  let top = height
  let right = -1
  let bottom = -1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4 + 3]! > threshold) {
        if (x < left) left = x
        if (x > right) right = x
        if (y < top) top = y
        if (y > bottom) bottom = y
      }
    }
  }
  return right < 0 ? null : { left, top, right, bottom }
}

/** Crop a data-URL sprite to its opaque content (plus a small breathing margin). */
export async function cropSpriteToContent(dataUrl: string, marginFrac = 0.04): Promise<string> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error("sprite failed to decode"))
    img.src = dataUrl
  })
  const canvas = document.createElement("canvas")
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext("2d")
  if (!ctx) return dataUrl
  ctx.drawImage(img, 0, 0)
  const box = alphaBBox(
    ctx.getImageData(0, 0, canvas.width, canvas.height).data,
    canvas.width,
    canvas.height,
  )
  if (!box) return dataUrl

  const mx = Math.round((box.right - box.left) * marginFrac)
  const my = Math.round((box.bottom - box.top) * marginFrac)
  const sx = Math.max(0, box.left - mx)
  const sy = Math.max(0, box.top - my)
  const sw = Math.min(canvas.width, box.right + mx + 1) - sx
  // bottom stays FLUSH: the scaffold bottom-anchors the sprite onto its
  // contact shadow, and any margin here reads as the character hovering
  const sh = Math.min(canvas.height, box.bottom + 1) - sy

  const out = document.createElement("canvas")
  out.width = sw
  out.height = sh
  const octx = out.getContext("2d")
  if (!octx) return dataUrl
  octx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh)
  return out.toDataURL("image/png")
}
