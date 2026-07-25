import type { APIEvent } from "@solidjs/start/server"
import { cookieHeader, requestIsSecure, UID_COOKIE } from "~/server/entitlement"
import { jsonWithCookies } from "~/server/plan"

/** Drop the identity cookie. The subscriber row (and any consent) is untouched. */
export async function POST(event: APIEvent): Promise<Response> {
  return jsonWithCookies({ signedOut: true }, [
    cookieHeader(UID_COOKIE, "", 0, requestIsSecure(event.request)),
  ])
}
