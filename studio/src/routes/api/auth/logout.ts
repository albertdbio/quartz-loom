import type { APIEvent } from "@solidjs/start/server"
import { cookieHeader, requestIsSecure, SUB_COOKIE, UID_COOKIE } from "~/server/entitlement"
import { jsonWithCookies } from "~/server/plan"

/**
 * Sign out completely: drop BOTH entitlement identities — the phone-verified
 * member cookie and the subscription/owner cookie. Real users need this on a
 * shared device, and it's the only way an owner-key holder can experience the
 * funnel they're shipping.
 *
 * The free-minute METER is deliberately left alone. Clearing it here would turn
 * sign-out into an infinite free-minute faucet: anyone could farm the paid API
 * by hitting this endpoint between sessions.
 */
const clearedCookies = (request: Request): ReadonlyArray<string> => {
  const secure = requestIsSecure(request)
  return [cookieHeader(UID_COOKIE, "", 0, secure), cookieHeader(SUB_COOKIE, "", 0, secure)]
}

export async function POST(event: APIEvent): Promise<Response> {
  return jsonWithCookies({ signedOut: true }, clearedCookies(event.request))
}

/**
 * Same thing, but reachable by just visiting a URL — which is what you need on
 * a phone, where issuing a POST by hand isn't practical. A forced sign-out over
 * GET is CSRF-able, but the worst outcome is an annoying logout: the meter (and
 * therefore spend) is untouched, so the convenience wins.
 */
export async function GET(event: APIEvent): Promise<Response> {
  const headers = new Headers({ Location: "/?signedout=1" })
  for (const c of clearedCookies(event.request)) headers.append("Set-Cookie", c)
  return new Response(null, { status: 302, headers })
}
