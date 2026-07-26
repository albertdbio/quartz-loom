import { createSignal, Show } from "solid-js"

/**
 * Phone sign-in sheet — the gate that replaces the paywall while payments are
 * off, and the reason free use can't be farmed by clearing cookies.
 *
 * Two steps: number → code. The consent box is unticked by default and is for
 * *marketing* only; the sign-in code itself is transactional and always sends.
 * Pre-ticking it would be both a TCPA problem and a trust problem.
 */
type Props = {
  onDone: () => void
  onDismiss: () => void
  reason: "exhausted" | "time-up"
}

export default function SmsSignIn(props: Props) {
  const [step, setStep] = createSignal<"phone" | "code">("phone")
  const [phone, setPhone] = createSignal("")
  const [code, setCode] = createSignal("")
  const [consent, setConsent] = createSignal(false)
  const [busy, setBusy] = createSignal(false)
  const [err, setErr] = createSignal("")

  async function sendCode() {
    setErr("")
    setBusy(true)
    try {
      const res = await fetch("/api/auth/sms/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone: phone() }),
      })
      const body = (await res.json()) as { sent?: boolean; error?: string }
      if (res.ok && body.sent) setStep("code")
      else setErr(body.error ?? "could not send your code")
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function verify() {
    setErr("")
    setBusy(true)
    try {
      const res = await fetch("/api/auth/sms/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: code(), marketingConsent: consent() }),
      })
      const body = (await res.json()) as { signedIn?: boolean; error?: string }
      if (res.ok && body.signedIn) props.onDone()
      else setErr(body.error ?? "could not verify that code")
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div class="sms-backdrop" onClick={(e) => e.target === e.currentTarget && props.onDismiss()}>
      <style>{CSS}</style>
      <div class="sms" role="dialog" aria-modal="true" aria-label="sign in">
        <Show
          when={step() === "phone"}
          fallback={
            <>
              <h2>Enter your code</h2>
              <p class="sms-sub">We texted a 6-digit code to {phone()}.</p>
              <input
                class="sms-input code"
                type="text"
                inputmode="numeric"
                autocomplete="one-time-code"
                maxlength={6}
                placeholder="123456"
                value={code()}
                onInput={(e) => setCode(e.currentTarget.value.replace(/\D/g, ""))}
                onKeyDown={(e) => e.key === "Enter" && void verify()}
              />
              <button class="sms-cta" disabled={busy() || code().length < 6} onClick={() => void verify()}>
                {busy() ? "checking…" : "Start casting ✨"}
              </button>
              <button class="sms-link" onClick={() => setStep("phone")}>
                ← use a different number
              </button>
            </>
          }
        >
          <h2>{props.reason === "time-up" ? "Your free minute is up ⏱" : "Keep the magic going"}</h2>
          <p class="sms-sub">
            Sign in with your phone for unlimited casting. No password, no card.
          </p>
          <input
            class="sms-input"
            type="tel"
            inputmode="tel"
            autocomplete="tel"
            placeholder="(555) 123-4567"
            value={phone()}
            onInput={(e) => setPhone(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && void sendCode()}
          />
          <label class="sms-consent">
            <input type="checkbox" checked={consent()} onChange={(e) => setConsent(e.currentTarget.checked)} />
            <span>
              <b>Optional — not required to sign in.</b> Text me new spells and tips from
              <b>Mochiverse</b> (about 2–4 messages a month).
            </span>
          </label>
          <button class="sms-cta" disabled={busy() || phone().replace(/\D/g, "").length < 7} onClick={() => void sendCode()}>
            {busy() ? "sending…" : "Text me a code"}
          </button>
          <p class="sms-fine">
            We text a one-time code to sign you in — your number is never shown to anyone
            else. Msg &amp; data rates may apply. Reply HELP for help, STOP to opt out.
            Outside the US? Include your country code (e.g. +44).
          </p>
        </Show>
        <Show when={err()}>
          <p class="sms-err">{err()}</p>
        </Show>
        <button class="sms-dismiss" onClick={props.onDismiss}>
          not now
        </button>
      </div>
    </div>
  )
}

const CSS = `
  .sms-backdrop { position:fixed; inset:0; z-index:30; background:rgba(4,3,10,.8);
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    display:flex; align-items:center; justify-content:center; padding:22px; }
  .sms { background:#141020; border:1px solid #2a2440; border-radius:22px; padding:28px 26px 20px;
    max-width:390px; width:100%; text-align:center; color:#f0ecff;
    font-family:system-ui,-apple-system,sans-serif; box-shadow:0 24px 80px rgba(0,0,0,.65); }
  .sms h2 { margin:0 0 8px; font-size:21px; letter-spacing:-.01em; }
  .sms-sub { color:#a79fc4; font-size:14px; line-height:1.5; margin:0 0 16px; }
  .sms-input { width:100%; box-sizing:border-box; background:rgba(24,20,42,.9); border:1px solid #2a2440;
    border-radius:14px; padding:15px 16px; color:#f0ecff; font:inherit; font-size:17px; text-align:center;
    letter-spacing:.02em; }
  .sms-input:focus { outline:none; border-color:#c9a0ff; }
  .sms-input.code { letter-spacing:.5em; font-size:22px; font-weight:700; padding-left:24px; }
  .sms-consent { display:flex; align-items:flex-start; gap:10px; text-align:left; margin:16px 2px 4px;
    color:#b5adcf; font-size:13px; line-height:1.5; cursor:pointer; background:rgba(24,20,42,.6);
    border:1px solid #2a2440; border-radius:12px; padding:11px 13px; }
  .sms-consent b { color:#f0ecff; font-weight:650; }
  .sms-consent input { margin-top:2px; accent-color:#c9a0ff; width:17px; height:17px; flex:0 0 auto; }
  .sms-cta { width:100%; margin-top:14px; background:linear-gradient(135deg,#c9a0ff,#7f6aff); color:#0d0620;
    border:none; border-radius:999px; padding:15px 30px; font:inherit; font-size:16px; font-weight:800;
    cursor:pointer; box-shadow:0 8px 30px rgba(127,106,255,.45); min-height:52px; }
  .sms-cta:disabled { background:#221c38; color:#6f6791; box-shadow:none; cursor:default; }
  .sms-fine { color:#a79fc4; font-size:12.5px; line-height:1.55; margin:14px 0 0; }
  .sms-err { color:#ff6b8a; font-size:13px; margin:12px 0 0; }
  .sms-link { background:none; border:none; color:#9b93bd; font:inherit; font-size:13px; cursor:pointer;
    padding:10px; margin-top:6px; }
  .sms-link:hover { color:#c9a0ff; }
  .sms-dismiss { background:transparent; border:1px solid #2a2440; border-radius:10px; color:#c8c1de;
    font:inherit; font-size:14px; cursor:pointer; padding:11px 22px; margin-top:14px; min-height:44px; }
  .sms-dismiss:hover { color:#f0ecff; }
`
