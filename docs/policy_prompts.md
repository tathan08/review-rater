# Policy Prompt Templates & Reasoning

## JSON Schema (all prompts)

We force compact, machine-readable JSON:

- `label`: APPROVE or REJECT
- `category`: No_Ads | Irrelevant | Rant_No_Visit | None
- `rationale`: 1–2 short sentences
- `flags`: { links, coupon, visit_claimed } to aid auditing

## Categories

### 1) No Advertisement / Promotional Content

**Reject when:** referral/promo codes, explicit solicitations (DM me, contact me), booking/ordering links, affiliate language, price lists/menus pasted, WhatsApp/Telegram.
**Allow:** normal positive reviews (even mentioning “cheap” or “good deal”).

**Why:** Promo spam distorts organic perception and violates platform policies. Precision matters to avoid false positives on legitimate positive reviews.

**Conflict handling:** If both Ads and Irrelevant seem true, prioritize **No_Ads** (more clear-cut policy breach).

### 2) No Irrelevant Content

**Reject when:** content is off-topic (politics, crypto hype, chain letters) or doesn’t describe this venue/service.
**Allow:** operational context (queue, parking, ambience, seating, noise, price at this place).

**Why:** Off-topic posts reduce signal quality for users evaluating the business.

**Conflict handling:** If both Irrelevant and Rant_No_Visit apply, prefer **Irrelevant** (since the core issue is being off-topic).

### 3) No Rants Without Visit

**Reject when:** generic insults/accusations with no evidence of actual visit or interaction.
**Allow:** specific experiences (time/date, items ordered, staff names, concrete incidents).

**Why:** Rants with no specifics offer low utility and are risky; specific experiences are actionable.

**Rescue rule:** If explicit visit markers exist, **approve** unless another policy (Ads/Irrelevant) is violated.

## Precedence

When multiple categories trigger, use: **No_Ads > Irrelevant > Rant_No_Visit**.
