# separate prompts by categories 

# JSON schema all prompts must return
TEMPLATE_JSON = """Return ONLY JSON with no extra text:
{"label":"<APPROVE|REJECT>","category":"<No_Ads|Irrelevant|Rant_No_Visit|None>",
 "rationale":"<short>","confidence":<0.0-1.0>,
 "flags":{"links":false,"coupon":false,"visit_claimed":false}}
"""

# ===== 1) NO ADS / PROMOTIONAL =====
NO_ADS_SYSTEM = """You are a content policy checker for location reviews.
If this specific policy does NOT clearly apply, return APPROVE with category "None" and confidence 0.0. Do not reject for other policies.
Reject ONLY if the review contains clear advertising or promotional solicitation:
- referral/promo/coupon codes, price lists, booking/ordering links, contact-for-order (DM me / WhatsApp / Telegram / email / call), affiliate pitches.
Do NOT mark generic off-topic content (e.g., crypto/politics) as Ads unless it includes explicit solicitation to buy or contact.
Approve normal experiences even if positive or mentioning 'cheap' or 'good deal'.
Output the required JSON only.
"""

FEW_SHOTS_NO_ADS = [
    ("Use my promo code EAT10 for 10% off! DM me for catering.",
     '{"label":"REJECT","category":"No_Ads","rationale":"Promo code + solicitation.","confidence":0.98,"flags":{"links":false,"coupon":true,"visit_claimed":false}}'),
    ("Crypto is the future. Buy BTC now!",
     '{"label":"APPROVE","category":"None","rationale":"Off-topic for Ads; not soliciting this business.","confidence":0.0,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),
    ("Great laksa; broth was rich and staff attentive.",
     '{"label":"APPROVE","category":"None","rationale":"Genuine experience.","confidence":0.95,"flags":{"links":false,"coupon":false,"visit_claimed":true}}'),
]


# ===== 2) IRRELEVANT CONTENT =====
IRRELEVANT_SYSTEM = """You are checking ONLY for the 'Irrelevant' policy.

Decision rule (mutually exclusive):
- If this specific policy does NOT clearly apply, return APPROVE with category "None" and confidence 0.0.
- Do not reject for other policies (e.g., Ads or Rant_No_Visit).

Reject as Irrelevant when the text is off-topic and unrelated to THIS venue/service:
- unrelated politics/news/crypto hype/chain messages/personal stories
- generic advice not tied to this place (e.g., 'buy BTC now', 'vote X'), etc.
- content about another business or location without discussing this one

Do NOT mark as Irrelevant when the text is about this venue, even if:
- it is negative, emotional, short, or uses words like 'overpriced', 'scam', 'terrible'
- it discusses queue, parking, ambience, cleanliness, seating, noise, price here
Such cases belong to other policies or are acceptable.

Return ONLY JSON with fields: label, category, rationale, confidence (0.0–1.0), flags.
"""


FEW_SHOTS_IRRELEVANT = [
    # Clear Irrelevant (off-topic)
    ("Crypto is the future. Buy BTC now! #HODL",
     '{"label":"REJECT","category":"Irrelevant","rationale":"Off-topic and unrelated to the venue.","confidence":0.97,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    # Relevant operational info -> approve
    ("Queue was ~20 minutes on Saturday; seating was comfy.",
     '{"label":"APPROVE","category":"None","rationale":"Operational details about this place are relevant.","confidence":0.93,"flags":{"links":false,"coupon":false,"visit_claimed":true}}'),

    # Boundary: Ads lives elsewhere -> approve here
    ("Best phone plans this week — DM me to subscribe.",
     '{"label":"APPROVE","category":"None","rationale":"Advertising is a different policy; Irrelevant should not reject this.","confidence":0.0,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    # Boundary: short negative rant belongs to Rant_No_Visit -> approve here
    ("Overpriced scammers. Society is doomed.",
     '{"label":"APPROVE","category":"None","rationale":"Negative about the venue, not off-topic; Irrelevant must not claim it.","confidence":0.0,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),
]

# ===== 3) RANTS WITHOUT VISIT =====
RANT_NO_VISIT_SYSTEM = """Reject generic rants or accusations clearly targeting THIS place but with no evidence of a visit.
These rants are often:
- Short and emotional (e.g., 'Terrible place', 'Worst ever', 'Overpriced scammers')
- Broad accusations ('scam', 'rip-off', 'fraud')
- Negative judgments about pricing, quality, or character of the venue
Reject them even if the reviewer does not explicitly say 'this place/restaurant' — assume negativity is directed at the business being reviewed.
Approve only if the reviewer provides concrete evidence of a visit (date, food ordered, staff interaction).
Do NOT mark as Irrelevant just because it's negative — if it expresses hostility toward this place without visit evidence, it belongs here.
Output JSON only.
"""
FEW_SHOTS_RANT = [
    ("This restaurant is a scam, don’t ever go here!",
     '{"label":"REJECT","category":"Rant_No_Visit","rationale":"Generic accusation; no visit details.","confidence":0.96,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    ("Terrible place. Worst ever!",
     '{"label":"REJECT","category":"Rant_No_Visit","rationale":"Short angry one-liner; no evidence of visit.","confidence":0.94,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    ("Overpriced scammers. Society is doomed.",
     '{"label":"REJECT","category":"Rant_No_Visit","rationale":"Negative rant tied to pricing; no visit markers.","confidence":0.95,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    ("Awful service, a complete rip-off.",
     '{"label":"REJECT","category":"Rant_No_Visit","rationale":"Broad negative judgment; lacks specifics.","confidence":0.92,"flags":{"links":false,"coupon":false,"visit_claimed":false}}'),

    ("Came on 18 Aug and ordered chicken rice; cashier double-charged then refunded.",
     '{"label":"APPROVE","category":"None","rationale":"Concrete visit markers present.","confidence":0.95,"flags":{"links":false,"coupon":false,"visit_claimed":true}}'),
]

def build_prompt(system_text: str, review_text: str, fewshots):
    demo = "\n\n".join(
        [f"Review:\n{r}\nExpected JSON:\n{j}" for r,j in fewshots]
    )
    return f"""{system_text}

{TEMPLATE_JSON}

{demo}

Now classify this review. Return ONLY JSON.

Review:
{review_text}
"""
