"""
Core Constants
Centralized constants and configurations for the review rater system
"""

# Policy Categories (matching policy_prompts.py)
POLICY_CATEGORIES = {
    'NO_ADS': 'No_Ads',
    'IRRELEVANT': 'Irrelevant', 
    'RANT_NO_VISIT': 'Rant_No_Visit',
    'NONE': 'None'
}

# Label Types (matching policy_prompts.py)
LABELS = {
    'APPROVE': 'APPROVE',
    'REJECT': 'REJECT'
}

# Model Configurations
DEFAULT_MODELS = {
    'SENTIMENT': "distilbert-base-uncased-finetuned-sst-2-english",
    'TOXIC': "unitary/toxic-bert", 
    'ZERO_SHOT': "facebook/bart-large-mnli",
    'GPT_DEFAULT': "gpt-3.5-turbo",
    'OLLAMA_DEFAULT': "mistral:7b-instruct"
}

# Classification Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.8,
    'MEDIUM': 0.6,
    'LOW': 0.4,
    'DEFAULT': 0.55
}

# Regex Patterns for Pre-classification (matching prompt_runner.py)
REGEX_PATTERNS = {
    'AD_SOLICIT': r"(promo\s*code|referr?al|use\s+code|discount\s*code|coupon|dm\s+me|whats?app|telegram|wa\.me|t\.me|contact\s+me|order\s+via|book\s+now|call\s+\+?\d)",
    'LINK': r"https?://|www\.",
    'OFFTOPIC': r"\b(crypto|bitcoin|btc|politics|election|stock tips|forex|nft|blockchain)\b",
    'VISIT_MARKERS': r"\b(i|we)\s+(visited|went|came|ordered|ate|bought|tried|dined)\b",
    'PLACE_REFERENCES': r"\b(this (place|shop|restaurant|cafe|store|hotel)|here)\b",
    'NEGATIVE_WORDS': ["scam", "scammers", "ripoff", "rip-off", "overpriced", "terrible", "awful", "worst"]
}

# Zero-shot Classification Labels (matching hf_pipeline.py)
ZERO_SHOT_LABELS = [
    "an advertisement or promotional solicitation for this business (promo code, referral, links, contact to buy)",
    "off-topic or unrelated to this business (e.g., politics, crypto, chain messages, personal stories not about this place)",
    "a generic negative rant about this business without evidence of a visit (short insults, 'scam', 'overpriced', 'worst ever')",
    "a relevant on-topic description of a visit or experience at this business"
]

# Mapping zero-shot labels to policy categories
ZERO_SHOT_TO_POLICY = {
    ZERO_SHOT_LABELS[0]: POLICY_CATEGORIES['NO_ADS'],
    ZERO_SHOT_LABELS[1]: POLICY_CATEGORIES['IRRELEVANT'],
    ZERO_SHOT_LABELS[2]: POLICY_CATEGORIES['RANT_NO_VISIT'],
    ZERO_SHOT_LABELS[3]: POLICY_CATEGORIES['NONE']
}

# Standard JSON Schema for all outputs
STANDARD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": [LABELS['APPROVE'], LABELS['REJECT']]},
        "category": {"type": "string", "enum": list(POLICY_CATEGORIES.values())},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "flags": {
            "type": "object",
            "properties": {
                "links": {"type": "boolean"},
                "coupon": {"type": "boolean"},
                "visit_claimed": {"type": "boolean"}
            }
        }
    },
    "required": ["label", "category", "rationale", "confidence", "flags"]
}

# Default Column Names for CSV Files
CSV_COLUMNS = {
    'ID': 'id',
    'TEXT': 'text',
    'PRED_LABEL': 'pred_label',
    'PRED_CATEGORY': 'pred_category',
    'GOLD_LABEL': 'gold_label',
    'GOLD_CATEGORY': 'gold_category',
    'CONFIDENCE': 'confidence',
    'RATIONALE': 'rationale'
}
