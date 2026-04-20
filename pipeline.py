import re
from typing import Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\b(i('| a)?m|i am|i will|i'll|im going to|i'm going to)\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou('?re| are)?\s+going to die\b", re.IGNORECASE),
        re.compile(r"\bsomeone should\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\bi('?ll| will)\s+find where you live\b", re.IGNORECASE),
        re.compile(r"\bi('?m| am)\s+going to\s+(destroy|hurt|kill)\s+you\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\bgo kill yourself\b", re.IGNORECASE),
        re.compile(r"\byou should kill yourself\b", re.IGNORECASE),
        re.compile(r"\bnobody would miss you if you died\b", re.IGNORECASE),
        re.compile(r"\bdo everyone a favou?r and disappear\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi know where you live\b", re.IGNORECASE),
        re.compile(r"\bi('?ll| will)\s+post your address\b", re.IGNORECASE),
        re.compile(r"\bi found your real name\b", re.IGNORECASE),
        re.compile(r"\beveryone will know who you really are\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b\w+\s+are not (?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are animals\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+should be exterminated\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are a disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\beveryone report\b(?=.*@\w+|this account|this user)?", re.IGNORECASE),
        re.compile(r"\blet('?s| us) all go after\b", re.IGNORECASE),
        re.compile(r"\bmass report (this account|this user|@\w+)\b", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> Optional[Dict[str, Any]]:
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0
                }
    return None


class ModerationPipeline:
    def __init__(
        self,
        model_dir: str = "./models_cpu_part1",
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
        max_length: int = 96
    ):
        self.model_dir = model_dir
        self.block_threshold = block_threshold
        self.allow_threshold = allow_threshold
        self.max_length = max_length

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict_score(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()[0]

        # simple clipped score
        return float(np.clip(prob, 0.0, 1.0))

    def predict(self, text: str) -> Dict[str, Any]:
        filter_hit = input_filter(text)
        if filter_hit is not None:
            return filter_hit

        score = self.predict_score(text)

        if score >= self.block_threshold:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": score
            }
        elif score <= self.allow_threshold:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": score
            }
        else:
            return {
                "decision": "review",
                "layer": "model",
                "confidence": score
            }
