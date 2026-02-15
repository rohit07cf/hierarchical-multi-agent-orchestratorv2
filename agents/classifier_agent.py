"""ClassifierAgent: handles NLP classification and sentiment analysis."""

from __future__ import annotations

from typing import Any

from agents import function_tool

from agents.base_agent import BaseAgent
from prompts.react_prompt import get_react_prompt
from models.tool_models import ToolResult

# Intent classification categories
INTENT_CATEGORIES = [
    "question", "command", "greeting", "complaint",
    "request", "feedback", "information", "other",
]

POSITIVE_KEYWORDS = {
    "love", "great", "amazing", "wonderful", "excellent", "fantastic",
    "good", "happy", "awesome", "beautiful", "perfect", "best",
    "enjoy", "like", "pleased", "satisfied", "brilliant", "superb",
}

NEGATIVE_KEYWORDS = {
    "hate", "terrible", "awful", "horrible", "bad", "worst",
    "poor", "disappointing", "frustrated", "angry", "annoying",
    "broken", "useless", "ugly", "fail", "dislike", "unhappy",
}


@function_tool
def classify_intent(text: str) -> str:
    """Classify the intent of the given text into predefined categories.

    Categories: question, command, greeting, complaint, request,
    feedback, information, other.

    Args:
        text: The text to classify.
    """
    text_lower = text.lower().strip()

    if text_lower.endswith("?") or text_lower.startswith(
        ("what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "is", "are", "do", "does")
    ):
        intent, confidence = "question", 0.85
    elif text_lower.startswith(("hi", "hello", "hey", "good morning", "good afternoon", "good evening")):
        intent, confidence = "greeting", 0.90
    elif any(w in text_lower for w in ("please", "could you", "would you", "can you", "i need", "i want")):
        intent, confidence = "request", 0.80
    elif any(w in text_lower for w in ("do", "run", "execute", "start", "stop", "create", "delete")):
        intent, confidence = "command", 0.75
    elif any(w in text_lower for w in NEGATIVE_KEYWORDS):
        intent, confidence = "complaint", 0.70
    elif any(w in text_lower for w in ("think", "believe", "suggest", "recommend", "opinion")):
        intent, confidence = "feedback", 0.65
    else:
        intent, confidence = "information", 0.50

    return ToolResult.ok(
        result={"intent": intent, "confidence": confidence, "text_analyzed": text},
        tool_name="classify_intent",
    ).model_dump_json()


@function_tool
def detect_sentiment(text: str) -> str:
    """Detect the sentiment of the given text as positive, negative, neutral, or mixed.

    Args:
        text: The text to analyze for sentiment.
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    positive_matches = words & POSITIVE_KEYWORDS
    negative_matches = words & NEGATIVE_KEYWORDS
    pos_count = len(positive_matches)
    neg_count = len(negative_matches)
    total = pos_count + neg_count

    if total == 0:
        sentiment, confidence = "neutral", 0.50
    elif pos_count > neg_count:
        sentiment = "positive"
        confidence = min(0.95, 0.60 + (pos_count - neg_count) * 0.10)
    elif neg_count > pos_count:
        sentiment = "negative"
        confidence = min(0.95, 0.60 + (neg_count - pos_count) * 0.10)
    else:
        sentiment, confidence = "mixed", 0.45

    return ToolResult.ok(
        result={
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_keywords": list(positive_matches),
            "negative_keywords": list(negative_matches),
            "text_analyzed": text,
        },
        tool_name="detect_sentiment",
    ).model_dump_json()


class ClassifierAgentDef(BaseAgent):
    """Agent for NLP classification and sentiment analysis.

    Equipped with classify_intent and detect_sentiment tools.
    """

    TOOLS = ["classify_intent", "detect_sentiment"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(name="ClassifierAgent", model=model)

    def _get_system_prompt(self) -> str:
        return get_react_prompt("ClassifierAgent", self.TOOLS)

    def _register_tools(self) -> list[Any]:
        return [classify_intent, detect_sentiment]
