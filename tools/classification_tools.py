"""NLP classification tools for the ClassifierAgent."""

from __future__ import annotations

import time

from models.tool_models import ToolResult


# Intent classification categories
INTENT_CATEGORIES = [
    "question",
    "command",
    "greeting",
    "complaint",
    "request",
    "feedback",
    "information",
    "other",
]

# Sentiment keywords for rule-based fallback
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


def classify_intent(text: str) -> str:
    """Classify the intent of the given text into predefined categories.

    Uses keyword-based heuristics to determine the most likely intent
    of the input text. Categories: question, command, greeting,
    complaint, request, feedback, information, other.

    Args:
        text: The text to classify.

    Returns:
        JSON string with the classification result including category and confidence.
    """
    start = time.perf_counter()
    text_lower = text.lower().strip()

    # Rule-based intent classification
    if text_lower.endswith("?") or text_lower.startswith(("what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "is", "are", "do", "does")):
        intent = "question"
        confidence = 0.85
    elif text_lower.startswith(("hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings")):
        intent = "greeting"
        confidence = 0.90
    elif any(word in text_lower for word in ("please", "could you", "would you", "can you", "i need", "i want")):
        intent = "request"
        confidence = 0.80
    elif any(word in text_lower for word in ("do", "run", "execute", "start", "stop", "create", "delete", "open", "close")):
        intent = "command"
        confidence = 0.75
    elif any(word in text_lower for word in NEGATIVE_KEYWORDS):
        intent = "complaint"
        confidence = 0.70
    elif any(word in text_lower for word in ("think", "believe", "suggest", "recommend", "opinion")):
        intent = "feedback"
        confidence = 0.65
    else:
        intent = "information"
        confidence = 0.50

    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result={
            "intent": intent,
            "confidence": confidence,
            "categories": INTENT_CATEGORIES,
            "text_analyzed": text,
        },
        tool_name="classify_intent",
        execution_time_ms=elapsed,
    ).model_dump_json()


def detect_sentiment(text: str) -> str:
    """Detect the sentiment of the given text as positive, negative, or neutral.

    Uses keyword-based analysis to determine sentiment polarity and confidence.

    Args:
        text: The text to analyze for sentiment.

    Returns:
        JSON string with sentiment label, confidence score, and keyword matches.
    """
    start = time.perf_counter()
    text_lower = text.lower()
    words = set(text_lower.split())

    positive_matches = words & POSITIVE_KEYWORDS
    negative_matches = words & NEGATIVE_KEYWORDS

    positive_score = len(positive_matches)
    negative_score = len(negative_matches)
    total = positive_score + negative_score

    if total == 0:
        sentiment = "neutral"
        confidence = 0.50
    elif positive_score > negative_score:
        sentiment = "positive"
        confidence = min(0.95, 0.60 + (positive_score - negative_score) * 0.10)
    elif negative_score > positive_score:
        sentiment = "negative"
        confidence = min(0.95, 0.60 + (negative_score - positive_score) * 0.10)
    else:
        sentiment = "mixed"
        confidence = 0.45

    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result={
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_keywords": list(positive_matches),
            "negative_keywords": list(negative_matches),
            "text_analyzed": text,
        },
        tool_name="detect_sentiment",
        execution_time_ms=elapsed,
    ).model_dump_json()
