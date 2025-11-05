"""Cold start strategy for new users via Holland codes (RIASEC)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)

# Holland codes (RIASEC) personality types
HollandCode = Literal["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

# Mapping of careers to primary Holland codes
CAREER_HOLLAND_MAPPING: dict[str, list[HollandCode]] = {
    # Realistic (R) - Doers
    "Mechanical Engineer": ["Realistic", "Investigative"],
    "Carpenter": ["Realistic"],
    "Electrician": ["Realistic"],
    "Pilot": ["Realistic", "Investigative"],
    "Chef": ["Realistic", "Artistic"],
    
    # Investigative (I) - Thinkers
    "Data Scientist": ["Investigative", "Conventional"],
    "Research Scientist": ["Investigative"],
    "Software Engineer": ["Investigative", "Conventional"],
    "Biomedical Researcher": ["Investigative"],
    "Actuary": ["Investigative", "Conventional"],
    
    # Artistic (A) - Creators
    "Graphic Designer": ["Artistic"],
    "Writer": ["Artistic"],
    "Musician": ["Artistic"],
    "Interior Designer": ["Artistic", "Enterprising"],
    "Video Producer": ["Artistic", "Enterprising"],
    
    # Social (S) - Helpers
    "Teacher": ["Social"],
    "Nurse": ["Social", "Investigative"],
    "Counselor": ["Social"],
    "Social Worker": ["Social"],
    "Human Resources Manager": ["Social", "Enterprising"],
    
    # Enterprising (E) - Persuaders
    "Sales Manager": ["Enterprising", "Social"],
    "Marketing Director": ["Enterprising", "Artistic"],
    "Entrepreneur": ["Enterprising"],
    "Real Estate Agent": ["Enterprising", "Social"],
    "Lawyer": ["Enterprising", "Investigative"],
    
    # Conventional (C) - Organizers
    "Accountant": ["Conventional"],
    "Financial Analyst": ["Conventional", "Investigative"],
    "Database Administrator": ["Conventional", "Investigative"],
    "Auditor": ["Conventional"],
    "Librarian": ["Conventional", "Social"]
}


@dataclass
class HollandQuizQuestion:
    """Holland code quiz question."""
    
    question: str
    holland_dimension: HollandCode
    weight: float = 1.0


# 10-question Holland code quiz (2 per dimension, weighted for bias correction)
HOLLAND_QUIZ: list[HollandQuizQuestion] = [
    # Realistic
    HollandQuizQuestion(
        "I enjoy working with tools, machines, or hands-on activities.",
        "Realistic",
        weight=1.2
    ),
    HollandQuizQuestion(
        "I prefer practical, tangible work over abstract concepts.",
        "Realistic",
        weight=0.8
    ),
    
    # Investigative
    HollandQuizQuestion(
        "I like solving complex problems and conducting research.",
        "Investigative",
        weight=1.0
    ),
    HollandQuizQuestion(
        "I enjoy analyzing data and discovering patterns.",
        "Investigative",
        weight=1.1
    ),
    
    # Artistic
    HollandQuizQuestion(
        "I am drawn to creative expression and aesthetic design.",
        "Artistic",
        weight=1.0
    ),
    HollandQuizQuestion(
        "I value originality and artistic freedom in my work.",
        "Artistic",
        weight=0.9
    ),
    
    # Social
    HollandQuizQuestion(
        "I find fulfillment in helping and teaching others.",
        "Social",
        weight=1.1
    ),
    HollandQuizQuestion(
        "I enjoy working in team-oriented, collaborative environments.",
        "Social",
        weight=0.9
    ),
    
    # Enterprising
    HollandQuizQuestion(
        "I am motivated by leadership roles and influencing others.",
        "Enterprising",
        weight=1.0
    ),
    HollandQuizQuestion(
        "I thrive in competitive, goal-driven settings.",
        "Enterprising",
        weight=1.0
    ),
    
    # Conventional
    HollandQuizQuestion(
        "I prefer structured tasks with clear procedures.",
        "Conventional",
        weight=1.0
    ),
    HollandQuizQuestion(
        "I am detail-oriented and value accuracy in my work.",
        "Conventional",
        weight=1.1
    )
]


def compute_holland_vector(quiz_responses: dict[int, int]) -> np.ndarray:
    """
    Compute 6-dimensional Holland code vector from quiz responses.
    
    Args:
        quiz_responses: Dictionary mapping question index (0-9) to response (1-5 Likert scale)
    
    Returns:
        6-dimensional Holland vector (R, I, A, S, E, C)
    """
    holland_scores = {
        "Realistic": 0.0,
        "Investigative": 0.0,
        "Artistic": 0.0,
        "Social": 0.0,
        "Enterprising": 0.0,
        "Conventional": 0.0
    }
    
    for question_idx, response in quiz_responses.items():
        if question_idx >= len(HOLLAND_QUIZ):
            LOGGER.warning(f"Invalid question index: {question_idx}")
            continue
        
        question = HOLLAND_QUIZ[question_idx]
        
        # Normalize response to 0-1
        normalized_response = (response - 1) / 4.0  # 1-5 → 0-1
        
        # Add weighted score to dimension
        holland_scores[question.holland_dimension] += normalized_response * question.weight
    
    # Normalize scores (sum to 1.0)
    total_score = sum(holland_scores.values())
    if total_score > 0:
        for key in holland_scores:
            holland_scores[key] /= total_score
    
    # Convert to vector (R, I, A, S, E, C)
    vector = np.array([
        holland_scores["Realistic"],
        holland_scores["Investigative"],
        holland_scores["Artistic"],
        holland_scores["Social"],
        holland_scores["Enterprising"],
        holland_scores["Conventional"]
    ], dtype=np.float32)
    
    return vector


def generate_interest_text_from_holland(holland_vector: np.ndarray) -> str:
    """
    Generate synthetic interest text from Holland vector.
    
    This allows cold-start users to leverage text embeddings without providing
    explicit interest descriptions.
    
    Args:
        holland_vector: 6-dimensional Holland code vector
    
    Returns:
        Natural language interest description
    """
    dimensions = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
    
    # Find top 2 dimensions
    top_indices = np.argsort(holland_vector)[-2:][::-1]
    primary = dimensions[top_indices[0]]
    secondary = dimensions[top_indices[1]]
    
    # Template-based text generation
    templates = {
        "Realistic": "hands-on work, building and fixing things, working with tools",
        "Investigative": "solving complex problems, research, data analysis",
        "Artistic": "creative expression, design, innovation",
        "Social": "helping others, teaching, collaboration",
        "Enterprising": "leadership, entrepreneurship, sales",
        "Conventional": "organization, data management, precision"
    }
    
    interest_text = (
        f"I am interested in {templates[primary]}. "
        f"I also enjoy {templates[secondary]}."
    )
    
    return interest_text


def recommend_careers_from_holland(
    holland_vector: np.ndarray,
    top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Recommend careers based on Holland code alignment.
    
    Args:
        holland_vector: User's Holland code vector
        top_k: Number of recommendations
    
    Returns:
        List of (career_title, alignment_score) tuples
    """
    career_scores = []
    
    dimensions = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
    
    for career, holland_codes in CAREER_HOLLAND_MAPPING.items():
        # Create career Holland vector
        career_vector = np.zeros(6, dtype=np.float32)
        for code in holland_codes:
            idx = dimensions.index(code)
            career_vector[idx] = 1.0 / len(holland_codes)  # Equal weight
        
        # Compute cosine similarity
        similarity = np.dot(holland_vector, career_vector) / (
            np.linalg.norm(holland_vector) * np.linalg.norm(career_vector) + 1e-8
        )
        
        career_scores.append((career, float(similarity)))
    
    # Sort by score
    career_scores.sort(key=lambda x: x[1], reverse=True)
    
    return career_scores[:top_k]


def cold_start_onboarding(quiz_responses: dict[int, int]) -> dict[str, any]:
    """
    Complete cold start onboarding flow.
    
    Args:
        quiz_responses: User's quiz responses (question_idx → 1-5 rating)
    
    Returns:
        Dictionary with Holland vector, synthetic interests, initial recommendations
    """
    # Compute Holland vector
    holland_vector = compute_holland_vector(quiz_responses)
    
    # Generate interest text
    interest_text = generate_interest_text_from_holland(holland_vector)
    
    # Get initial recommendations
    recommendations = recommend_careers_from_holland(holland_vector, top_k=5)
    
    LOGGER.info(f"Cold start onboarding complete for Holland profile: {holland_vector}")
    
    return {
        "holland_vector": holland_vector.tolist(),
        "interest_text": interest_text,
        "initial_recommendations": recommendations
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: User completes quiz
    example_responses = {
        0: 4,  # Realistic Q1: 4/5
        1: 3,  # Realistic Q2: 3/5
        2: 5,  # Investigative Q1: 5/5
        3: 5,  # Investigative Q2: 5/5
        4: 2,  # Artistic Q1: 2/5
        5: 2,  # Artistic Q2: 2/5
        6: 3,  # Social Q1: 3/5
        7: 3,  # Social Q2: 3/5
        8: 4,  # Enterprising Q1: 4/5
        9: 3,  # Enterprising Q2: 3/5
    }
    
    result = cold_start_onboarding(example_responses)
    
    print("\n=== Cold Start Onboarding Results ===")
    print(f"Holland Vector: {result['holland_vector']}")
    print(f"Interest Text: {result['interest_text']}")
    print("\nInitial Recommendations:")
    for career, score in result['initial_recommendations']:
        print(f"  {career}: {score:.3f}")
