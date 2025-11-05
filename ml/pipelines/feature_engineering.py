"""Feature engineering for user profiles and career descriptions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .embedder import embed_text

LOGGER = logging.getLogger(__name__)

# Education hierarchy for ordinal encoding
EDUCATION_HIERARCHY = {
    "some high school": 0,
    "high school": 1,
    "ged": 1,
    "some college": 2,
    "associate degree": 3,
    "associate's": 3,
    "bachelor's": 4,
    "bachelor": 4,
    "bs": 4,
    "ba": 4,
    "master's": 5,
    "master": 5,
    "ms": 5,
    "ma": 5,
    "mba": 5,
    "phd": 6,
    "doctorate": 6,
    "bootcamp": 4,  # Equivalent to bachelor's
    "certificate": 2,
    "certification": 2
}


@dataclass
class UserProfile:
    """User profile input schema."""
    
    age: int
    education_level: str
    grades_gpa: float | None = None
    interests: str | None = None
    skills: str | None = None
    past_roles: list[str] | None = None
    location: str | None = None
    years_experience: int = 0


def encode_education(education_level: str) -> np.ndarray:
    """
    Encode education level as ordinal + one-hot.
    
    Args:
        education_level: Education level string
    
    Returns:
        8-dimensional feature vector (1 ordinal + 7 one-hot)
    """
    # Normalize
    level = education_level.lower().strip()
    
    # Get ordinal value
    ordinal = EDUCATION_HIERARCHY.get(level, 2)  # Default: Some College
    
    # Create one-hot vector
    onehot = np.zeros(7, dtype=np.float32)
    onehot[ordinal] = 1.0
    
    # Combine: [normalized_ordinal, one_hot_0, ..., one_hot_6]
    return np.concatenate([
        [ordinal / 6.0],  # Normalize to 0-1
        onehot
    ])


def normalize_grades(gpa: float | None, scale: str = "4.0") -> float:
    """
    Normalize GPA to 0-1 scale.
    
    Args:
        gpa: Grade point average
        scale: Grading scale ("4.0" or "100")
    
    Returns:
        Normalized GPA score
    """
    if gpa is None or pd.isna(gpa):
        return 0.5  # Neutral default
    
    if scale == "4.0":
        return min(float(gpa) / 4.0, 1.0)
    elif scale == "100":
        return min(float(gpa) / 100.0, 1.0)
    else:
        return 0.5


def bucket_performance(gpa_normalized: float) -> int:
    """
    Convert normalized GPA to performance bucket.
    
    Args:
        gpa_normalized: GPA on 0-1 scale
    
    Returns:
        Performance bucket (0=Below Avg, 1=Average, 2=Above Avg, 3=Excellent)
    """
    if gpa_normalized < 0.6:
        return 0
    elif gpa_normalized < 0.75:
        return 1
    elif gpa_normalized < 0.9:
        return 2
    else:
        return 3


def encode_age(age: int) -> np.ndarray:
    """
    Encode age as one-hot buckets.
    
    Args:
        age: User age in years
    
    Returns:
        5-dimensional one-hot vector (<25, 25-35, 35-45, 45-55, 55+)
    """
    age_bucket = min((age - 15) // 10, 4) if age >= 15 else 0
    age_bucket = max(0, age_bucket)
    
    onehot = np.zeros(5, dtype=np.float32)
    onehot[age_bucket] = 1.0
    
    return onehot


def encode_past_roles(
    past_roles: list[str] | None,
    job_embeddings: dict[str, np.ndarray],
    embedding_dim: int = 128
) -> np.ndarray:
    """
    Aggregate past role embeddings with recency weighting.
    
    Args:
        past_roles: List of job titles (most recent first)
        job_embeddings: Dictionary of job title -> Node2Vec embedding
        embedding_dim: Dimension of job embeddings
    
    Returns:
        Weighted average embedding vector
    """
    if not past_roles or past_roles is None:
        return np.zeros(embedding_dim, dtype=np.float32)
    
    valid_embeddings = []
    for role in past_roles:
        if role in job_embeddings:
            valid_embeddings.append(job_embeddings[role])
        else:
            # Use zero vector for unknown roles
            valid_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
    
    if not valid_embeddings:
        return np.zeros(embedding_dim, dtype=np.float32)
    
    # Exponential decay weights (recent roles weighted higher)
    weights = np.exp(-0.3 * np.arange(len(valid_embeddings)))
    weights = weights / weights.sum()  # Normalize
    
    # Weighted average
    weighted_avg = np.average(valid_embeddings, axis=0, weights=weights)
    
    return weighted_avg.astype(np.float32)


def build_user_vector(
    user: UserProfile,
    job_embeddings: dict[str, np.ndarray],
    skill_embedding_dim: int = 384,
    career_embedding_dim: int = 128
) -> np.ndarray:
    """
    Construct complete user feature vector.
    
    Feature composition:
    - skill_embedding: 384
    - interest_embedding: 384
    - education_features: 8
    - gpa_features: 5 (1 normalized + 4 bucket onehot)
    - experience_embedding: 128
    - age_features: 5
    
    Total: 914 dimensions
    
    Args:
        user: User profile object
        job_embeddings: Career node embeddings from Node2Vec
        skill_embedding_dim: Dimension of skill embeddings
        career_embedding_dim: Dimension of career embeddings
    
    Returns:
        Complete user feature vector
    """
    # Text embeddings
    skill_emb = embed_text(user.skills)
    interest_emb = embed_text(user.interests)
    
    # Education features (8 dims)
    edu_features = encode_education(user.education_level)
    
    # GPA features (5 dims)
    gpa_norm = normalize_grades(user.grades_gpa)
    perf_bucket = bucket_performance(gpa_norm)
    perf_onehot = np.zeros(4, dtype=np.float32)
    perf_onehot[perf_bucket] = 1.0
    gpa_features = np.concatenate([[gpa_norm], perf_onehot])
    
    # Experience embedding (128 dims)
    exp_emb = encode_past_roles(user.past_roles, job_embeddings, career_embedding_dim)
    
    # Age features (5 dims)
    age_features = encode_age(user.age)
    
    # Concatenate all features
    user_vector = np.concatenate([
        skill_emb,          # 384
        interest_emb,       # 384
        edu_features,       # 8
        gpa_features,       # 5
        exp_emb,           # 128
        age_features       # 5
    ])
    
    return user_vector.astype(np.float32)


def build_career_vector(
    career_title: str,
    career_embedding: np.ndarray,
    required_skills: str | None = None,
    education_requirement: int = 2,
    typical_gpa: float = 3.0,
    centrality_score: float = 0.5,
    skill_embedding_dim: int = 384
) -> np.ndarray:
    """
    Construct career feature vector.
    
    Feature composition:
    - career_embedding: 128 (from Node2Vec)
    - skill_embedding: 384 (from required skills text)
    - education_requirement: 1
    - typical_gpa: 1
    - centrality_score: 1
    - metadata_features: 3
    
    Total: 518 dimensions
    
    Args:
        career_title: Job title
        career_embedding: Node2Vec embedding for this career
        required_skills: Text description of required skills
        education_requirement: Minimum education ordinal (0-6)
        typical_gpa: Average GPA of successful candidates
        centrality_score: PageRank centrality in career graph
        skill_embedding_dim: Dimension of skill embeddings
    
    Returns:
        Complete career feature vector
    """
    # Career graph embedding
    if career_embedding is None or len(career_embedding) == 0:
        career_emb = np.zeros(128, dtype=np.float32)
    else:
        career_emb = career_embedding.astype(np.float32)
    
    # Required skills embedding
    skill_emb = embed_text(required_skills) if required_skills else np.zeros(skill_embedding_dim, dtype=np.float32)
    
    # Metadata features
    metadata = np.array([
        education_requirement / 6.0,  # Normalize
        typical_gpa / 4.0,             # Normalize
        centrality_score               # Already 0-1
    ], dtype=np.float32)
    
    # Concatenate
    career_vector = np.concatenate([
        career_emb,      # 128
        skill_emb,       # 384
        metadata         # 3
    ])
    
    return career_vector.astype(np.float32)


def batch_build_user_vectors(
    users_df: pd.DataFrame,
    job_embeddings: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Build user vectors for multiple users efficiently.
    
    Args:
        users_df: DataFrame with columns matching UserProfile fields
        job_embeddings: Career embeddings
    
    Returns:
        Array of shape (n_users, 914)
    """
    vectors = []
    
    for _, row in users_df.iterrows():
        user = UserProfile(
            age=row.get("age", 25),
            education_level=row.get("education_level", "Bachelor's"),
            grades_gpa=row.get("grades_gpa"),
            interests=row.get("interests"),
            skills=row.get("skills"),
            past_roles=row.get("past_roles", []) if pd.notna(row.get("past_roles")) else [],
            years_experience=row.get("years_experience", 0)
        )
        
        vector = build_user_vector(user, job_embeddings)
        vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)
