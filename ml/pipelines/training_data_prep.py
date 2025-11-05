"""Prepare training data for the career re-ranker model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_catalog import load_all_datasets
from .feature_engineering import (
    UserProfile,
    build_career_vector,
    build_user_vector
)
from .graph_builder import load_graph_artifacts

LOGGER = logging.getLogger(__name__)


def load_preference_data(catalog: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load and prepare user-career preference data.
    
    Args:
        catalog: Dictionary of dataset name -> DataFrame
    
    Returns:
        DataFrame with user_id, career_title, label (1=selected, 0=not)
    """
    # Primary source: AI-based Career Recommendation System.csv
    if "ai_based_career_recommendation_system" in catalog:
        df = catalog["ai_based_career_recommendation_system"].copy()
        LOGGER.info(f"Loaded {len(df)} preference records from AI-based Career Recommendation System")
        return df
    
    # Fallback: education_career_success
    if "education_career_success" in catalog:
        df = catalog["education_career_success"].copy()
        LOGGER.info(f"Loaded {len(df)} records from education_career_success")
        return df
    
    raise ValueError("No suitable preference data found in catalog")


def extract_positive_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract positive training examples (user → selected career).
    
    Args:
        df: Raw preference data
    
    Returns:
        DataFrame with positive user-career pairs
    """
    # Filter for explicit selections/successes
    # Assuming columns like: user_id, career_title, selected, success_score, etc.
    
    # Heuristic: Look for boolean/binary selection columns
    positive_indicators = ["selected", "accepted", "success", "hired", "enrolled"]
    
    label_col = None
    for col in df.columns:
        if any(indicator in col.lower() for indicator in positive_indicators):
            label_col = col
            break
    
    if label_col:
        positives = df[df[label_col] == 1].copy()
        LOGGER.info(f"Found {len(positives)} positive examples using column '{label_col}'")
    else:
        # If no explicit label, treat all as positive
        positives = df.copy()
        LOGGER.warning(f"No label column found; treating all {len(positives)} rows as positive")
    
    positives["label"] = 1
    return positives


def generate_hard_negatives(
    positives: pd.DataFrame,
    catalog: dict[str, pd.DataFrame],
    n_negatives_per_user: int = 3
) -> pd.DataFrame:
    """
    Generate hard negative examples (similar careers not selected).
    
    Strategy:
    1. For each user, identify their positive career(s)
    2. Find similar careers (same field, different role) they did NOT select
    3. Sample negatives to balance dataset
    
    Args:
        positives: Positive user-career pairs
        catalog: All datasets
        n_negatives_per_user: Number of negatives per user
    
    Returns:
        DataFrame with negative user-career pairs
    """
    # Get all unique careers from catalog
    all_careers = set()
    for df_name, df in catalog.items():
        career_cols = [col for col in df.columns if "career" in col.lower() or "job" in col.lower() or "title" in col.lower()]
        for col in career_cols:
            if df[col].dtype == "object":
                all_careers.update(df[col].dropna().unique())
    
    LOGGER.info(f"Found {len(all_careers)} unique careers across all datasets")
    
    negatives = []
    
    user_col = None
    career_col = None
    
    # Identify user and career columns
    for col in positives.columns:
        if "user" in col.lower() or "student" in col.lower() or "person" in col.lower():
            user_col = col
        if "career" in col.lower() or "job" in col.lower() or "title" in col.lower():
            career_col = col
    
    if not user_col or not career_col:
        LOGGER.warning("Could not identify user/career columns; using heuristic negatives")
        # Fallback: random sampling
        user_col = positives.columns[0]
        career_col = positives.columns[1]
    
    # Generate negatives for each user
    for user_id, group in positives.groupby(user_col):
        selected_careers = set(group[career_col])
        
        # Candidate negatives: careers NOT selected
        candidate_negatives = list(all_careers - selected_careers)
        
        if not candidate_negatives:
            continue
        
        # Sample negatives
        n_samples = min(n_negatives_per_user, len(candidate_negatives))
        sampled = np.random.choice(candidate_negatives, size=n_samples, replace=False)
        
        for career in sampled:
            neg_row = group.iloc[0].copy()  # Copy user features
            neg_row[career_col] = career
            neg_row["label"] = 0
            negatives.append(neg_row)
    
    negatives_df = pd.DataFrame(negatives)
    LOGGER.info(f"Generated {len(negatives_df)} hard negative examples")
    
    return negatives_df


def merge_features_and_labels(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    job_embeddings: dict[str, np.ndarray],
    career_metadata: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature vectors for all user-career pairs.
    
    Args:
        positives: Positive examples
        negatives: Negative examples
        job_embeddings: Career Node2Vec embeddings
        career_metadata: Career metadata (education req, GPA, centrality)
    
    Returns:
        Tuple of (user_vectors, career_vectors, labels)
    """
    # Combine positives and negatives
    all_data = pd.concat([positives, negatives], ignore_index=True)
    
    # Identify columns
    user_col = None
    career_col = None
    for col in all_data.columns:
        if "user" in col.lower() or "student" in col.lower():
            user_col = col
        if "career" in col.lower() or "job" in col.lower() or "title" in col.lower():
            career_col = col
    
    if not user_col or not career_col:
        raise ValueError("Cannot identify user and career columns")
    
    user_vectors = []
    career_vectors = []
    labels = []
    
    for _, row in all_data.iterrows():
        # Build user profile
        user = UserProfile(
            age=row.get("age", 25),
            education_level=row.get("education_level", "Bachelor's"),
            grades_gpa=row.get("gpa") or row.get("grades_gpa"),
            interests=row.get("interests") or row.get("interest"),
            skills=row.get("skills") or row.get("skill"),
            past_roles=row.get("past_roles", []),
            years_experience=row.get("years_experience", 0)
        )
        
        # Build user vector
        user_vec = build_user_vector(user, job_embeddings)
        
        # Get career embedding
        career_title = row[career_col]
        career_emb = job_embeddings.get(career_title, np.zeros(128))
        
        # Get career metadata
        metadata = career_metadata.get(career_title, {})
        
        # Build career vector
        career_vec = build_career_vector(
            career_title=career_title,
            career_embedding=career_emb,
            required_skills=metadata.get("required_skills"),
            education_requirement=metadata.get("education_requirement", 2),
            typical_gpa=metadata.get("typical_gpa", 3.0),
            centrality_score=metadata.get("centrality", 0.5)
        )
        
        user_vectors.append(user_vec)
        career_vectors.append(career_vec)
        labels.append(row["label"])
    
    return (
        np.array(user_vectors, dtype=np.float32),
        np.array(career_vectors, dtype=np.float32),
        np.array(labels, dtype=np.float32)
    )


def split_data(
    user_vectors: np.ndarray,
    career_vectors: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test sets.
    
    Args:
        user_vectors: User feature arrays
        career_vectors: Career feature arrays
        labels: Target labels
        train_size: Proportion for training
        val_size: Proportion for validation
        random_state: Random seed
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    # First split: train vs (val + test)
    X_train_u, X_temp_u, X_train_c, X_temp_c, y_train, y_temp = train_test_split(
        user_vectors,
        career_vectors,
        labels,
        train_size=train_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio = val_size / (1 - train_size)
    X_val_u, X_test_u, X_val_c, X_test_c, y_val, y_test = train_test_split(
        X_temp_u,
        X_temp_c,
        y_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    LOGGER.info(f"Split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    return {
        "train": (X_train_u, X_train_c, y_train),
        "val": (X_val_u, X_val_c, y_val),
        "test": (X_test_u, X_test_c, y_test)
    }


def prepare_training_data(
    artifacts_dir: str | Path = "ml/artifacts",
    n_negatives_per_user: int = 3,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Main pipeline to prepare training data.
    
    Steps:
    1. Load all datasets via catalog
    2. Load graph artifacts (Node2Vec embeddings, centrality)
    3. Extract positive examples
    4. Generate hard negatives
    5. Build feature vectors
    6. Split into train/val/test
    
    Args:
        artifacts_dir: Directory with graph artifacts
        n_negatives_per_user: Negatives per user
        train_size: Training set proportion
        val_size: Validation set proportion
        random_state: Random seed
    
    Returns:
        Dictionary with train/val/test splits
    """
    artifacts_path = Path(artifacts_dir)
    
    # Load datasets
    LOGGER.info("Loading datasets...")
    catalog = load_all_datasets()
    
    # Load graph artifacts
    LOGGER.info("Loading graph artifacts...")
    graph_artifacts = load_graph_artifacts(artifacts_path)
    job_embeddings = graph_artifacts["embeddings"]
    centrality_scores = graph_artifacts["centrality"]
    
    # Build career metadata
    career_metadata = {}
    for career, centrality in centrality_scores.items():
        career_metadata[career] = {
            "centrality": centrality,
            "education_requirement": 2,  # Default: Some College
            "typical_gpa": 3.0,
            "required_skills": None
        }
    
    # Load preference data
    LOGGER.info("Loading preference data...")
    pref_data = load_preference_data(catalog)
    
    # Extract positives
    LOGGER.info("Extracting positive examples...")
    positives = extract_positive_examples(pref_data)
    
    # Generate negatives
    LOGGER.info("Generating hard negatives...")
    negatives = generate_hard_negatives(positives, catalog, n_negatives_per_user)
    
    # Build features
    LOGGER.info("Building feature vectors...")
    user_vecs, career_vecs, labels = merge_features_and_labels(
        positives, negatives, job_embeddings, career_metadata
    )
    
    # Split data
    LOGGER.info("Splitting data...")
    splits = split_data(user_vecs, career_vecs, labels, train_size, val_size, random_state)
    
    LOGGER.info("Training data preparation complete!")
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    splits = prepare_training_data()
    
    # Save splits to disk
    save_dir = Path("ml/artifacts/training_data")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, (user_vecs, career_vecs, labels) in splits.items():
        np.savez(
            save_dir / f"{split_name}.npz",
            user_vectors=user_vecs,
            career_vectors=career_vecs,
            labels=labels
        )
        LOGGER.info(f"Saved {split_name} split to {save_dir / split_name}.npz")
