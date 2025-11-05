# Recommendation Explanation System - Design Specification

## Overview
This specification defines a deterministic, auditable system for generating personalized recommendation explanations. The system provides transparent reasoning for career recommendations and actionable skill development guidance.

---

## 1. Input Schema

### 1.1 User Profile Context
```python
{
  "user_skills": ["python", "sql", "data analysis", "statistics"],
  "user_interests": ["machine learning", "research"],
  "user_education": "bachelor's",  # ordinal: 0-6
  "user_gpa": 3.4,  # normalized: 0-1
  "user_experience_years": 3,
  "user_past_roles": ["Data Analyst", "Business Analyst"],
  "user_age": 26
}
```

### 1.2 Target Career Context
```python
{
  "career_title": "Senior Data Scientist",
  "required_skills": ["python", "machine learning", "deep learning", "spark", "statistics"],
  "required_education": "master's",  # ordinal: 0-6
  "typical_gpa": 3.6,
  "required_experience_years": 5,
  "career_centrality": 0.82  # PageRank in career graph
}
```

### 1.3 Model Attribution Weights
```python
{
  "skill_similarity": 0.78,  # cosine(user_skill_emb, career_skill_emb)
  "interest_alignment": 0.65,
  "education_match": 0.90,
  "gpa_proximity": 0.85,
  "experience_fit": 0.60,
  "career_graph_proximity": 0.72,
  "rerank_score": 0.88  # final MLP output
}
```

---

## 2. Explanation Generation Logic

### 2.1 "Why Recommended" (Top-3 Factors)

**Objective**: Surface the strongest positive signals that justify the recommendation.

**Algorithm**:
```python
def generate_why_recommended(
    user_profile: dict,
    career: dict,
    weights: dict
) -> list[str]:
    """
    Generate top-3 reasons for recommendation.
    
    Returns:
        List of 3 human-readable explanation strings, ordered by impact.
    """
    factors = []
    
    # Factor 1: Skill Overlap
    if weights["skill_similarity"] >= 0.7:
        matching_skills = set(user_profile["user_skills"]) & set(career["required_skills"])
        if len(matching_skills) >= 3:
            top_skills = list(matching_skills)[:3]
            skill_str = ", ".join([f"'{s}'" for s in top_skills])
            factors.append({
                "score": weights["skill_similarity"],
                "text": f"Strong skill alignment in {skill_str}, matching {len(matching_skills)} of {len(career['required_skills'])} required competencies"
            })
        elif len(matching_skills) == 2:
            skill_str = " and ".join([f"'{s}'" for s in matching_skills])
            factors.append({
                "score": weights["skill_similarity"],
                "text": f"Core competencies in {skill_str} align with role requirements"
            })
        elif len(matching_skills) == 1:
            skill = list(matching_skills)[0]
            factors.append({
                "score": weights["skill_similarity"],
                "text": f"Foundational expertise in '{skill}' provides entry point to this role"
            })
    
    # Factor 2: Interest-Career Alignment
    if weights["interest_alignment"] >= 0.6:
        # Extract interest keywords from embeddings or explicit text
        interest_keywords = user_profile.get("user_interests", [])
        if interest_keywords:
            primary_interest = interest_keywords[0]
            factors.append({
                "score": weights["interest_alignment"],
                "text": f"Your passion for {primary_interest} strongly correlates with typical {career['career_title']} responsibilities"
            })
    
    # Factor 3: Career Trajectory Fit
    if weights["career_graph_proximity"] >= 0.65:
        if user_profile.get("user_past_roles"):
            most_recent = user_profile["user_past_roles"][-1]
            factors.append({
                "score": weights["career_graph_proximity"],
                "text": f"Natural progression from {most_recent} based on 500+ observed career transitions in our network"
            })
    
    # Factor 4: Education Qualification
    if weights["education_match"] >= 0.8:
        user_ed = user_profile["user_education"]
        req_ed = career["required_education"]
        if education_ordinal(user_ed) >= education_ordinal(req_ed):
            factors.append({
                "score": weights["education_match"],
                "text": f"Your {user_ed} degree meets or exceeds typical educational requirements"
            })
        elif education_ordinal(user_ed) == education_ordinal(req_ed) - 1:
            factors.append({
                "score": weights["education_match"],
                "text": f"Your {user_ed} background provides 80% of required educational foundation"
            })
    
    # Factor 5: Experience Level Proximity
    if weights["experience_fit"] >= 0.55:
        exp_gap = career["required_experience_years"] - user_profile["user_experience_years"]
        if exp_gap <= 1:
            factors.append({
                "score": weights["experience_fit"],
                "text": f"Your {user_profile['user_experience_years']} years of experience aligns well with role expectations"
            })
        elif exp_gap <= 2:
            factors.append({
                "score": weights["experience_fit"],
                "text": f"Within typical experience range; {user_profile['user_experience_years']} years provides solid foundation for growth"
            })
    
    # Factor 6: High-Demand Career
    if weights.get("career_centrality", 0) >= 0.75:
        factors.append({
            "score": weights["career_centrality"],
            "text": f"High-demand role with strong career mobility (top 25% of career network centrality)"
        })
    
    # Sort by score descending and take top 3
    factors.sort(key=lambda x: x["score"], reverse=True)
    return [f["text"] for f in factors[:3]]
```

**Template Rules**:
1. **Specificity**: Always include concrete numbers, skill names, or transition paths
2. **Quantification**: Use percentages, counts, or comparative metrics where possible
3. **Confidence Bounds**: For probabilistic statements, cite data source size (e.g., "500+ transitions")
4. **Avoid Generics**: Never use vague terms like "good fit" without quantification

**Example Output**:
```json
[
  "Strong skill alignment in 'python', 'statistics', 'sql', matching 3 of 5 required competencies",
  "Natural progression from Data Analyst based on 500+ observed career transitions in our network",
  "Your bachelor's degree meets or exceeds typical educational requirements"
]
```

---

### 2.2 "What to Learn Next" (Top-5 Skills)

**Objective**: Prioritize missing skills by expected impact on career transition probability.

**Algorithm**:
```python
def generate_learning_plan(
    user_profile: dict,
    career: dict,
    weights: dict,
    skill_importance_model: callable  # Trained feature importance or heuristic
) -> list[dict]:
    """
    Generate top-5 skill development recommendations.
    
    Returns:
        List of dicts with skill name, impact score, learning resources.
    """
    user_skills = set(user_profile["user_skills"])
    required_skills = set(career["required_skills"])
    
    # Identify skill gaps
    missing_skills = required_skills - user_skills
    
    if not missing_skills:
        return []  # User already has all required skills
    
    # Score each missing skill by impact
    skill_scores = []
    for skill in missing_skills:
        # Impact components:
        # 1. Skill frequency in target role (from corpus)
        frequency_score = get_skill_frequency_in_role(skill, career["career_title"])
        
        # 2. Skill centrality in career graph (co-occurrence)
        centrality_score = get_skill_centrality(skill)
        
        # 3. Transferability from user's existing skills
        transfer_score = max([
            skill_similarity(skill, user_skill) 
            for user_skill in user_skills
        ], default=0.0)
        
        # 4. Market demand signal (job postings, salary correlation)
        demand_score = get_market_demand(skill, career["career_title"])
        
        # Weighted composite impact
        impact = (
            0.35 * frequency_score +
            0.25 * centrality_score +
            0.20 * (1 - transfer_score) +  # Prioritize novel skills
            0.20 * demand_score
        )
        
        skill_scores.append({
            "skill": skill,
            "impact": impact,
            "frequency": frequency_score,
            "transfer_difficulty": 1 - transfer_score,
            "demand": demand_score
        })
    
    # Rank by impact
    skill_scores.sort(key=lambda x: x["impact"], reverse=True)
    
    # Generate explanations for top 5
    recommendations = []
    for idx, item in enumerate(skill_scores[:5], start=1):
        skill = item["skill"]
        
        # Contextual reasoning
        if item["frequency"] >= 0.8:
            reason = f"Core requirement (appears in 80%+ of {career['career_title']} roles)"
        elif item["demand"] >= 0.75:
            reason = f"High market demand (top quartile for salary impact)"
        elif item["transfer_difficulty"] <= 0.3:
            reason = f"Natural extension of your '{closest_user_skill(skill, user_skills)}' expertise"
        else:
            reason = f"Bridges gap between current skillset and role expectations"
        
        # Learning time estimate (heuristic)
        if item["transfer_difficulty"] <= 0.3:
            learning_time = "2-4 weeks (fast upskill)"
        elif item["transfer_difficulty"] <= 0.6:
            learning_time = "2-3 months (moderate investment)"
        else:
            learning_time = "6+ months (foundational build)"
        
        recommendations.append({
            "rank": idx,
            "skill": skill,
            "impact_score": round(item["impact"], 2),
            "reason": reason,
            "estimated_learning_time": learning_time,
            "resources": generate_learning_resources(skill)
        })
    
    return recommendations
```

**Skill Impact Heuristics** (for MVP without trained importance model):
```python
# Fallback when ML feature importance unavailable
SKILL_FREQUENCY_CORPUS = {
    # Manually curated from dataset analysis
    "python": {"Data Scientist": 0.95, "ML Engineer": 0.92},
    "machine learning": {"Data Scientist": 0.88, "ML Engineer": 0.90},
    "deep learning": {"ML Engineer": 0.85, "Data Scientist": 0.65},
    # ... etc
}

SKILL_TRANSFERABILITY = {
    # (source_skill, target_skill): similarity_score
    ("sql", "spark"): 0.65,
    ("python", "r"): 0.70,
    ("statistics", "machine learning"): 0.60,
    # ... etc
}

MARKET_DEMAND_SIGNALS = {
    # From job posting scraping or salary data
    "python": 0.90,
    "kubernetes": 0.82,
    "react": 0.78,
    # ... etc
}
```

**Template Rules**:
1. **Actionability**: Every skill must include concrete learning time estimate
2. **Prioritization Transparency**: Explain WHY this skill ranks where it does
3. **Personalization**: Reference user's existing skills for transfer learning context
4. **Resource Sufficiency**: Provide at least 2 learning resources per skill

**Example Output**:
```json
[
  {
    "rank": 1,
    "skill": "machine learning",
    "impact_score": 0.87,
    "reason": "Core requirement (appears in 80%+ of Senior Data Scientist roles)",
    "estimated_learning_time": "2-3 months (moderate investment)",
    "resources": [
      {"type": "course", "name": "Stanford CS229", "url": "https://..."},
      {"type": "book", "name": "Hands-On ML with Scikit-Learn", "isbn": "..."}
    ]
  },
  {
    "rank": 2,
    "skill": "deep learning",
    "impact_score": 0.74,
    "reason": "High market demand (top quartile for salary impact)",
    "estimated_learning_time": "6+ months (foundational build)",
    "resources": [
      {"type": "course", "name": "fast.ai", "url": "https://..."}
    ]
  },
  {
    "rank": 3,
    "skill": "spark",
    "impact_score": 0.68,
    "reason": "Natural extension of your 'sql' expertise",
    "estimated_learning_time": "2-4 weeks (fast upskill)",
    "resources": [
      {"type": "tutorial", "name": "PySpark Tutorial", "url": "https://..."}
    ]
  }
]
```

---

## 3. Anti-Patterns to Avoid

### 3.1 Generic Wording
❌ **Bad**: "You're a good fit for this role"  
✅ **Good**: "Your 3 years as Data Analyst plus 'python' and 'sql' expertise match 60% of required competencies"

### 3.2 Unexplained Metrics
❌ **Bad**: "High compatibility score"  
✅ **Good**: "88% match based on skill overlap (0.78), education fit (0.90), and career graph proximity (0.72)"

### 3.3 Unmeasurable Claims
❌ **Bad**: "Many people transition to this role"  
✅ **Good**: "Natural progression from Data Analyst based on 500+ observed career transitions in our network"

### 3.4 Vague Learning Paths
❌ **Bad**: "Learn machine learning to improve"  
✅ **Good**: "Master 'machine learning' (2-3 months) - core requirement in 88% of Senior Data Scientist roles"

---

## 4. Audit Trail Requirements

Every explanation must be traceable to:
1. **Feature Values**: Exact user/career attributes used
2. **Model Weights**: Specific contribution scores from retrieval + re-ranker
3. **Template Logic**: Which conditional branch generated the text
4. **Timestamp**: When explanation was generated (for A/B testing)

**Audit Log Schema**:
```python
{
  "user_id": "user_12345",
  "career_id": "career_789",
  "timestamp": "2025-11-05T14:23:45Z",
  "model_version": "reranker_v1.2",
  "feature_values": {
    "skill_similarity": 0.78,
    "interest_alignment": 0.65,
    # ... all features
  },
  "explanation_template_ids": ["skill_overlap_high", "career_graph_natural_progression"],
  "generated_text": {
    "why_recommended": [...],
    "learning_plan": [...]
  }
}
```

---

## 5. Implementation Pseudo-Code

```python
# ml/models/explainer.py

from dataclasses import dataclass
from typing import Any


@dataclass
class ExplanationResult:
    why_recommended: list[str]
    learning_plan: list[dict[str, Any]]
    audit_log: dict[str, Any]


class RecommendationExplainer:
    def __init__(
        self,
        skill_corpus: dict,
        transferability_matrix: dict,
        market_signals: dict
    ):
        self.skill_corpus = skill_corpus
        self.transferability = transferability_matrix
        self.market_signals = market_signals
    
    def explain(
        self,
        user_profile: dict,
        career: dict,
        model_weights: dict
    ) -> ExplanationResult:
        """Generate complete explanation package."""
        
        # Component 1: Why recommended
        why_factors = self._generate_why_recommended(
            user_profile, career, model_weights
        )
        
        # Component 2: Learning plan
        learning_plan = self._generate_learning_plan(
            user_profile, career, model_weights
        )
        
        # Component 3: Audit trail
        audit_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "feature_values": model_weights,
            "template_ids": self._used_templates,
            "generated_text": {
                "why_recommended": why_factors,
                "learning_plan": learning_plan
            }
        }
        
        return ExplanationResult(
            why_recommended=why_factors,
            learning_plan=learning_plan,
            audit_log=audit_log
        )
    
    def _generate_why_recommended(self, user, career, weights) -> list[str]:
        # Implementation from Section 2.1
        pass
    
    def _generate_learning_plan(self, user, career, weights) -> list[dict]:
        # Implementation from Section 2.2
        pass
```

---

## 6. Validation & Testing

### 6.1 Unit Tests
```python
def test_explanation_specificity():
    """Ensure no generic wording appears in output."""
    result = explainer.explain(user, career, weights)
    
    for text in result.why_recommended:
        assert not contains_generic_phrase(text)
        assert contains_quantification(text)
        assert len(text.split()) >= 10  # Minimum detail threshold

def test_learning_plan_actionability():
    """Verify all skills have resources and time estimates."""
    result = explainer.explain(user, career, weights)
    
    for item in result.learning_plan:
        assert "estimated_learning_time" in item
        assert "resources" in item
        assert len(item["resources"]) >= 2
```

### 6.2 Human Evaluation Rubric
- **Clarity**: Can user understand WHY without domain expertise? (1-5 scale)
- **Actionability**: Can user act on learning plan immediately? (1-5 scale)
- **Trust**: Are quantifications believable and sourced? (1-5 scale)
- **Personalization**: Does explanation reference user's actual profile? (1-5 scale)

Target: ≥4.0 average across all dimensions

---

## 7. Future Enhancements

1. **SHAP-based Attribution**: Replace heuristic weights with SHAP values from re-ranker
2. **Dynamic Resource Curation**: Pull learning resources from live APIs (Coursera, Udemy)
3. **A/B Testing Framework**: Experiment with explanation phrasing to optimize conversion
4. **Multilingual Support**: Template system for i18n
5. **Interactive Refinement**: Let users ask "why not?" for rejected careers

---

## 8. Example End-to-End Flow

**Input**:
```json
{
  "user": {
    "skills": ["python", "sql", "pandas"],
    "education": "bachelor's",
    "experience_years": 2
  },
  "career": {
    "title": "Machine Learning Engineer",
    "required_skills": ["python", "tensorflow", "kubernetes", "mlops"]
  },
  "weights": {
    "skill_similarity": 0.45,
    "career_graph_proximity": 0.78
  }
}
```

**Output**:
```json
{
  "why_recommended": [
    "Natural progression from your current role based on 350+ observed Data Analyst → ML Engineer transitions",
    "Foundational expertise in 'python' provides entry point to this role",
    "Your 2 years of experience within typical range for junior ML positions"
  ],
  "learning_plan": [
    {
      "rank": 1,
      "skill": "tensorflow",
      "impact_score": 0.89,
      "reason": "Core requirement (appears in 85%+ of ML Engineer roles)",
      "estimated_learning_time": "2-3 months (moderate investment)"
    },
    {
      "rank": 2,
      "skill": "kubernetes",
      "impact_score": 0.76,
      "reason": "High market demand (top quartile for salary impact)",
      "estimated_learning_time": "6+ months (foundational build)"
    }
  ]
}
```

---

**End of Specification**
