# Explanation System Demo

This document demonstrates the recommendation explanation system with concrete examples.

## System Overview

The explanation system provides two key components:
1. **Why Recommended**: Top-3 factors justifying the recommendation
2. **What to Learn Next**: Top-5 skills ranked by impact with learning time estimates

## Example 1: Data Analyst → Data Scientist Transition

### Input: User Profile
```json
{
  "user_id": "user_001",
  "user_skills": ["python", "sql", "statistics", "pandas"],
  "user_interests": ["data analysis", "visualization"],
  "user_education": "bachelor's",
  "user_experience_years": 3,
  "user_past_roles": ["Data Analyst", "Junior Analyst"],
  "user_gpa": 3.6,
  "age": 28
}
```

### Input: Career Recommendation
```json
{
  "career_id": "career_ds_001",
  "career_title": "Data Scientist",
  "required_skills": ["python", "machine learning", "statistics", "sql", "deep learning"],
  "required_education": "bachelor's",
  "required_experience_years": 3,
  "centrality": 0.85
}
```

### Input: Model Weights (from ML pipeline)
```json
{
  "skill_similarity": 0.78,
  "interest_alignment": 0.72,
  "education_match": 0.95,
  "experience_fit": 0.88,
  "career_graph_proximity": 0.70,
  "career_centrality": 0.85,
  "rerank_score": 0.82
}
```

### Output: Explanation Result

#### Why Recommended (Top 3 Factors)
1. **"Your bachelor's degree meets or exceeds typical educational requirements"**
   - Score: 0.95
   - Template: `education_meets_requirement`
   - Logic: `user_education >= required_education` and `education_match >= 0.8`

2. **"Your 3 years of experience aligns well with role expectations"**
   - Score: 0.88
   - Template: `experience_aligned`
   - Logic: `|user_experience - required_experience| <= 1` and `experience_fit >= 0.55`

3. **"Strong skill alignment in 'python', 'sql', 'statistics', matching 3 of 5 required competencies"**
   - Score: 0.78
   - Template: `skill_overlap_high`
   - Logic: `len(user_skills ∩ required_skills) >= 3` and `skill_similarity >= 0.7`

#### What to Learn Next (Top 5 Skills)

**Missing Skills**: `["machine learning", "deep learning"]`

1. **Machine Learning**
   - Impact Score: 0.82
   - Reason: "Core requirement (appears in 80%+ of Data Scientist roles)"
   - Estimated Learning Time: "2-3 months (moderate investment)"
   - Resources:
     - Course: Stanford CS229 (https://cs229.stanford.edu)
     - Book: Hands-On ML with Scikit-Learn (https://www.oreilly.com)
   - Impact Breakdown:
     - Frequency: 0.88 (appears in 88% of Data Scientist roles)
     - Centrality: 0.75 (high co-occurrence with other skills)
     - Transfer Difficulty: 0.40 (moderate - builds on statistics knowledge)
     - Market Demand: 0.88 (top-tier salary impact)

2. **Deep Learning**
   - Impact Score: 0.71
   - Reason: "Natural extension of your 'statistics' expertise"
   - Estimated Learning Time: "2-3 months (moderate investment)"
   - Resources:
     - Course: fast.ai (https://fast.ai)
     - Course: Deep Learning Specialization (https://coursera.org)
   - Impact Breakdown:
     - Frequency: 0.65 (appears in 65% of Data Scientist roles)
     - Centrality: 0.70 (common in ML-focused roles)
     - Transfer Difficulty: 0.25 (low - leverages existing ML knowledge via transferability matrix)
     - Market Demand: 0.80 (strong growth trajectory)

### Audit Trail
```json
{
  "timestamp": "2025-11-05T12:34:56.789Z",
  "user_id": "user_001",
  "career_id": "career_ds_001",
  "career_title": "Data Scientist",
  "feature_values": {
    "skill_similarity": 0.78,
    "interest_alignment": 0.72,
    "education_match": 0.95,
    "experience_fit": 0.88,
    "career_graph_proximity": 0.70,
    "career_centrality": 0.85,
    "rerank_score": 0.82
  },
  "template_ids": [
    "education_meets_requirement",
    "experience_aligned",
    "skill_overlap_high"
  ],
  "generated_text": {
    "why_recommended": [
      "Your bachelor's degree meets or exceeds typical educational requirements",
      "Your 3 years of experience aligns well with role expectations",
      "Strong skill alignment in 'python', 'sql', 'statistics', matching 3 of 5 required competencies"
    ],
    "learning_plan": [...]
  }
}
```

---

## Example 2: Entry-Level → Senior Role (Stretch Recommendation)

### Input: User Profile
```json
{
  "user_id": "user_002",
  "user_skills": ["python", "pandas"],
  "user_interests": ["coding"],
  "user_education": "bachelor's",
  "user_experience_years": 1,
  "user_past_roles": ["Junior Developer"],
  "user_gpa": 3.2,
  "age": 24
}
```

### Input: Career Recommendation
```json
{
  "career_id": "career_mle_001",
  "career_title": "Senior ML Engineer",
  "required_skills": ["machine learning", "deep learning", "tensorflow", "pytorch", "kubernetes"],
  "required_education": "master's",
  "required_experience_years": 5,
  "centrality": 0.92
}
```

### Input: Model Weights
```json
{
  "skill_similarity": 0.25,
  "interest_alignment": 0.45,
  "education_match": 0.65,
  "experience_fit": 0.30,
  "career_graph_proximity": 0.40,
  "career_centrality": 0.92,
  "rerank_score": 0.52
}
```

### Output: Explanation Result

#### Why Recommended (Top 1 Factor)
1. **"High-demand role with strong career mobility (top 25% of career network centrality)"**
   - Score: 0.92
   - Template: `high_centrality`
   - Logic: `career_centrality >= 0.75`

**Note**: Only 1 factor generated because all other weights fall below thresholds. This signals a "stretch" recommendation.

#### What to Learn Next (Top 5 Skills)

**Missing Skills**: `["machine learning", "deep learning", "tensorflow", "pytorch", "kubernetes"]`

1. **Machine Learning**
   - Impact Score: 0.85
   - Reason: "Core requirement (appears in 80%+ of Senior ML Engineer roles)"
   - Estimated Learning Time: "6+ months (foundational build)"
   - Transfer Difficulty: 0.80 (high - requires significant new knowledge)

2. **Deep Learning**
   - Impact Score: 0.78
   - Reason: "High market demand (top quartile for salary impact)"
   - Estimated Learning Time: "6+ months (foundational build)"
   - Transfer Difficulty: 0.85

3. **TensorFlow**
   - Impact Score: 0.70
   - Reason: "Bridges gap between current skillset and role expectations"
   - Estimated Learning Time: "2-3 months (moderate investment)"
   - Transfer Difficulty: 0.55

4. **PyTorch**
   - Impact Score: 0.68
   - Reason: "High market demand (top quartile for salary impact)"
   - Estimated Learning Time: "2-3 months (moderate investment)"
   - Transfer Difficulty: 0.50

5. **Kubernetes**
   - Impact Score: 0.62
   - Reason: "Bridges gap between current skillset and role expectations"
   - Estimated Learning Time: "2-3 months (moderate investment)"
   - Transfer Difficulty: 0.60

---

## Key Design Principles Demonstrated

### 1. **Quantification Over Generics**
- ❌ "You're a good fit"
- ✅ "Strong skill alignment in 'python', 'sql', 'statistics', matching 3 of 5 required competencies"

### 2. **Threshold-Based Logic**
Each factor has specific activation thresholds:
- `skill_similarity >= 0.7` → skill overlap explanations
- `education_match >= 0.8` → education qualification text
- `career_centrality >= 0.75` → high-demand role text

### 3. **Impact-Driven Learning Plans**
Skills ranked by composite score:
```
impact = 0.35 * frequency_score 
       + 0.25 * centrality_score 
       + 0.20 * (1 - transfer_score) 
       + 0.20 * demand_score
```

### 4. **Learning Time Estimates**
Based on transfer difficulty:
- `transfer_difficulty <= 0.3` → "2-4 weeks (fast upskill)"
- `transfer_difficulty <= 0.6` → "2-3 months (moderate investment)"
- `transfer_difficulty > 0.6` → "6+ months (foundational build)"

### 5. **Audit Trail for Traceability**
Every explanation includes:
- Timestamp
- User/career IDs
- Feature values (inputs)
- Template IDs (logic branches taken)
- Generated text (outputs)

---

## Testing the Explanation System

### Quick Test
```python
from ml.models.explainer import RecommendationExplainer

explainer = RecommendationExplainer()

user_profile = {
    "user_id": "test_user",
    "user_skills": ["python", "sql", "statistics"],
    "user_interests": ["data analysis"],
    "user_education": "bachelor's",
    "user_experience_years": 3,
    "user_past_roles": ["Analyst"]
}

career = {
    "career_id": "test_career",
    "career_title": "Data Scientist",
    "required_skills": ["python", "machine learning", "statistics", "sql"],
    "required_education": "bachelor's",
    "required_experience_years": 3,
    "centrality": 0.85
}

model_weights = {
    "skill_similarity": 0.78,
    "interest_alignment": 0.72,
    "education_match": 0.95,
    "experience_fit": 0.88,
    "career_graph_proximity": 0.70,
    "career_centrality": 0.85
}

result = explainer.explain(user_profile, career, model_weights)

print("Why Recommended:")
for reason in result.why_recommended:
    print(f"  - {reason}")

print("\nWhat to Learn Next:")
for item in result.learning_plan:
    print(f"  {item['rank']}. {item['skill']} (impact: {item['impact_score']})")
    print(f"     Reason: {item['reason']}")
    print(f"     Time: {item['estimated_learning_time']}")
```

### Integration with Recommender
```bash
# Generate recommendations with explanations (default)
python -m ml.models.recommender infer --user_json user_profile.json

# Disable explanations for faster inference
python -m ml.models.recommender infer --user_json user_profile.json --no-explanations
```

---

## Validation Metrics

From unit tests (`tests/ml/test_explainer.py`):

✅ **Specificity**: No generic phrases ("good fit", "great match")  
✅ **Quantification**: All explanations contain numbers or quoted skill names  
✅ **Actionability**: Learning plans include concrete time estimates and resources  
✅ **Top-K Limit**: Max 3 "why" factors, max 5 "learning" skills  
✅ **Impact Ranking**: Skills sorted by composite impact score (descending)  
✅ **Traceability**: Audit logs contain timestamp, feature values, template IDs

---

## Production Deployment Notes

1. **Custom Heuristics**: Override default skill corpus/transferability/demand signals with domain-specific data
2. **Dynamic Resources**: Replace placeholder resources with API calls to Coursera/Udemy/LinkedIn Learning
3. **A/B Testing**: Log explanation variants and track user engagement metrics
4. **Human Evaluation**: Periodically sample explanations and score on Clarity/Actionability/Trust/Personalization (target ≥4.0/5)
5. **SHAP Integration**: Future enhancement to use model-agnostic attribution instead of threshold heuristics
