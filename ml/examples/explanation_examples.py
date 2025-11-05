"""End-to-end example demonstrating the complete recommendation + explanation pipeline."""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This example shows how to use the explainer standalone or integrated with the recommender

def example_1_standalone_explainer():
    """Example 1: Using the explainer directly."""
    from ml.models.explainer import RecommendationExplainer
    
    print("=" * 80)
    print("Example 1: Standalone Explainer")
    print("=" * 80)
    
    # Initialize explainer
    explainer = RecommendationExplainer()
    
    # User profile (from frontend or database)
    user_profile = {
        "user_id": "user_123",
        "user_skills": ["python", "sql", "statistics", "pandas"],
        "user_interests": ["data analysis", "visualization"],
        "user_education": "bachelor's",
        "user_experience_years": 3,
        "user_past_roles": ["Data Analyst", "Junior Analyst"],
        "user_gpa": 3.6,
        "age": 28
    }
    
    # Career recommendation (from ML pipeline)
    career = {
        "career_id": "career_ds_001",
        "career_title": "Data Scientist",
        "required_skills": ["python", "machine learning", "statistics", "sql", "deep learning"],
        "required_education": "bachelor's",
        "required_experience_years": 3,
        "centrality": 0.85
    }
    
    # Model weights (from recommender scores)
    model_weights = {
        "skill_similarity": 0.78,
        "interest_alignment": 0.72,
        "education_match": 0.95,
        "experience_fit": 0.88,
        "career_graph_proximity": 0.70,
        "career_centrality": 0.85,
        "rerank_score": 0.82
    }
    
    # Generate explanations
    result = explainer.explain(user_profile, career, model_weights)
    
    # Display results
    print("\n📊 WHY RECOMMENDED:")
    for idx, reason in enumerate(result.why_recommended, 1):
        print(f"  {idx}. {reason}")
    
    print("\n🎯 WHAT TO LEARN NEXT:")
    for item in result.learning_plan:
        print(f"\n  {item['rank']}. {item['skill'].upper()} (Impact: {item['impact_score']:.2f})")
        print(f"     💡 {item['reason']}")
        print(f"     ⏱️  {item['estimated_learning_time']}")
        print(f"     📚 Resources:")
        for resource in item['resources']:
            print(f"        - {resource['type'].title()}: {resource['name']}")
    
    print("\n🔍 AUDIT TRAIL:")
    print(f"   Timestamp: {result.audit_log['timestamp']}")
    print(f"   Templates Used: {', '.join(result.audit_log['template_ids'])}")
    
    return result


def example_2_integrated_with_recommender():
    """Example 2: Using the recommender with integrated explanations."""
    from ml.models.recommender import CareerRecommender, RecommenderConfig
    
    print("\n\n" + "=" * 80)
    print("Example 2: Integrated Recommender + Explainer")
    print("=" * 80)
    
    # Setup (in production, these would already exist)
    config = RecommenderConfig()
    
    # Check if artifacts exist
    if not config.model_path.exists():
        print("\n⚠️  Model artifacts not found. Run training first:")
        print("   python -m ml.models.recommender train")
        return None
    
    # Initialize recommender
    recommender = CareerRecommender(config=config)
    
    # Create user profile JSON (temporary file for demo)
    user_data = {
        "user_id": "user_456",
        "user_skills": ["javascript", "react", "node.js"],
        "user_interests": ["web development", "frontend"],
        "user_education": "bachelor's",
        "user_experience_years": 2,
        "user_past_roles": ["Frontend Developer"],
        "user_gpa": 3.4,
        "age": 26
    }
    
    user_json_path = Path("temp_user_profile.json")
    user_json_path.write_text(json.dumps(user_data), encoding="utf-8")
    
    try:
        # Get recommendations WITH explanations (default)
        print("\n🔮 Generating recommendations with explanations...")
        recommendations = recommender.infer(user_json_path, top_k=3, include_explanations=True)
        
        # Display results
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{'─' * 80}")
            print(f"Recommendation #{idx}: {rec['career_title']}")
            print(f"{'─' * 80}")
            print(f"Retrieval Score: {rec['retrieval_score']:.3f}")
            print(f"Re-rank Score: {rec['rerank_score']:.3f}")
            
            if 'explanations' in rec:
                print("\n📊 Why Recommended:")
                for reason in rec['explanations']['why_recommended']:
                    print(f"  • {reason}")
                
                if rec['explanations']['learning_plan']:
                    print("\n🎯 What to Learn Next:")
                    for item in rec['explanations']['learning_plan'][:3]:  # Show top 3
                        print(f"  {item['rank']}. {item['skill']} — {item['reason']}")
        
        # Compare with fast inference (no explanations)
        print("\n\n⚡ Fast inference without explanations...")
        fast_recommendations = recommender.infer(user_json_path, top_k=3, include_explanations=False)
        
        print(f"Result: {len(fast_recommendations)} recommendations returned")
        print(f"Format: {list(fast_recommendations[0].keys())}")
        
        return recommendations
        
    finally:
        # Cleanup temp file
        if user_json_path.exists():
            user_json_path.unlink()


def example_3_custom_heuristics():
    """Example 3: Using custom skill heuristics for domain-specific explanations."""
    from ml.models.explainer import RecommendationExplainer
    
    print("\n\n" + "=" * 80)
    print("Example 3: Custom Domain-Specific Heuristics")
    print("=" * 80)
    
    # Custom skill corpus for healthcare domain
    healthcare_corpus = {
        "ehr_systems": {"Clinical Informaticist": 0.95, "Healthcare Data Analyst": 0.85},
        "hipaa_compliance": {"Clinical Informaticist": 0.90, "Health IT Manager": 0.88},
        "medical_terminology": {"Clinical Informaticist": 0.92, "Healthcare Data Analyst": 0.75},
        "python": {"Clinical Informaticist": 0.80, "Healthcare Data Analyst": 0.85},
        "sql": {"Clinical Informaticist": 0.75, "Healthcare Data Analyst": 0.90},
    }
    
    # Custom transferability (healthcare-specific)
    healthcare_transfer = {
        ("medical_terminology", "clinical_workflows"): 0.70,
        ("ehr_systems", "health_informatics"): 0.65,
        ("python", "healthcare_analytics"): 0.60,
    }
    
    # Custom market demand (healthcare focus)
    healthcare_demand = {
        "ehr_systems": 0.92,
        "hipaa_compliance": 0.88,
        "medical_terminology": 0.75,
        "healthcare_analytics": 0.85,
    }
    
    # Initialize explainer with custom heuristics
    explainer = RecommendationExplainer(
        skill_corpus=healthcare_corpus,
        transferability_matrix=healthcare_transfer,
        market_signals=healthcare_demand
    )
    
    # Healthcare user profile
    user_profile = {
        "user_id": "health_user_001",
        "user_skills": ["python", "sql", "medical_terminology"],
        "user_interests": ["healthcare IT"],
        "user_education": "master's",
        "user_experience_years": 4,
        "user_past_roles": ["Healthcare Data Analyst"],
    }
    
    # Healthcare career
    career = {
        "career_id": "health_career_001",
        "career_title": "Clinical Informaticist",
        "required_skills": ["ehr_systems", "hipaa_compliance", "medical_terminology", "python"],
        "required_education": "master's",
        "required_experience_years": 4,
        "centrality": 0.88,
    }
    
    model_weights = {
        "skill_similarity": 0.72,
        "interest_alignment": 0.80,
        "education_match": 1.0,
        "experience_fit": 0.92,
        "career_graph_proximity": 0.75,
        "career_centrality": 0.88,
    }
    
    result = explainer.explain(user_profile, career, model_weights)
    
    print("\n📊 Healthcare-Specific Explanations:")
    for idx, reason in enumerate(result.why_recommended, 1):
        print(f"  {idx}. {reason}")
    
    print("\n🎯 Healthcare Skills to Develop:")
    for item in result.learning_plan:
        print(f"  {item['rank']}. {item['skill']} (Impact: {item['impact_score']:.2f})")
        print(f"     {item['reason']}")
    
    return result


def example_4_api_response_format():
    """Example 4: JSON response format for API integration."""
    print("\n\n" + "=" * 80)
    print("Example 4: API Response Format")
    print("=" * 80)
    
    from ml.models.explainer import RecommendationExplainer
    
    explainer = RecommendationExplainer()
    
    user_profile = {
        "user_id": "api_user_789",
        "user_skills": ["java", "spring boot"],
        "user_interests": ["backend development"],
        "user_education": "bachelor's",
        "user_experience_years": 2,
        "user_past_roles": ["Junior Backend Developer"],
    }
    
    career = {
        "career_id": "backend_senior_001",
        "career_title": "Senior Backend Engineer",
        "required_skills": ["java", "spring boot", "microservices", "kubernetes"],
        "required_education": "bachelor's",
        "required_experience_years": 5,
        "centrality": 0.80,
    }
    
    model_weights = {
        "skill_similarity": 0.65,
        "interest_alignment": 0.85,
        "education_match": 0.95,
        "experience_fit": 0.55,
        "career_graph_proximity": 0.60,
        "career_centrality": 0.80,
    }
    
    result = explainer.explain(user_profile, career, model_weights)
    
    # Format for API response
    api_response = {
        "recommendation_id": "rec_12345",
        "career": {
            "id": career["career_id"],
            "title": career["career_title"],
            "match_score": 0.75,
        },
        "explanation": {
            "why_recommended": result.why_recommended,
            "learning_plan": [
                {
                    "skill": item["skill"],
                    "priority": item["rank"],
                    "impact": item["impact_score"],
                    "time_estimate": item["estimated_learning_time"],
                    "reasoning": item["reason"],
                    "resources": item["resources"]
                }
                for item in result.learning_plan
            ],
        },
        "metadata": {
            "generated_at": result.audit_log["timestamp"],
            "model_version": "v1.0.0",
            "confidence": 0.82,
        }
    }
    
    print("\n📤 API Response JSON:")
    print(json.dumps(api_response, indent=2))
    
    return api_response


if __name__ == "__main__":
    # Run all examples
    print("\n🚀 CareerLens Explanation System - Examples\n")
    
    # Example 1: Basic usage
    result1 = example_1_standalone_explainer()
    
    # Example 2: Integrated pipeline (requires trained model)
    result2 = example_2_integrated_with_recommender()
    
    # Example 3: Custom domain
    result3 = example_3_custom_heuristics()
    
    # Example 4: API format
    result4 = example_4_api_response_format()
    
    print("\n\n✅ All examples completed!")
    print("\nNext steps:")
    print("  1. Integrate into FastAPI backend: backend/app/services/recommendations.py")
    print("  2. Update frontend to display explanations: frontend/src/pages/RecommendationsPage.tsx")
    print("  3. Run with real data: python -m ml.models.recommender infer --user_json path/to/user.json")
