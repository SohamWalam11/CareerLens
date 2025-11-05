# RAG System for Career Recommendation Explanations

## System Architecture

This RAG (Retrieval-Augmented Generation) layer provides concise, contextual answers to career-related queries by combining:
1. **Explanation dictionaries** (from `explanations.py`)
2. **Curated knowledge base** (role descriptions, skill stacks, transition paths)
3. **EDA insights** (dataset statistics, career trends, skill frequencies)

---

## System Prompt

```
You are CareerLens Assistant, a career guidance expert providing evidence-based recommendations.

ROLE:
- Answer career questions using only the provided context (explanations, knowledge base, insights)
- Keep responses under 180 words with bullet points for clarity
- Be specific, actionable, and supportive

RESPONSE FORMAT:
- Start with direct answer to the question
- Use bullet points for key details
- Include 1-2 specific action items
- Cite data sources when available (e.g., "Based on analysis of 500+ transitions...")

TONE:
- Professional yet approachable
- Confident but not prescriptive
- Encouraging without being generic

CONSTRAINTS:
- Never invent data or make up statistics
- Don't provide financial, legal, or medical advice
- Don't guarantee job outcomes or salary ranges
- Redirect out-of-scope questions to appropriate resources

DEFLECTION RULES:
- Salary questions → "Salary varies by location, company, and experience. Check Glassdoor or Levels.fyi for regional data."
- Specific company hiring → "For company-specific roles, visit their careers page or LinkedIn."
- Resume reviews → "Consider professional resume services or career coaches for personalized feedback."
- Legal/visa questions → "Immigration and work authorization vary by country. Consult an immigration attorney."
- Mental health/burnout → "Career transitions can be stressful. Consider speaking with a counselor or career coach."

When uncertain, say: "Based on available data, [answer]. For personalized guidance, consider scheduling a career consultation."
```

---

## Retrieval Format

### 1. Explanation Dictionary Schema

```json
{
  "user_id": "user_123",
  "career_id": "career_ds_001",
  "career_title": "Data Scientist",
  "timestamp": "2025-11-05T14:30:00Z",
  "explanation": {
    "reasons": [
      "Skill overlap: matched 3 of 5 core skills (python, sql, statistics).",
      "Profile similarity score: 0.78 (cosine).",
      "Model weight for skill alignment scored 0.80."
    ],
    "gaps": [
      {
        "skill": "Machine Learning",
        "reason": "Machine Learning is required for the role and is not yet in the profile.",
        "suggested_action": "Schedule focused practice within the next quarter."
      }
    ],
    "confidence": 0.85
  },
  "user_profile": {
    "current_role": "Data Analyst",
    "skills": ["python", "sql", "statistics", "pandas"],
    "experience_years": 3,
    "education": "bachelor's"
  }
}
```

**Retrieval Query**: 
```
Query: "Why was Data Scientist recommended for me?"
Retrieval: Filter by user_id + career_title match
Context Fields: explanation.reasons, explanation.confidence, user_profile.current_role
```

---

### 2. Knowledge Base Schema

**File**: `backend/knowledge_base/roles/{role_slug}.md`

```markdown
# Data Scientist

## Overview
Extracts insights from data using statistical analysis, machine learning, and visualization. Bridges business strategy and technical implementation.

## Core Responsibilities
- Design and deploy predictive models
- Communicate insights to stakeholders
- Collaborate with engineering on production systems
- Validate data quality and assumptions

## Required Skills
- **Technical**: Python, SQL, Statistics, Machine Learning, Data Visualization
- **Tools**: pandas, scikit-learn, TensorFlow/PyTorch, Tableau/PowerBI
- **Soft**: Communication, Problem-solving, Business acumen

## Typical Background
- Bachelor's or Master's in Computer Science, Statistics, Math, or related field
- 2-5 years experience with data analysis or software engineering
- Portfolio of data projects demonstrating end-to-end workflows

## Career Progression
- Junior Data Scientist → Data Scientist → Senior Data Scientist → Lead/Principal → Director of Data Science
- Alternative paths: ML Engineer, Data Engineering, Analytics Engineering

## Transition Paths

### From Data Analyst (12-month plan)
**Month 1-3: Foundations**
- Complete Stanford CS229 Machine Learning course
- Build 2-3 classification/regression projects (Kaggle datasets)
- Practice: scikit-learn, model evaluation metrics

**Month 4-6: Deep Learning**
- fast.ai Practical Deep Learning course
- Implement neural networks (PyTorch or TensorFlow)
- Project: Image classification or NLP task

**Month 7-9: Production Skills**
- Learn model deployment (Flask API, Docker)
- Study MLOps basics (versioning, monitoring)
- Project: Deploy ML model to cloud (AWS/GCP)

**Month 10-12: Job Prep**
- Build portfolio website with 5+ projects
- Network: attend meetups, contribute to open source
- Apply: target junior/mid-level DS roles

**Key Skills to Acquire**: Machine Learning, Deep Learning, Model Deployment
**Expected Time**: 15-20 hours/week for 12 months

### From Software Engineer (9-month plan)
**Month 1-3: Statistics & ML Theory**
- Statistics fundamentals (distributions, hypothesis testing)
- ML algorithms (linear models, trees, ensembles)
- Resource: "Introduction to Statistical Learning" book

**Month 4-6: Data Manipulation**
- pandas for data wrangling
- SQL for complex queries (joins, window functions)
- Visualization: matplotlib, seaborn, Plotly

**Month 7-9: Portfolio & Domain**
- 3-4 end-to-end projects (problem → model → insights)
- Pick domain focus (finance, healthcare, marketing)
- Prepare for case interviews

**Key Skills to Acquire**: Statistics, pandas, Data Storytelling
**Expected Time**: 10-15 hours/week for 9 months
```

**Retrieval Query**:
```
Query: "How do I transition from Data Analyst to Data Scientist in 12 months?"
Retrieval: Load roles/data-scientist.md → extract "Transition Paths > From Data Analyst"
Context Fields: Monthly milestones, skills to acquire, time commitment
```

---

### 3. EDA Insights Schema

**File**: `backend/knowledge_base/insights/dataset_insights.json`

```json
{
  "dataset_version": "2025-11-01",
  "total_records": 9000,
  "insights": [
    {
      "category": "skill_frequency",
      "data": {
        "python": {"count": 7200, "percentage": 80.0, "roles": ["Data Scientist", "ML Engineer", "Backend Engineer"]},
        "machine_learning": {"count": 4500, "percentage": 50.0, "roles": ["Data Scientist", "ML Engineer"]},
        "sql": {"count": 6300, "percentage": 70.0, "roles": ["Data Analyst", "Data Scientist", "Backend Engineer"]}
      }
    },
    {
      "category": "transition_success_rate",
      "data": {
        "Data Analyst → Data Scientist": {
          "observed_transitions": 523,
          "avg_time_months": 14,
          "success_rate": 0.72,
          "common_skills_added": ["machine learning", "deep learning", "tensorflow"]
        },
        "Software Engineer → Data Scientist": {
          "observed_transitions": 312,
          "avg_time_months": 11,
          "success_rate": 0.68,
          "common_skills_added": ["statistics", "pandas", "data visualization"]
        }
      }
    },
    {
      "category": "role_demand",
      "data": {
        "Data Scientist": {"centrality": 0.85, "growth_trend": "high", "avg_experience_years": 3.5},
        "ML Engineer": {"centrality": 0.82, "growth_trend": "very_high", "avg_experience_years": 4.2}
      }
    },
    {
      "category": "skill_transferability",
      "data": {
        "sql → spark": 0.65,
        "python → r": 0.70,
        "statistics → machine_learning": 0.60,
        "machine_learning → deep_learning": 0.75
      }
    }
  ],
  "metadata": {
    "last_updated": "2025-11-01",
    "source": "CareerLens EDA pipeline",
    "coverage": "Global tech roles, 2020-2025"
  }
}
```

**Retrieval Query**:
```
Query: "How common is the transition from Data Analyst to Data Scientist?"
Retrieval: insights.transition_success_rate["Data Analyst → Data Scientist"]
Context Fields: observed_transitions, avg_time_months, success_rate, common_skills_added
```

---

## Query Handlers

### Query Type 1: "Why was <role> recommended for me?"

**Retrieval Strategy**:
1. Fetch explanation dict by `user_id` + `career_title`
2. Load role knowledge base: `roles/{role_slug}.md` (Overview, Required Skills sections)
3. Load EDA insights: skill_frequency for matched skills

**Context Template**:
```
EXPLANATION:
{explanation.reasons}
Confidence: {explanation.confidence}

ROLE OVERVIEW:
{knowledge_base.overview}
Required Skills: {knowledge_base.required_skills}

DATA INSIGHTS:
- {skill_name} appears in {percentage}% of {role} roles (based on {count} profiles)
- Your current skills match {matched_count}/{total_required} core competencies
```

**Response Example**:
```
Data Scientist was recommended because:

• **Strong skill alignment**: You match 3 of 5 core skills (Python, SQL, Statistics)
• **Natural progression**: 523 successful Data Analyst → Data Scientist transitions observed in our dataset (avg 14 months)
• **High confidence**: Model scored this match at 0.85 based on your 3 years of experience

Key gaps to address:
• Machine Learning (required by 88% of Data Scientist roles)
• Deep Learning (fast-growing demand, seen in 65% of postings)

Next steps:
• Complete Stanford CS229 course (3 months)
• Build 2-3 ML projects for portfolio
```

---

### Query Type 2: "How do I transition from <current> to <target> in 12 months?"

**Retrieval Strategy**:
1. Load role knowledge base: `roles/{target_role}.md` → extract "Transition Paths > From {current_role}"
2. Load EDA insights: transition_success_rate for `{current} → {target}`
3. Fetch user's current skills from profile → compute skill gaps

**Context Template**:
```
TRANSITION PLAN:
{knowledge_base.transition_paths[current_to_target]}

DATA INSIGHTS:
- {observed_transitions} people successfully made this transition
- Average time: {avg_time_months} months
- Success rate: {success_rate}
- Common skills added: {common_skills_added}

YOUR SKILL GAPS:
{gaps from explanation dict}
```

**Response Example**:
```
Data Analyst → Data Scientist in 12 months (based on 523 observed transitions, 72% success rate):

**Months 1-3**: Foundations
• Stanford CS229 Machine Learning
• Build 2-3 regression/classification projects
• Master scikit-learn basics

**Months 4-6**: Advanced techniques
• fast.ai Deep Learning course
• PyTorch or TensorFlow implementation
• Kaggle competitions for practice

**Months 7-9**: Production skills
• Model deployment (Docker, Flask API)
• MLOps fundamentals
• Cloud deployment (AWS/GCP)

**Months 10-12**: Job prep
• Portfolio site with 5+ projects
• Networking (meetups, open source)
• Target junior/mid-level roles

Time commitment: 15-20 hours/week
```

---

## Safety & Deflection Rules

### Out-of-Scope Categories

| Question Type | Detection Keywords | Deflection Response |
|--------------|-------------------|---------------------|
| **Salary/Compensation** | salary, pay, compensation, earnings, income | "Salary varies by location, company, and experience. Check Glassdoor, Levels.fyi, or Payscale for regional data." |
| **Company-Specific Hiring** | hiring at, interview at, apply to, <company_name> | "For company-specific roles and hiring processes, visit their careers page or reach out to recruiters on LinkedIn." |
| **Resume/CV Review** | review my resume, feedback on CV, resume tips | "For personalized resume feedback, consider professional services like TopResume or work with a career coach." |
| **Legal/Immigration** | visa, work permit, h1b, sponsorship, immigration | "Immigration and work authorization vary by country and company. Consult an immigration attorney for legal advice." |
| **Mental Health** | burnout, stress, anxiety, depression, overwhelmed | "Career transitions can be challenging. Consider speaking with a counselor, therapist, or career coach for support." |
| **Financial Advice** | invest, loans, student debt, savings | "For financial planning, consult a certified financial advisor. I can help with career skill development instead." |
| **Job Guarantees** | guarantee job, promise hire, ensure placement | "I provide career guidance, but can't guarantee outcomes. Success depends on market conditions, effort, and timing." |
| **Age/Discrimination** | too old, age limit, ageism | "Age isn't a barrier to career change. Focus on demonstrating relevant skills and passion. Many successful transitions happen at all ages." |
| **Non-Tech Careers** | lawyer, doctor, teacher, chef (non-tech roles) | "My expertise is in tech career paths. For other industries, seek domain-specific career resources or advisors." |
| **Academic Admissions** | MBA, PhD, grad school, university admissions | "For graduate school guidance, contact university admissions offices or academic advisors. I focus on professional skill development." |

### Detection Logic (Regex Patterns)

```python
OUT_OF_SCOPE_PATTERNS = {
    "salary": r"\b(salary|salaries|pay|compensation|earnings?|income)\b",
    "company_hiring": r"\b(hiring at|interview at|apply to|working at)\b",
    "resume": r"\b(resume|cv|curriculum vitae)\s+(review|feedback|critique)\b",
    "legal": r"\b(visa|h1b|work permit|sponsorship|immigration|green card)\b",
    "mental_health": r"\b(burnout|stress|anxiety|depression|overwhelmed|mental health)\b",
    "financial": r"\b(invest|loan|debt|savings|finance|budget)\b",
    "guarantee": r"\b(guarantee|promise|ensure|certain)\s+(job|hire|placement)\b",
}
```

### Response Templates

**Generic Deflection**:
```
I specialize in career skill development and transition planning. Your question about [topic] is outside my scope.

For [topic], I recommend:
• [Specific resource or action]

How can I help with your career skill development instead?
```

**Uncertainty Handling**:
```
Based on available data, [partial answer if any].

For more personalized guidance on [specific aspect], consider:
• Scheduling a 1-on-1 career consultation
• Joining our community forum for peer insights
• Reviewing [specific resource]
```

---

## Retrieval Pipeline

### Step 1: Query Classification

```python
def classify_query(query: str) -> str:
    """Classify user query into supported types."""
    query_lower = query.lower()
    
    if "why" in query_lower and "recommend" in query_lower:
        return "why_recommended"
    
    if "transition" in query_lower or "switch" in query_lower or "move from" in query_lower:
        return "transition_plan"
    
    if any(pattern in query_lower for pattern in ["skill gap", "what to learn", "need to learn"]):
        return "skill_gaps"
    
    if any(pattern in query_lower for pattern in ["common", "typical", "average time"]):
        return "data_insights"
    
    return "general"
```

### Step 2: Context Retrieval

```python
def retrieve_context(query_type: str, user_id: str, params: dict) -> dict:
    """Retrieve relevant context based on query type."""
    
    context = {}
    
    if query_type == "why_recommended":
        # Fetch explanation dict
        context["explanation"] = get_explanation(user_id, params["career_title"])
        # Load role knowledge base
        context["role_kb"] = load_role_kb(params["career_title"])
        # Get skill frequency insights
        context["insights"] = get_skill_insights(params["matched_skills"])
    
    elif query_type == "transition_plan":
        # Load transition path from knowledge base
        context["transition"] = load_transition_path(params["source_role"], params["target_role"])
        # Get transition statistics
        context["stats"] = get_transition_stats(params["source_role"], params["target_role"])
        # Compute user's skill gaps
        context["gaps"] = compute_skill_gaps(user_id, params["target_role"])
    
    return context
```

### Step 3: Response Generation

```python
def generate_response(query_type: str, context: dict, max_words: int = 180) -> str:
    """Generate concise, bullet-point response from context."""
    
    # Check for out-of-scope patterns
    deflection = check_deflection_rules(query)
    if deflection:
        return deflection
    
    # Build response based on query type
    if query_type == "why_recommended":
        return format_why_recommended(context)
    
    elif query_type == "transition_plan":
        return format_transition_plan(context)
    
    # Enforce word limit
    return truncate_response(response, max_words)
```

---

## Example Conversations

### Example 1: Why Recommended

**User**: "Why was Machine Learning Engineer recommended for me?"

**Retrieval**:
- Explanation dict: reasons, confidence
- Role KB: `roles/ml-engineer.md` (overview, skills)
- Insights: skill_frequency for matched skills

**Response**:
```
ML Engineer was recommended based on:

• **Strong technical foundation**: You match 4 of 6 core skills (Python, TensorFlow, Statistics, SQL)
• **Natural progression**: Your 3 years as Data Scientist provide 80% of required competencies
• **High demand**: ML Engineer has 0.82 career centrality (top 15% of tech roles)

Your profile scored 0.87 confidence due to:
• Deep Learning expertise (appears in 85% of ML Engineer roles)
• Production engineering experience
• Strong model deployment background

To strengthen your candidacy:
• Add Kubernetes (65% of roles require)
• Gain MLOps experience (monitoring, versioning)

Based on 312 successful Data Scientist → ML Engineer transitions (avg 9 months).
```

---

### Example 2: Transition Plan

**User**: "How do I transition from Software Engineer to Data Scientist in 12 months?"

**Retrieval**:
- Role KB: `roles/data-scientist.md` → "From Software Engineer" section
- Insights: transition_success_rate, common_skills_added
- User profile: current skills, gaps

**Response**:
```
Software Engineer → Data Scientist in 12 months (312 successful transitions observed, 68% success rate):

**Months 1-3**: Statistics & ML foundations
• "Intro to Statistical Learning" book
• ML algorithms: regression, trees, ensembles
• Practice: scikit-learn on toy datasets

**Months 4-6**: Data manipulation
• pandas for wrangling
• SQL advanced queries
• Visualization: matplotlib, seaborn

**Months 7-9**: Portfolio building
• 3-4 end-to-end projects (Kaggle datasets)
• Domain focus (finance, healthcare)
• Practice case interviews

**Months 10-12**: Job prep
• Portfolio site with projects
• Network at data meetups
• Target junior/mid roles

Time: 10-15 hrs/week
Key skills: Statistics, pandas, Data Storytelling
```

---

### Example 3: Out-of-Scope (Salary)

**User**: "What's the average salary for a Data Scientist?"

**Detection**: Keyword "salary" matched

**Response**:
```
Salary varies significantly by location, company, experience, and specialization.

For accurate compensation data, I recommend:
• Glassdoor (company-specific ranges)
• Levels.fyi (tech compensation with equity)
• Payscale (regional breakdowns)

I can help with skill development to maximize your career potential. What skills are you looking to build?
```

---

### Example 4: Out-of-Scope (Company Hiring)

**User**: "Is Google hiring Data Scientists right now?"

**Detection**: Keyword "hiring at" matched

**Response**:
```
For company-specific hiring and open roles:
• Visit Google Careers page directly
• Check LinkedIn job postings
• Reach out to Google recruiters on LinkedIn

I can help you prepare by:
• Identifying skills Google DS roles typically require
• Building projects that demonstrate relevant expertise
• Planning your learning path

Would you like guidance on skill development for top-tier DS roles?
```

---

## Implementation Notes

### File Structure

```
backend/
├── app/
│   └── services/
│       ├── explanations.py          # Existing
│       └── rag_service.py           # New: RAG query handler
├── knowledge_base/
│   ├── roles/
│   │   ├── data-scientist.md
│   │   ├── ml-engineer.md
│   │   ├── data-analyst.md
│   │   └── ...
│   └── insights/
│       └── dataset_insights.json
└── tests/
    ├── test_explanations.py         # Existing
    └── test_rag_service.py          # New: RAG tests
```

### Performance Considerations

1. **Caching**: Cache knowledge base markdown files in memory (reload on update)
2. **Indexing**: Use simple dict lookup for explanation retrieval (fast for <10k users)
3. **Response Time**: Target <500ms for query processing
4. **Fallback**: If context retrieval fails, return generic guidance with apology

### Quality Metrics

1. **Response Length**: 95% under 180 words
2. **Deflection Accuracy**: 90%+ correct out-of-scope detection
3. **User Satisfaction**: Track thumbs up/down on responses
4. **Coverage**: Monitor "I don't know" rate (target <5% for in-scope queries)

---

## Safety Guardrails

### Content Filtering

- **No Hallucination**: Only use retrieved context; if data missing, say so
- **No Guarantees**: Never promise job outcomes or specific timelines
- **No Personal Data**: Don't expose other users' profiles or identifiable info
- **No Bias**: Avoid age, gender, race, or nationality in recommendations

### Audit Trail

Log all queries and responses for review:
```json
{
  "timestamp": "2025-11-05T14:30:00Z",
  "user_id": "user_123",
  "query": "Why was Data Scientist recommended?",
  "query_type": "why_recommended",
  "context_sources": ["explanation_dict", "role_kb", "insights"],
  "response": "...",
  "word_count": 156,
  "deflection": false
}
```

---

## Next Steps

1. **Implement `rag_service.py`**: Query classification, retrieval, response formatting
2. **Create knowledge base**: Author 10-15 role markdown files with transition paths
3. **Generate insights.json**: Export EDA statistics to JSON format
4. **Unit tests**: Cover all query types, deflection rules, edge cases
5. **API endpoint**: Add `/api/v1/ask` POST endpoint
6. **Frontend integration**: Chat interface with conversation history
