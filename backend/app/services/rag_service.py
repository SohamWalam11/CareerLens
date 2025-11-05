"""RAG service for answering career-related questions with context retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """Supported query types."""
    
    WHY_RECOMMENDED = "why_recommended"
    TRANSITION_PLAN = "transition_plan"
    SKILL_GAPS = "skill_gaps"
    DATA_INSIGHTS = "data_insights"
    GENERAL = "general"
    OUT_OF_SCOPE = "out_of_scope"


OUT_OF_SCOPE_PATTERNS = {
    "salary": r"\b(salary|salaries|pay|compensation|earnings?|income)\b",
    "company_hiring": r"\b(is|are)\s+\w+\s+(hiring|interviewing)|hiring\s+(at|with|for)\s+\w+",
    "resume": r"\b(review|critique|feedback)\s+(my\s+)?(resume|cv)\b",
    "legal": r"\b(visa|h1b|work permit|sponsorship|immigration|green card)\b",
    "mental_health": r"\b(burnout|stress|anxiety|depression|overwhelmed|mental health)\b",
    "financial": r"\b(invest|loan|debt|savings|finance|budget)\b",
    "guarantee": r"\b(guarantee|promise|ensure|certain)\s+(job|hire|placement)\b",
}

DEFLECTION_RESPONSES = {
    "salary": "Salary varies by location, company, and experience. Check Glassdoor, Levels.fyi, or Payscale for regional data.\n\nHow can I help with your career skill development instead?",
    "company_hiring": "For company-specific roles and hiring processes, visit their careers page or reach out to recruiters on LinkedIn.\n\nI can help you prepare by identifying required skills. What role interests you?",
    "resume": "For personalized resume feedback, consider professional services like TopResume or work with a career coach.\n\nI can help with skill development strategies. What areas would you like to strengthen?",
    "legal": "Immigration and work authorization vary by country and company. Consult an immigration attorney for legal advice.\n\nI focus on career skill development. What technical skills are you building?",
    "mental_health": "Career transitions can be challenging. Consider speaking with a counselor, therapist, or career coach for support.\n\nI can help with structured learning plans. What career path interests you?",
    "financial": "For financial planning, consult a certified financial advisor.\n\nI can help with career skill development and transition planning. What's your goal?",
    "guarantee": "I provide career guidance, but can't guarantee outcomes. Success depends on market conditions, effort, and timing.\n\nI can help you build competitive skills. What role are you targeting?",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Retrieved context for answering a query."""
    
    explanation: dict[str, Any] | None = None
    role_kb: dict[str, Any] | None = None
    insights: dict[str, Any] | None = None
    transition_plan: str | None = None
    skill_gaps: list[dict[str, str]] | None = None


@dataclass
class RAGResponse:
    """Structured response from RAG service."""
    
    answer: str
    query_type: QueryType
    sources: list[str]
    word_count: int
    deflection: bool = False


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class CareerRAGService:
    """Answer career questions using retrieval-augmented generation."""
    
    SYSTEM_PROMPT = """You are CareerLens Assistant, a career guidance expert providing evidence-based recommendations.

ROLE:
- Answer career questions using only the provided context
- Keep responses under 180 words with bullet points for clarity
- Be specific, actionable, and supportive

RESPONSE FORMAT:
- Start with direct answer to the question
- Use bullet points for key details
- Include 1-2 specific action items
- Cite data sources when available

TONE: Professional yet approachable, confident but not prescriptive, encouraging without being generic.

CONSTRAINTS:
- Never invent data or make up statistics
- Don't provide financial, legal, or medical advice
- Don't guarantee job outcomes
"""
    
    def __init__(
        self,
        kb_path: Path | None = None,
        insights_path: Path | None = None,
    ):
        """Initialize RAG service with knowledge base and insights paths."""
        self.kb_path = kb_path or Path(__file__).parent.parent / "knowledge_base" / "roles"
        self.insights_path = insights_path or Path(__file__).parent.parent / "knowledge_base" / "insights" / "dataset_insights.json"
        self._kb_cache: dict[str, dict] = {}
        self._insights_cache: dict | None = None
    
    def answer_query(
        self,
        query: str,
        user_id: str,
        explanation_dict: dict[str, Any] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Answer a career-related query with context retrieval.
        
        Args:
            query: User's question
            user_id: User identifier for personalization
            explanation_dict: Pre-fetched explanation from explanations.py
            user_profile: User's current skills and background
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Check for out-of-scope queries first
        deflection = self._check_deflection(query)
        if deflection:
            return RAGResponse(
                answer=deflection,
                query_type=QueryType.OUT_OF_SCOPE,
                sources=["deflection_rules"],
                word_count=len(deflection.split()),
                deflection=True,
            )
        
        # Classify query
        query_type = self._classify_query(query)
        
        # Retrieve context
        context = self._retrieve_context(
            query_type=query_type,
            query=query,
            user_id=user_id,
            explanation_dict=explanation_dict,
            user_profile=user_profile,
        )
        
        # Generate response
        answer = self._generate_response(query_type, context, query)
        
        # Determine sources used
        sources = self._get_sources(context)
        
        # Enforce word limit
        answer = self._truncate_response(answer, max_words=180)
        
        return RAGResponse(
            answer=answer,
            query_type=query_type,
            sources=sources,
            word_count=len(answer.split()),
            deflection=False,
        )
    
    # -----------------------------------------------------------------------
    # Query Classification
    # -----------------------------------------------------------------------
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the query type based on keywords and patterns."""
        query_lower = query.lower()
        
        # Check in priority order (most specific first)
        if "why" in query_lower and ("recommend" in query_lower or "suggested" in query_lower):
            return QueryType.WHY_RECOMMENDED
        
        if any(phrase in query_lower for phrase in ["skill gap", "what to learn", "need to learn", "should i learn"]):
            return QueryType.SKILL_GAPS
        
        # DATA_INSIGHTS should come before TRANSITION_PLAN (more specific)
        if any(phrase in query_lower for phrase in ["how common", "typical", "average time", "success rate", "how many"]):
            return QueryType.DATA_INSIGHTS
        
        if "transition" in query_lower or "switch" in query_lower or "move from" in query_lower:
            return QueryType.TRANSITION_PLAN
        
        return QueryType.GENERAL
    
    def _check_deflection(self, query: str) -> str | None:
        """Check if query is out of scope and return deflection response."""
        query_lower = query.lower()
        
        for category, pattern in OUT_OF_SCOPE_PATTERNS.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return DEFLECTION_RESPONSES.get(
                    category,
                    "I specialize in career skill development and transition planning. For personalized guidance, consider scheduling a career consultation.",
                )
        
        return None
    
    # -----------------------------------------------------------------------
    # Context Retrieval
    # -----------------------------------------------------------------------
    
    def _retrieve_context(
        self,
        query_type: QueryType,
        query: str,
        user_id: str,
        explanation_dict: dict | None,
        user_profile: dict | None,
    ) -> QueryContext:
        """Retrieve relevant context based on query type."""
        context = QueryContext()
        
        if query_type == QueryType.WHY_RECOMMENDED:
            context.explanation = explanation_dict
            
            # Extract career title from query or explanation
            career_title = self._extract_career_title(query, explanation_dict)
            if career_title:
                context.role_kb = self._load_role_kb(career_title)
                context.insights = self._load_insights()
        
        elif query_type == QueryType.TRANSITION_PLAN:
            # Extract source and target roles
            source_role, target_role = self._extract_transition_roles(query, user_profile)
            
            if target_role:
                role_kb = self._load_role_kb(target_role)
                if role_kb and "transition_paths" in role_kb:
                    # Find matching transition path
                    for path_key, path_content in role_kb["transition_paths"].items():
                        if source_role and source_role.lower() in path_key.lower():
                            context.transition_plan = path_content
                            break
                    
                    # Fallback to first available path if no match
                    if not context.transition_plan and role_kb["transition_paths"]:
                        context.transition_plan = list(role_kb["transition_paths"].values())[0]
            
            context.insights = self._load_insights()
            context.skill_gaps = explanation_dict.get("gaps") if explanation_dict else None
        
        elif query_type == QueryType.SKILL_GAPS:
            context.explanation = explanation_dict
            context.skill_gaps = explanation_dict.get("gaps") if explanation_dict else None
        
        elif query_type == QueryType.DATA_INSIGHTS:
            context.insights = self._load_insights()
        
        return context
    
    def _load_role_kb(self, career_title: str) -> dict[str, Any] | None:
        """Load role knowledge base from markdown file."""
        # Normalize title to slug
        slug = career_title.lower().replace(" ", "-").replace("/", "-")
        
        # Check cache
        if slug in self._kb_cache:
            return self._kb_cache[slug]
        
        kb_file = self.kb_path / f"{slug}.md"
        if not kb_file.exists():
            return None
        
        # Parse markdown into structured dict
        content = kb_file.read_text(encoding="utf-8")
        kb_data = self._parse_role_markdown(content)
        
        # Cache result
        self._kb_cache[slug] = kb_data
        return kb_data
    
    def _load_insights(self) -> dict[str, Any] | None:
        """Load EDA insights from JSON file."""
        if self._insights_cache:
            return self._insights_cache
        
        if not self.insights_path.exists():
            return None
        
        self._insights_cache = json.loads(self.insights_path.read_text(encoding="utf-8"))
        return self._insights_cache
    
    def _parse_role_markdown(self, content: str) -> dict[str, Any]:
        """Parse role markdown into structured dictionary."""
        # Simple parser - extract sections by headers
        data: dict[str, Any] = {}
        current_section = None
        current_content: list[str] = []
        
        for line in content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_section:
                    data[current_section] = "\n".join(current_content).strip()
                
                # Start new section
                current_section = line[3:].strip().lower().replace(" ", "_")
                current_content = []
            elif line.startswith("### "):
                # Subsection for transition paths
                if "transition" in (current_section or ""):
                    subsection_name = line[4:].strip()
                    if "transition_paths" not in data:
                        data["transition_paths"] = {}
                    current_section = f"transition_path_{len(data['transition_paths'])}"
                    data["transition_paths"][subsection_name] = []
                    current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            if current_section.startswith("transition_path_"):
                # Append to last transition path
                last_key = list(data["transition_paths"].keys())[-1]
                data["transition_paths"][last_key] = "\n".join(current_content).strip()
            else:
                data[current_section] = "\n".join(current_content).strip()
        
        return data
    
    # -----------------------------------------------------------------------
    # Response Generation
    # -----------------------------------------------------------------------
    
    def _generate_response(
        self,
        query_type: QueryType,
        context: QueryContext,
        query: str,
    ) -> str:
        """Generate response based on query type and context."""
        if query_type == QueryType.WHY_RECOMMENDED:
            return self._format_why_recommended(context)
        
        elif query_type == QueryType.TRANSITION_PLAN:
            return self._format_transition_plan(context, query)
        
        elif query_type == QueryType.SKILL_GAPS:
            return self._format_skill_gaps(context)
        
        elif query_type == QueryType.DATA_INSIGHTS:
            return self._format_data_insights(context, query)
        
        else:
            return "I can help with career recommendations, transition planning, and skill development. Try asking:\n• Why was [role] recommended?\n• How do I transition from [current] to [target]?\n• What skills should I learn?"
    
    def _format_why_recommended(self, context: QueryContext) -> str:
        """Format 'why recommended' response."""
        if not context.explanation:
            return "I don't have enough information about this recommendation. Please provide more details about your profile and the recommended role."
        
        parts: list[str] = []
        explanation = context.explanation
        
        # Opening statement
        career_title = explanation.get("career_title", "this role")
        confidence = explanation.get("confidence", 0.0)
        parts.append(f"{career_title} was recommended based on:")
        parts.append("")
        
        # Add reasons as bullets
        reasons = explanation.get("reasons", [])
        for reason in reasons[:3]:  # Top 3 reasons
            parts.append(f"• {reason}")
        
        parts.append("")
        parts.append(f"**Confidence**: {confidence:.2f}")
        
        # Add role context if available
        if context.role_kb and "overview" in context.role_kb:
            overview = context.role_kb["overview"][:150]  # Truncate
            parts.append("")
            parts.append(f"**About the role**: {overview}...")
        
        # Add skill gaps
        gaps = explanation.get("gaps", [])
        if gaps:
            parts.append("")
            parts.append("**Key gaps to address**:")
            for gap in gaps[:2]:  # Top 2 gaps
                parts.append(f"• {gap['skill']}")
        
        return "\n".join(parts)
    
    def _format_transition_plan(self, context: QueryContext, query: str) -> str:
        """Format transition plan response."""
        if not context.transition_plan:
            return "I don't have a specific transition plan for this path. Consider:\n• Identifying skill gaps between roles\n• Building 3-5 portfolio projects\n• Networking in your target field\n• Seeking mentorship from professionals in the role"
        
        parts: list[str] = []
        
        # Extract transition details from query
        source_role, target_role = self._extract_transition_roles(query, None)
        
        # Add header with stats if available
        if context.insights and "insights" in context.insights:
            for insight in context.insights["insights"]:
                if insight.get("category") == "transition_success_rate":
                    transition_key = f"{source_role} → {target_role}"
                    if transition_key in insight.get("data", {}):
                        stats = insight["data"][transition_key]
                        parts.append(f"{transition_key} ({stats.get('observed_transitions', 0)} transitions observed, {int(stats.get('success_rate', 0) * 100)}% success rate):")
                        parts.append("")
        
        # Add transition plan (truncate if too long)
        plan_lines = context.transition_plan.split("\n")
        for line in plan_lines[:15]:  # Limit lines
            if line.strip():
                parts.append(line)
        
        return "\n".join(parts)
    
    def _format_skill_gaps(self, context: QueryContext) -> str:
        """Format skill gaps response."""
        gaps = context.skill_gaps or (context.explanation or {}).get("gaps", [])
        
        if not gaps:
            return "No significant skill gaps identified. Focus on deepening existing expertise and building portfolio projects."
        
        parts: list[str] = ["Skills to develop:", ""]
        
        for idx, gap in enumerate(gaps[:5], 1):
            skill = gap["skill"]
            action = gap.get("suggested_action", "Practice within the next quarter")
            parts.append(f"{idx}. **{skill}**")
            parts.append(f"   {action}")
            parts.append("")
        
        return "\n".join(parts)
    
    def _format_data_insights(self, context: QueryContext, query: str) -> str:
        """Format data insights response."""
        if not context.insights:
            return "I don't have statistical insights available. Please try a different question."
        
        # Try to match query to insight category
        query_lower = query.lower()
        
        insights = context.insights.get("insights", [])
        for insight in insights:
            category = insight.get("category", "")
            
            if "transition" in query_lower and category == "transition_success_rate":
                # Find matching transition
                data = insight.get("data", {})
                if data:
                    # Return first available transition stat
                    key, stats = next(iter(data.items()))
                    return f"**{key}**:\n• {stats.get('observed_transitions', 0)} transitions observed\n• Average time: {stats.get('avg_time_months', 0)} months\n• Success rate: {int(stats.get('success_rate', 0) * 100)}%\n• Common skills added: {', '.join(stats.get('common_skills_added', [])[:3])}"
        
        return "Based on available data, I can provide insights on transition success rates, skill frequencies, and role demand. What specific metric interests you?"
    
    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    
    def _extract_career_title(self, query: str, explanation: dict | None) -> str | None:
        """Extract career title from query or explanation dict."""
        if explanation and "career_title" in explanation:
            return explanation["career_title"]
        
        # Try to extract from query (simple heuristic)
        # Look for pattern: "why was [role] recommended"
        match = re.search(r"why was (.+?) recommended", query.lower())
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_transition_roles(
        self,
        query: str,
        user_profile: dict | None,
    ) -> tuple[str | None, str | None]:
        """Extract source and target roles from query."""
        query_lower = query.lower()
        
        # Pattern: "from X to Y" - capture until " to " and from " to " until end/punctuation
        match = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\s+in\b|\?|$)", query_lower)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        # Pattern: "transition to Y" with profile context
        match = re.search(r"(?:transition|switch|move) to (.+?)(?:\s|$|in|\?)", query_lower)
        if match:
            target = match.group(1).strip()
            source = user_profile.get("current_role") if user_profile else None
            return source, target
        
        return None, None
    
    def _get_sources(self, context: QueryContext) -> list[str]:
        """Determine which sources were used for the response."""
        sources: list[str] = []
        
        if context.explanation:
            sources.append("explanation_dict")
        if context.role_kb:
            sources.append("role_knowledge_base")
        if context.insights:
            sources.append("eda_insights")
        if context.transition_plan:
            sources.append("transition_plan")
        
        return sources or ["general_guidance"]
    
    def _truncate_response(self, response: str, max_words: int = 180) -> str:
        """Truncate response to max words while preserving structure."""
        words = response.split()
        if len(words) <= max_words:
            return response
        
        # Truncate and add ellipsis
        truncated = " ".join(words[:max_words])
        # Try to end at a complete sentence
        last_period = truncated.rfind(".")
        if last_period > max_words * 0.7:  # If close to end
            truncated = truncated[:last_period + 1]
        else:
            truncated += "..."
        
        return truncated


__all__ = ["CareerRAGService", "RAGResponse", "QueryType"]
