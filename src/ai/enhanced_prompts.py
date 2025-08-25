"""
Enhanced prompt templates and context management for improved LLM responses.
This module provides sophisticated prompt engineering for better meeting insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


class EnhancedPromptManager:
    """Manages enhanced prompts for detailed, context-aware responses."""
    
    # Maximum context tokens to utilize (leaving room for response)
    MAX_CONTEXT_TOKENS = 90000  # ~90k tokens for context, 10k for response
    SUMMARY_CONTEXT_TOKENS = 95000  # More context for summaries
    
    def __init__(self):
        """Initialize enhanced prompt manager."""
        self.universal_template = """You are a helpful AI assistant analyzing meeting documents.

USER QUERY: "{query}"

{context_instructions}

MEETING CONTEXT ({document_count} documents{date_range}):
{context}

{response_instructions}

{format_instructions}

CRITICAL CITATION REQUIREMENTS:
- When referencing any information, you MUST cite the specific document filename
- Format: "According to [filename]: [information]" 
- Example: "According to Meeting_2024_01_15.docx: The budget was approved"
- NEVER use generic references like "Document: Chunk X", "the document", or chunk numbers
- Every piece of information must be traceable to its source meeting filename

CITATION FORMAT EXAMPLES:
- "According to Print_Migration_Meeting.docx: Adrian expressed concerns about the timeline"
- "From Daily_Standup_2024_01_15.docx: The team decided to postpone the release"
- "As mentioned in Project_Review.docx: Sarah proposed a new budget allocation"
- "During the meeting recorded in Team_Sync_2024_02_10.docx: The following decisions were made" """
        





    def optimize_context_for_token_limit(
        self, 
        context_chunks: List[Dict[str, Any]], 
        query: str,
        max_tokens: int = None
    ) -> Tuple[str, List[str]]:
        """
        Optimize context to fit within token limits while maintaining quality.
        
        Args:
            context_chunks: List of context chunks with metadata
            query: User query for relevance scoring
            max_tokens: Maximum tokens to use (defaults to class constants)
            
        Returns:
            Tuple of (optimized_context, included_documents)
        """
        if max_tokens is None:
            # Detect summary queries for higher token allocation
            is_summary = self._detect_summary_query(query)
            max_tokens = self.SUMMARY_CONTEXT_TOKENS if is_summary else self.MAX_CONTEXT_TOKENS
        
        # Sort chunks by relevance and recency
        scored_chunks = self._score_chunks_for_relevance(context_chunks, query)
        
        # Build optimized context grouped by document within token limit
        document_chunks = {}
        included_documents = set()
        current_tokens = 0
        
        for chunk_data in scored_chunks:
            chunk_text = chunk_data['text']
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_tokens:
                # If this is critical content, try to summarize instead of dropping
                if chunk_data['score'] > 0.8:  # High relevance threshold
                    summarized = self._summarize_chunk_content(chunk_text, max_tokens - current_tokens)
                    if summarized:
                        doc_name = chunk_data.get('document_name', 'Unknown_Document')
                        if doc_name not in document_chunks:
                            document_chunks[doc_name] = []
                        document_chunks[doc_name].append(summarized)
                        current_tokens += self._estimate_tokens(summarized)
                        included_documents.add(doc_name)
                break
            
            doc_name = chunk_data.get('document_name', 'Unknown_Document')
            if doc_name not in document_chunks:
                document_chunks[doc_name] = []
            document_chunks[doc_name].append(chunk_text)
            current_tokens += chunk_tokens
            included_documents.add(doc_name)
        
        # Format context with clear document source headers
        final_context = self._format_context_with_sources(document_chunks)
        
        total_chunks = sum(len(chunks) for chunks in document_chunks.values())
        logger.info(f"Context optimization: {len(context_chunks)} chunks -> {total_chunks} chunks grouped by document")
        logger.info(f"Estimated tokens: {current_tokens}/{max_tokens}")
        logger.info(f"Documents included: {len(included_documents)}")
        
        return final_context, list(included_documents)
    
    def _score_chunks_for_relevance(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Score and sort chunks by relevance to query."""
        scored_chunks = []
        
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            score = 0.0
            
            # Keyword matching
            text_words = set(re.findall(r'\b\w+\b', text))
            keyword_overlap = len(query_keywords.intersection(text_words))
            score += keyword_overlap / max(len(query_keywords), 1) * 0.4
            
            # Semantic importance (based on content indicators)
            importance_indicators = ['decision', 'action', 'next steps', 'agreed', 'concluded', 'summary']
            for indicator in importance_indicators:
                if indicator in text:
                    score += 0.1
            
            # Recency bonus (if timestamp available)
            if 'timestamp' in chunk:
                try:
                    chunk_date = datetime.fromisoformat(chunk['timestamp'])
                    days_old = (datetime.now() - chunk_date).days
                    recency_score = max(0, 1 - days_old / 365) * 0.2  # Linear decay over a year
                    score += recency_score
                except:
                    pass
            
            # Length penalty for very short chunks
            if len(text) < 100:
                score *= 0.7
            
            scored_chunks.append({
                'score': score,
                'text': chunk.get('text', ''),
                'document_name': chunk.get('document_name', 'Unknown'),
                'timestamp': chunk.get('timestamp')
            })
        
        # Sort by score (descending)
        return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _summarize_chunk_content(self, text: str, max_tokens: int) -> Optional[str]:
        """Summarize chunk content to fit within token limit."""
        if self._estimate_tokens(text) <= max_tokens:
            return text
        
        # Simple truncation with intelligent cut-off at sentence boundaries
        target_chars = max_tokens * 4
        if len(text) <= target_chars:
            return text
        
        # Find last complete sentence within limit
        truncated = text[:target_chars]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > target_chars * 0.7:  # If we can keep most content
            return truncated[:last_sentence_end + 1] + " [Content summarized for context optimization]"
        else:
            return truncated + "... [Content truncated for context optimization]"
    
    def _detect_summary_query(self, query: str) -> bool:
        """Detect if query is asking for a summary/comprehensive overview."""
        summary_indicators = [
            'summary', 'summarize', 'overview', 'comprehensive', 'all meetings',
            'across all', 'complete picture', 'full scope', 'everything',
            'all documents', 'overall', 'total', 'entire', 'whole'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in summary_indicators)
    
    def generate_enhanced_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an enhanced prompt using the universal template.
        
        Args:
            query: User query
            context_chunks: Context chunks with metadata
            additional_metadata: Additional metadata for template variables
            
        Returns:
            Enhanced prompt string
        """
        # Optimize context for token limits
        optimized_context, included_documents = self.optimize_context_for_token_limit(
            context_chunks, query
        )
        
        # Analyze query characteristics
        is_summary = self._detect_summary_query(query)
        is_comprehensive = len(context_chunks) > 50
        
        # Prepare template variables
        template_vars = {
            'query': query,
            'context': optimized_context,
            'document_count': len(included_documents),
            'date_range': self._get_date_range_text(context_chunks),
            'context_instructions': self._get_context_instructions(is_summary, is_comprehensive),
            'response_instructions': self._get_response_instructions(query, is_summary),
            'format_instructions': self._get_format_instructions(query)
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            template_vars.update(additional_metadata)
        
        try:
            formatted_prompt = self.universal_template.format(**template_vars)
            logger.info(f"Generated enhanced prompt: {len(formatted_prompt)} characters")
            return formatted_prompt
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using fallback")
            # Fallback with minimal variables
            return self.universal_template.format(
                query=query,
                context=optimized_context,
                document_count=len(included_documents),
                date_range=" spanning available meetings",
                context_instructions="",
                response_instructions="Please answer the question naturally and comprehensively.",
                format_instructions=""
            )
    
    def _calculate_date_range(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Calculate date range from context chunks."""
        dates = []
        
        for chunk in context_chunks:
            if 'timestamp' in chunk:
                try:
                    date = datetime.fromisoformat(chunk['timestamp'])
                    dates.append(date)
                except:
                    continue
        
        if not dates:
            return "Recent meetings"
        
        dates.sort()
        if len(dates) == 1:
            return dates[0].strftime("%Y-%m-%d")
        else:
            return f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    
    def _get_date_range_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Get formatted date range text for template."""
        date_range = self._calculate_date_range(context_chunks)
        if date_range and date_range != "Recent meetings":
            return f" spanning {date_range}"
        return ""
    
    def _get_context_instructions(self, is_summary: bool, is_comprehensive: bool) -> str:
        """Generate context-specific instructions."""
        if is_comprehensive:
            return "This is a comprehensive analysis requiring detailed examination of all documents."
        elif is_summary:
            return "This is a summary request requiring overview of key information."
        return ""
    
    def _get_response_instructions(self, query: str, is_summary: bool) -> str:
        """Generate response instructions based on query type."""
        if is_summary:
            return """Please provide a comprehensive summary organized by meeting dates:
1. Start with a brief overview
2. Organize by meeting dates using clear headers (e.g., "## July 14, 2025")
3. Under each date include: purpose, participants, topics, decisions, action items"""
        
        query_lower = query.lower()
        if any(word in query_lower for word in ['analyze', 'analysis', 'detailed', 'examine', 'assess']):
            return """Please provide detailed analysis with:
1. Contextual background and significance
2. Multiple perspectives and evidence
3. Key insights and implications
4. Actionable recommendations where appropriate"""
        
        return """Please answer the question naturally and comprehensively:
1. Address the specific question asked
2. Provide relevant details and context
3. Include supporting evidence from the documents"""
    
    def _get_format_instructions(self, query: str) -> str:
        """Generate format instructions based on query content."""
        query_lower = query.lower()
        instructions = []
        
        if any(word in query_lower for word in ['timeline', 'chronological', 'when']):
            instructions.append("- Organize information chronologically")
        if any(word in query_lower for word in ['decision', 'decided', 'conclusion']):
            instructions.append("- Highlight decisions made and their context")
        if any(word in query_lower for word in ['action', 'tasks', 'todo', 'next steps']):
            instructions.append("- Extract action items with owners and deadlines")
        if any(word in query_lower for word in ['who', 'participants', 'attendees']):
            instructions.append("- Clearly identify participants and their roles")
        
        return "\n".join(instructions) if instructions else ""
    
    def _format_context_with_sources(self, document_chunks: Dict[str, List[str]]) -> str:
        """Format context with clear document source headers."""
        if not document_chunks:
            return "No relevant content found."
        
        formatted_sections = []
        
        for doc_name, chunks in document_chunks.items():
            if chunks:
                section = f"=== SOURCE: {doc_name} ===\n"
                section += "\n\n".join(chunks)
                formatted_sections.append(section)
        
        return "\n\n".join(formatted_sections)
    
