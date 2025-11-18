"""
Collaborative Discussion Mode Manager for PolyCouncil
Manages turn-based conversational workflow between agents.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import aiohttp

# Core system instruction for Collaborative Discussion Mode
CORE_SYSTEM_INSTRUCTION = """You are participating in a Collaborative Discussion to fully examine the user's request. Speak in character based on your assigned persona. Your message must be brief, constructive, and move the discussion forward. The discussion is ongoing until a comprehensive, multi-sided consensus or ruling is reached."""


class DiscussionManager:
    """Manages the collaborative discussion workflow."""
    
    def __init__(
        self,
        base_url: str,
        agents: List[Dict],
        user_prompt: str,
        context_block: str = "",
        images: List[str] = [],  # List of base64 image strings
        web_search_enabled: bool = False,
        temperature: float = 0.7,
        status_callback: Optional[Callable[[str], None]] = None,
        update_callback: Optional[Callable[[Dict], None]] = None,
        max_turns: int = 10,
        max_concurrency: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize the discussion manager.
        
        Args:
            base_url: LM Studio base URL
            agents: List of agent configurations with persona_config
            user_prompt: The user's main question/prompt
            context_block: Pre-formatted context from file parsing (if any)
            images: List of base64 encoded image strings
            web_search_enabled: Whether to enable web search tools
            temperature: Model temperature
            status_callback: Optional callback for status updates
            max_turns: Maximum number of discussion turns
            max_concurrency: Maximum concurrent model calls
            is_cancelled: Callable to check for cancellation
        """
        self.base_url = base_url
        self.agents = agents
        self.user_prompt = user_prompt
        self.context_block = context_block
        self.images = images
        self.web_search_enabled = web_search_enabled
        self.temperature = temperature
        self.status_callback = status_callback
        self.update_callback = update_callback
        self.max_turns = max_turns
        self.max_concurrency = max_concurrency
        self.is_cancelled = is_cancelled
        
        self.transcript: List[Dict[str, str]] = []
        self.turn_count = 0
        self.consensus_reached = False
        self.conversation_summary: str = ""  # Rolling summary of older conversation
        self.summarized_context: str = ""  # Summarized file context
    
    def _get_persona_prompt(self, agent: Dict) -> str:
        """
        Get the persona prompt for an agent based on persona_config.
        
        Args:
            agent: Agent configuration dict
            
        Returns:
            Persona prompt string
        """
        persona_config = agent.get("persona_config", {})
        source = persona_config.get("source", "default")
        
        if source == "one_time":
            return persona_config.get("one_time_prompt", "")
        elif source == "default":
            persona_id = persona_config.get("id")
            if persona_id:
                return self._load_default_persona(persona_id)
        elif source == "user_custom":
            persona_id = persona_config.get("id")
            if persona_id:
                return self._load_user_persona(persona_id)
        
        return ""
    
    def _load_default_persona(self, persona_id: str) -> str:
        """Load persona prompt from default_personas.json."""
        try:
            default_path = Path(__file__).parent.parent / "config" / "default_personas.json"
            if default_path.exists():
                with open(default_path, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                    for persona in personas:
                        if persona.get("id") == persona_id:
                            return persona.get("prompt_instruction", "")
        except Exception as e:
            print(f"Error loading default persona {persona_id}: {e}")
        return ""
    
    def _load_user_persona(self, persona_id: str) -> str:
        """Load persona prompt from user_personas.json."""
        try:
            user_path = Path(__file__).parent.parent / "config" / "user_personas.json"
            if user_path.exists():
                with open(user_path, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                    for persona in personas:
                        if persona.get("id") == persona_id:
                            return persona.get("prompt_instruction", "")
        except Exception as e:
            print(f"Error loading user persona {persona_id}: {e}")
        return ""
    
    def _assemble_prompt(self, agent: Dict, conversation_history: List[Dict]) -> str:
        """
        Assemble the final prompt for an agent with intelligent context management.
        Total target: ~6000 tokens to leave room for response.
        
        Order:
        1. Summarized file context (if file uploaded) - ~800 tokens max
        2. CORE_SYSTEM_INSTRUCTION - ~50 tokens
        3. Agent's Persona Prompt - ~50 tokens
        4. User's Main Prompt - ~100 tokens
        5. Conversation Summary (older turns) - ~300 tokens max
        6. Immediate History (last 2-3 exchanges) - ~500 tokens max
        """
        parts = []
        total_estimated_tokens = 0
        max_total_tokens = 6000  # Conservative limit for 10k context window
        
        # 1. Summarized file context (intelligently processed) - ~800 tokens
        if self.summarized_context:
            file_context = f"=== CONTEXT FROM UPLOADED FILE ===\n\n{self.summarized_context}\n\n=== END CONTEXT ===\n"
            file_tokens = self._estimate_tokens(file_context)
            if total_estimated_tokens + file_tokens <= max_total_tokens * 0.4:  # Max 40% for file
                parts.append(file_context)
                total_estimated_tokens += file_tokens
            else:
                # Further truncate if needed
                max_file_chars = int((max_total_tokens * 0.4 - total_estimated_tokens) * 4)
                truncated_file = self.summarized_context[:max_file_chars]
                if len(self.summarized_context) > max_file_chars:
                    truncated_file += "\n\n[... file context truncated ...]"
                parts.append(f"=== CONTEXT FROM UPLOADED FILE ===\n\n{truncated_file}\n\n=== END CONTEXT ===\n")
                total_estimated_tokens += self._estimate_tokens(truncated_file)
        
        # 2. Core system instruction - ~50 tokens
        parts.append(CORE_SYSTEM_INSTRUCTION)
        total_estimated_tokens += self._estimate_tokens(CORE_SYSTEM_INSTRUCTION)
        
        # 3. Persona prompt - ~50 tokens
        persona_prompt = self._get_persona_prompt(agent)
        if persona_prompt:
            persona_text = f"Your Persona: {persona_prompt}"
            parts.append(persona_text)
            total_estimated_tokens += self._estimate_tokens(persona_text)
        
        # 4. User prompt - ~100 tokens
        user_text = f"\nUser's Request: {self.user_prompt}\n"
        parts.append(user_text)
        total_estimated_tokens += self._estimate_tokens(user_text)
        
        # 5. Conversation summary (older turns) - ~300 tokens max
        if self.conversation_summary:
            summary_text = f"=== EARLIER DISCUSSION SUMMARY ===\n{self.conversation_summary}\n=== END SUMMARY ===\n"
            summary_tokens = self._estimate_tokens(summary_text)
            if total_estimated_tokens + summary_tokens <= max_total_tokens * 0.8:  # Leave room for recent history
                parts.append(summary_text)
                total_estimated_tokens += summary_tokens
            else:
                # Truncate summary if needed
                max_summary_chars = int((max_total_tokens * 0.8 - total_estimated_tokens) * 4)
                truncated_summary = self.conversation_summary[:max_summary_chars]
                if len(self.conversation_summary) > max_summary_chars:
                    truncated_summary += "... [summary truncated]"
                parts.append(f"=== EARLIER DISCUSSION SUMMARY ===\n{truncated_summary}\n=== END SUMMARY ===\n")
                total_estimated_tokens += self._estimate_tokens(truncated_summary)
        
        # 6. Immediate conversation history (last 2 exchanges, not 3) - ~500 tokens max
        if conversation_history:
            # Determine which entries are "immediate" (not already summarized)
            immediate_turns = []
            if self.conversation_summary:
                # Only show turns that came after the summary
                if conversation_history:
                    summary_cutoff = max([e.get("turn", 0) for e in conversation_history]) - 2
                    immediate_turns = [e for e in conversation_history if e.get("turn", 0) > summary_cutoff]
            else:
                immediate_turns = conversation_history[-2:]  # Only last 2 to save tokens
            
            if immediate_turns:
                # Build history text with token limits
                history_lines = []
                remaining_tokens = max_total_tokens - total_estimated_tokens - 200  # Reserve 200 for final instruction
                
                for entry in immediate_turns[-2:]:  # Max 2 exchanges
                    agent_name = entry.get('agent', 'Unknown')
                    message = entry.get('message', '')
                    
                    # Estimate tokens for this entry
                    entry_text = f"[Turn {entry.get('turn', 0)}] {agent_name}: {message}"
                    entry_tokens = self._estimate_tokens(entry_text)
                    
                    # If message is too long, truncate it
                    if entry_tokens > 200:  # Max 200 tokens per message
                        max_msg_chars = 200 * 4  # ~800 characters
                        message = message[:max_msg_chars] + "... [message continues]"
                        entry_text = f"[Turn {entry.get('turn', 0)}] {agent_name}: {message}"
                        entry_tokens = self._estimate_tokens(entry_text)
                    
                    if remaining_tokens >= entry_tokens:
                        history_lines.append(entry_text)
                        remaining_tokens -= entry_tokens
                    else:
                        break
                
                if history_lines:
                    parts.append("=== RECENT DISCUSSION (Last 2 Exchanges) ===")
                    parts.extend(history_lines)
                    parts.append("=== END RECENT DISCUSSION ===\n")
        
        parts.append("Your turn to contribute to the discussion:")
        
        final_prompt = "\n\n".join(parts)
        
        # Final safety check - if still too large, truncate aggressively
        final_tokens = self._estimate_tokens(final_prompt)
        if final_tokens > max_total_tokens:
            # Emergency truncation - keep essential parts, cut from middle
            essential_parts = [
                CORE_SYSTEM_INSTRUCTION,
                f"User's Request: {self.user_prompt}",
                "Your turn to contribute to the discussion:"
            ]
            if persona_prompt:
                essential_parts.insert(1, f"Your Persona: {persona_prompt}")
            
            # Add minimal context if available
            if self.summarized_context:
                min_context = self.summarized_context[:1000]  # Very small
                essential_parts.insert(0, f"Context: {min_context}...")
            
            # Add only the very last exchange
            if conversation_history:
                last_entry = conversation_history[-1]
                agent_name = last_entry.get('agent', 'Unknown')
                message = last_entry.get('message', '')[:400]  # Very short
                essential_parts.insert(-1, f"Last: {agent_name}: {message}...")
            
            return "\n\n".join(essential_parts)
        
        return final_prompt
    
    async def _call_agent(
        self,
        session: aiohttp.ClientSession,
        agent: Dict,
        conversation_history: List[Dict]
    ) -> Optional[str]:
        """Call a single agent in the discussion."""
        model = agent.get("model")
        if not model:
            return None
        
        prompt = self._assemble_prompt(agent, conversation_history)
        
        try:
            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            
            # Construct message content (multimodal if images exist)
            if self.images:
                content_list = [{"type": "text", "text": prompt}]
                for img_b64 in self.images:
                    content_list.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                messages = [{"role": "user", "content": content_list}]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Inject Web Search Tools if enabled
            if self.web_search_enabled:
                payload["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": "google_search",
                        "description": "Search the web for current information.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"]
                        }
                    }
                }]
            
            timeout = aiohttp.ClientTimeout(total=120)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:800]}")
                
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
                
        except Exception as e:
            print(f"Error calling agent {agent.get('name', model)}: {e}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation: ~1 token per 4 characters for English text.
        This is a conservative estimate.
        """
        if not text:
            return 0
        return len(text) // 4
    
    def _summarize_file_context(self, context_block: str, user_prompt: str) -> str:
        """
        Intelligently summarize file context using keyword-based chunk retrieval.
        
        Args:
            context_block: Full file content
            user_prompt: User's question/prompt to extract keywords
            
        Returns:
            Summarized context (max ~800 tokens = ~3200 characters)
        """
        if not context_block:
            return ""
        
        # More aggressive limit: ~800 tokens = ~3200 characters
        max_chars = 3200
        max_tokens = 800
        
        # If already small enough, return as-is
        if len(context_block) <= max_chars:
            return context_block
        
        # Extract keywords from user prompt (simple approach)
        # Get meaningful words (3+ characters, not common stop words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', user_prompt.lower())
        keywords = [w for w in words if w not in stop_words][:10]  # Top 10 keywords
        
        if not keywords:
            # No keywords found, use simple truncation with smart cutoff
            if len(context_block) <= max_chars:
                return context_block
            # Try to cut at sentence boundary
            truncated = context_block[:max_chars]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cutoff = max(last_period, last_newline)
            if cutoff > max_chars * 0.7:  # Only use smart cutoff if it's not too early
                return context_block[:cutoff + 1] + "\n\n[... context continues ...]"
            return truncated + "\n\n[... context truncated ...]"
        
        # Split context into chunks (by paragraphs or sentences)
        chunks = re.split(r'\n\n+', context_block)
        if len(chunks) == 1:
            # No paragraph breaks, try sentence breaks
            chunks = re.split(r'[.!?]\s+', context_block)
        
        # Score chunks by keyword relevance
        chunk_scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(chunk_lower.count(keyword) for keyword in keywords)
            if score > 0:
                chunk_scores.append((score, chunk))
        
        # Sort by relevance and take top chunks
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Build summarized context from top chunks
        summarized_parts = []
        total_length = 0
        
        for score, chunk in chunk_scores:
            chunk_tokens = self._estimate_tokens(chunk)
            current_tokens = self._estimate_tokens("\n\n".join(summarized_parts))
            
            if current_tokens + chunk_tokens <= max_tokens:
                summarized_parts.append(chunk)
                total_length += len(chunk)
            else:
                # Add partial chunk if there's room
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Only if meaningful space left (~200 chars)
                    remaining_chars = remaining_tokens * 4
                    summarized_parts.append(chunk[:remaining_chars] + "...")
                break
        
        if summarized_parts:
            result = "\n\n".join(summarized_parts)
            if len(context_block) > max_chars:
                result += "\n\n[... additional context available in full document ...]"
            return result
        
        # Fallback: if no keyword matches, use first portion with smart cutoff
        truncated = context_block[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cutoff = max(last_period, last_newline)
        if cutoff > max_chars * 0.7:
            return context_block[:cutoff + 1] + "\n\n[... context truncated ...]"
        return truncated + "\n\n[... context truncated ...]"
    
    async def _generate_conversation_summary(
        self,
        session: aiohttp.ClientSession,
        older_history: List[Dict],
        summary_agent: Dict
    ) -> str:
        """
        Generate a rolling summary of older conversation turns.
        
        Args:
            session: aiohttp session
            older_history: Conversation entries from turns 4-10 (or similar)
            summary_agent: Agent to use for summarization
            
        Returns:
            Concise summary string
        """
        if not older_history or len(older_history) < 2:
            return ""
        
        # Format older history for summarization
        history_text = "\n\n".join([
            f"[Turn {entry.get('turn', 0)}] {entry.get('agent', 'Unknown')}: {entry.get('message', '')[:200]}"
            for entry in older_history
        ])
        
        summary_prompt = f"""Summarize the following discussion history in 2-3 concise sentences. Focus on:
1. Key points and arguments raised
2. Areas of agreement or disagreement
3. Important conclusions or insights

Discussion History:
{history_text}

Summary:"""
        
        try:
            model = summary_agent.get("model")
            if not model:
                return ""
            
            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "temperature": 0.3,
                "max_tokens": 200,  # Keep summary short
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return ""
                
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip()
                
        except Exception as e:
            print(f"Error generating conversation summary: {e}")
            return ""
    
    async def run_discussion(self) -> Tuple[List[Dict], Optional[str]]:
        """
        Run the collaborative discussion.
        
        Returns:
            Tuple of (transcript, final_synthesis)
        """
        if self.status_callback:
            self.status_callback("Starting collaborative discussion...")
        
        # Pre-process file context with intelligent summarization
        if self.context_block:
            original_size = len(self.context_block)
            self.summarized_context = self._summarize_file_context(self.context_block, self.user_prompt)
            summarized_size = len(self.summarized_context)
            if self.status_callback:
                if original_size > summarized_size:
                    reduction = ((original_size - summarized_size) / original_size) * 100
                    self.status_callback(f"File context summarized: {original_size:,} → {summarized_size:,} chars ({reduction:.1f}% reduction)")
                else:
                    self.status_callback("File context processed...")
        
        async with aiohttp.ClientSession() as session:
            conversation_history = []
            
            for turn in range(self.max_turns):
                self.turn_count = turn + 1
                
                if self.status_callback:
                    self.status_callback(f"Discussion turn {self.turn_count}/{self.max_turns}...")
                
                # Collect responses from all active agents
                active_agents = [a for a in self.agents if a.get("is_active", True)]
                
                if not active_agents:
                    break
                
                # Call agents concurrently (limited by max_concurrency)
                semaphore = asyncio.Semaphore(self.max_concurrency)
                
                async def call_with_semaphore(agent):
                    async with semaphore:
                        return await self._call_agent(session, agent, conversation_history)
                
                tasks = [call_with_semaphore(agent) for agent in active_agents]
                responses = await asyncio.gather(*tasks)
                
                # Record responses in transcript and emit updates
                for agent, response in zip(active_agents, responses):
                    if response:
                        agent_name = agent.get("name", agent.get("model", "Unknown"))
                        persona_name = agent.get("persona_name")
                        entry = {
                            "turn": self.turn_count,
                            "agent": agent_name,
                            "model": agent.get("model"),
                            "persona": persona_name,
                            "message": response
                        }
                        self.transcript.append(entry)
                        conversation_history.append(entry)
                        
                        # Emit real-time update
                        if self.update_callback:
                            self.update_callback(entry)
                
                # Generate rolling summary every 3-4 turns
                if self.turn_count >= 4 and self.turn_count % 3 == 0:
                    # Summarize turns 1 through (current - 3)
                    older_turns = [e for e in conversation_history if e.get("turn", 0) < self.turn_count - 2]
                    if older_turns and len(older_turns) > 2:
                        active_agents = [a for a in self.agents if a.get("is_active", True)]
                        if active_agents:
                            summary = await self._generate_conversation_summary(
                                session, older_turns, active_agents[0]
                            )
                            if summary:
                                self.conversation_summary = summary
                                if self.status_callback:
                                    self.status_callback(f"Discussion summary updated (turn {self.turn_count})...")
                
                # Check for consensus (simple heuristic: can be enhanced)
                if self._check_consensus(conversation_history):
                    self.consensus_reached = True
                    if self.status_callback:
                        self.status_callback("Consensus reached, generating synthesis...")
                    break
                
                # Small delay between turns
                await asyncio.sleep(0.5)
            
            # Generate final synthesis
            final_synthesis = await self._generate_synthesis(session, conversation_history)
            
            if self.status_callback:
                self.status_callback("Discussion complete.")
            
            return self.transcript, final_synthesis
    
    def _check_consensus(self, history: List[Dict]) -> bool:
        """
        Simple heuristic to check if consensus is reached.
        Can be enhanced with more sophisticated logic.
        """
        if len(history) < 3:
            return False
        
        # Check if last few messages indicate convergence
        recent = history[-3:]
        # Simple check: if we have enough turns, consider it done
        # This can be made more sophisticated
        return self.turn_count >= 5
    
    async def _generate_synthesis(
        self,
        session: aiohttp.ClientSession,
        conversation_history: List[Dict]
    ) -> Optional[str]:
        """Generate a final synthesis of the discussion."""
        try:
            if self.status_callback:
                self.status_callback("Generating final synthesis...")
            
            # Use the first active agent to generate synthesis
            active_agents = [a for a in self.agents if a.get("is_active", True)]
            if not active_agents:
                if self.status_callback:
                    self.status_callback("No active agents for synthesis generation")
                return None
            
            synthesis_agent = active_agents[0]
            model = synthesis_agent.get("model")
            
            # Format discussion for synthesis (limit to prevent context overflow)
            # Use summarized version if available, otherwise use recent history
            if self.conversation_summary and len(conversation_history) > 5:
                # Use summary + recent turns
                recent_turns = conversation_history[-3:]
                discussion_text = f"{self.conversation_summary}\n\nRecent turns:\n" + "\n\n".join([
                    f"{entry.get('agent', 'Unknown')}: {entry.get('message', '')[:500]}"
                    for entry in recent_turns
                ])
            else:
                # Use all history but truncate long messages
                discussion_text = "\n\n".join([
                    f"{entry.get('agent', 'Unknown')}: {entry.get('message', '')[:800]}"
                    for entry in conversation_history[-10:]  # Last 10 entries max
                ])
            
            synthesis_prompt = f"""Based on the following collaborative discussion, provide a comprehensive, multi-faceted synthesis that:

1. Summarizes the key viewpoints and arguments presented
2. Identifies areas of agreement and disagreement
3. Presents a balanced conclusion that incorporates multiple perspectives
4. Clearly explains the reasoning process

Discussion:
{discussion_text}

Synthesis:"""
            
            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": synthesis_prompt}],
                "temperature": 0.5,
            }
            
            timeout = aiohttp.ClientTimeout(total=180)  # Longer timeout for synthesis
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if self.status_callback:
                        self.status_callback(f"Synthesis generation failed: HTTP {resp.status}")
                    print(f"Synthesis generation error: HTTP {resp.status} - {error_text[:500]}")
                    return None
                
                data = await resp.json()
                
                # Extract content - handle different response formats
                content = None
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                    elif "text" in choice:
                        content = choice.get("text", "")
                
                # Also check for direct content field
                if not content and "content" in data:
                    content = data["content"]
                
                if content:
                    content = content.strip()
                    if self.status_callback:
                        self.status_callback(f"Synthesis generated ({len(content)} characters)")
                    return content
                else:
                    if self.status_callback:
                        self.status_callback("Synthesis response was empty")
                    print(f"Synthesis response had no content. Full response: {data}")
                    return None
                
        except asyncio.TimeoutError:
            if self.status_callback:
                self.status_callback("Synthesis generation timed out")
            print("Error generating synthesis: Timeout")
            return None
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Synthesis generation error: {str(e)[:50]}")
            print(f"Error generating synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None

