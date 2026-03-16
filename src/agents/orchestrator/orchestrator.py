"""
orchestrator.py — The Master Orchestrator
==========================================
Receives natural language messages from the user (via Telegram),
uses Groq LLM with tool calling to understand intent, executes
the appropriate action, and returns a human-readable response.

    LangChain handles the plumbing:
    1. User message → LLM (with tools bound)
    2. LLM returns a tool call
    3. We execute the tool
    4. Tool result → LLM again
    5. LLM generates a final human-readable response

Session Store:
    The Orchestrator maintains a dict mapping video names to their
    completed analysis results. This allows multi-video conversations
    where the user can ask about different matches.
"""

import os
import time
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

from src.graph.match_graph import build_match_graph

load_dotenv()


# =============================================================================
# TOOLS — Functions the LLM can call
# =============================================================================
# Each @tool function has a docstring that the LLM reads to decide
# WHEN to use it. The better the docstring, the smarter the routing.
# =============================================================================


@tool
def analyze_video(video_path: str) -> str:
    """
    Run the full match analysis pipeline on a video file.
    Use this when the user wants to analyze a new match video.
    The video must exist in the data/ folder.

    Args:
        video_path: Path to the video file (e.g. "data/match_clip.mp4")
    """
    # This gets overridden by the Orchestrator
    return f"Analysis started for {video_path}"


@tool
def get_match_summary(video_name: str) -> str:
    """
    Get the full match report for a previously analyzed video.
    Use this when the user asks for a summary, report, or overview
    of a match that has already been analyzed.

    Args:
        video_name: Name of the video (e.g. "match_clip" or "match_clip.mp4")
    """
    return f"Summary requested for {video_name}"


@tool
def get_player_stats(video_name: str, track_id: int) -> str:
    """
    Get statistics for a specific player by their track ID.
    Use this when the user asks about a specific player's performance,
    like "how did player 7 perform?" or "stats for track 96".

    Args:
        video_name: Name of the video to look up
        track_id: The player's tracking ID number
    """
    return f"Player stats requested: track {track_id} in {video_name}"


@tool
def get_team_stats(video_name: str) -> str:
    """
    Get and compare team statistics for a previously analyzed video.
    Use this when the user asks about team performance, possession,
    or wants to compare Team A vs Team B.

    Args:
        video_name: Name of the video to look up
    """
    return f"Team stats requested for {video_name}"


@tool
def list_analyses() -> str:
    """
    List all videos that have been analyzed in this session.
    Use this when the user asks "what matches have I analyzed?"
    or "which videos are available?"
    """
    return "Listing analyses"


TOOLS = [
    analyze_video,
    get_match_summary,
    get_player_stats,
    get_team_stats,
    list_analyses,
]


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class Orchestrator:
    """
    The master agent that coordinates the entire system.

    Responsibilities:
        - Parse user messages using LLM + tool calling
        - Run the LangGraph pipeline when requested
        - Store and retrieve analysis results per video
        - Answer follow-up questions about analyzed matches
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        # Bind tools so the LLM knows it can call them
        self.llm_with_tools = self.llm.bind_tools(TOOLS)

        # Session store: video_name → analysis results
        # This is what enables multi-video conversations
        self.session_store = {}

        # Conversation history for context
        self.conversation_history = []

        # Compiled LangGraph pipeline
        self.graph = build_match_graph()

        self.system_prompt = SystemMessage(
            content="""You are a football match analysis assistant.
You help users analyze football match videos and answer questions about the results.

RULES:
- When the user wants to analyze a video, use the analyze_video tool.
- When the user asks about match results, use get_match_summary or get_team_stats.
- When the user asks about a specific player, use get_player_stats.
- When the user asks what's been analyzed, use list_analyses.
- If only one video has been analyzed, assume the user is asking about that one.
- Be concise and use football terminology naturally.
- When presenting stats, focus on insights, not just raw numbers.
- All distances/speeds are in pixels — compare players relatively, don't state absolute values as meaningful.
- When tool results contain formatted tables or stats, present them EXACTLY as received. Do not rewrite or paraphrase formatted data."""
        )

    def _normalize_video_name(self, name: str) -> str:
        """Strip extension and path to get a clean video name."""
        return Path(name).stem

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        Actually execute a tool call from the LLM.

        This is where the real logic lives — the @tool functions above
        are just schemas for the LLM. The actual work happens here.
        """

        if tool_name == "analyze_video":
            return self._run_analysis(tool_args["video_path"])

        elif tool_name == "get_match_summary":
            name = self._normalize_video_name(tool_args["video_name"])
            if name not in self.session_store:
                return f"No analysis found for '{name}'. Available: {list(self.session_store.keys())}"
            report = self.session_store[name].get("report", "No report available.")
            # Truncate to avoid blowing Groq's token limit in conversation history
            if len(report) > 1000:
                return report[:1000] + "\n\n[Truncated — ask about specific players or teams for details]"
            
            return report

        elif tool_name == "get_player_stats":
            name = self._normalize_video_name(tool_args["video_name"])
            track_id = tool_args["track_id"]
            if name not in self.session_store:
                return f"No analysis found for '{name}'."
            player_stats = self.session_store[name].get("player_stats", {})
            stats = player_stats.get(track_id) or player_stats.get(str(track_id))
            if not stats:
                available = sorted([str(k) for k in player_stats.keys()])[:20]
                return f"No player with track ID {track_id}. Available IDs: {', '.join(available)}"
            return (
                f"📊 PLAYER #{stats['track_id']} — Team {stats['team']}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Passes made:          {stats['passes_made']}\n"
                f"Passes received:      {stats['passes_received']}\n"
                f"Tackles made:         {stats['tackles_made']}\n"
                f"Tackles suffered:     {stats['tackles_suffered']}\n"
                f"Interceptions made:   {stats['interceptions_made']}\n"
                f"Interceptions suffered: {stats['interceptions_suffered']}\n"
                f"Turnovers lost:       {stats['turnovers_lost']}\n"
                f"Turnovers won:        {stats['turnovers_won']}\n"
                f"Possession frames:    {stats['possession_frames']}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Distance covered:     {stats['distance_px']}px\n"
                f"Top speed:            {stats['top_speed_px_s']}px/s\n"
                f"Sprints:              {stats['sprint_count']}"
            )

        elif tool_name == "get_team_stats":
            name = self._normalize_video_name(tool_args["video_name"])
            if name not in self.session_store:
                return f"No analysis found for '{name}'."
            result = self.session_store[name]
            team_stats = result.get("team_stats", {})
            total_poss = sum(t.get("possession_frames", 0) for t in team_stats.values())

            ts_a = team_stats.get("A", {})
            ts_b = team_stats.get("B", {})
            poss_a = (ts_a.get("possession_frames", 0) / total_poss * 100) if total_poss > 0 else 0
            poss_b = (ts_b.get("possession_frames", 0) / total_poss * 100) if total_poss > 0 else 0

            return (
                f"⚽ TEAM COMPARISON — {name}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{'Stat':<20} {'Team A':>8} {'Team B':>8}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{'Possession':<20} {poss_a:>7.1f}% {poss_b:>7.1f}%\n"
                f"{'Passes':<20} {ts_a.get('total_passes', 0):>8} {ts_b.get('total_passes', 0):>8}\n"
                f"{'Tackles Won':<20} {ts_a.get('total_tackles', 0):>8} {ts_b.get('total_tackles', 0):>8}\n"
                f"{'Interceptions':<20} {ts_a.get('total_interceptions', 0):>8} {ts_b.get('total_interceptions', 0):>8}\n"
                f"{'Turnovers Lost':<20} {ts_a.get('total_turnovers_lost', 0):>8} {ts_b.get('total_turnovers_lost', 0):>8}\n"
                f"{'Avg Distance (px)':<20} {ts_a.get('avg_distance_px', 0):>8} {ts_b.get('avg_distance_px', 0):>8}\n"
                f"{'Max Speed (px/s)':<20} {ts_a.get('avg_top_speed_px_s', 0):>8} {ts_b.get('avg_top_speed_px_s', 0):>8}\n"
                f"{'Total Sprints':<20} {ts_a.get('total_sprints', 0):>8} {ts_b.get('total_sprints', 0):>8}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

        elif tool_name == "list_analyses":
            if not self.session_store:
                return "No videos have been analyzed yet."
            lines = []
            for name, data in self.session_store.items():
                status = data.get("status", "unknown")
                n_players = len(data.get("player_stats", {}))
                lines.append(f"  {name}: status={status}, {n_players} players tracked")
            return "Analyzed videos:\n" + "\n".join(lines)

        return f"Unknown tool: {tool_name}"

    def _run_analysis(self, video_path: str) -> str:
        """Run the full LangGraph pipeline on a video."""
        if not os.path.exists(video_path):
            return f"Video file not found: {video_path}"

        name = self._normalize_video_name(video_path)

        initial_state = {
            "video_path": video_path,
            "max_frames": None,
            "tracking_data": [],
            "enriched_data": [],
            "possession_log": [],
            "passes": [],
            "turnovers": [],
            "tackles": [],
            "interceptions": [],
            "player_stats": {},
            "team_stats": {},
            "report": "",
            "status": "starting",
            "errors": [],
        }

        result = self.graph.invoke(initial_state)

        # Store results in session
        self.session_store[name] = result

        # Build a quick summary to return to the LLM
        summary = (
            f"Analysis complete for {name}.\n"
            f"Players tracked: {len(result.get('player_stats', {}))}\n"
            f"Passes: {len(result.get('passes', []))}\n"
            f"Tackles: {len(result.get('tackles', []))}\n"
            f"Interceptions: {len(result.get('interceptions', []))}\n"
            f"Status: {result.get('status', 'unknown')}"
        )
        return summary
    
    def _trim_history(self, max_tokens: int = 6000):
        """
        Trim conversation history to stay under the token budget.
        Removes oldest messages first. Estimates ~4 chars per token.

        This is critical for Groq free tier (12k TPM limit).
        System prompt uses ~500 tokens, LLM response needs ~2000,
        so history gets ~6000 tokens max.
        """
        while self._estimate_tokens() > max_tokens and len(self.conversation_history) > 2:
            self.conversation_history.pop(0)

    def _sanitize_response(self, text: str) -> str:
        """
        Clean up LLM responses before sending to the user.
        Removes leaked function call tags that Llama sometimes hallucinates.
        """
        import re
        # Remove <function=...>...</function> tags
        text = re.sub(r'<function=\w+>.*?</function>', '', text)
        # Remove any leftover XML-like tags
        text = re.sub(r'<\|.*?\|>', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    def _estimate_tokens(self) -> int:
        """Rough token estimate for conversation history. ~4 chars = 1 token."""
        total_chars = 0
        for msg in self.conversation_history:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            total_chars += len(content)
        return total_chars // 4
    
    def _safe_llm_invoke(self, messages, use_tools=False):
        """
        Call Groq with automatic retry on rate limit errors.
        Waits and retries up to 3 times if TPM limit is hit.

        Groq free tier: 12,000 tokens per minute.
        When the limit is hit, we wait 60 seconds and retry.
        """
        llm = self.llm_with_tools if use_tools else self.llm
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return llm.invoke(messages)
            except Exception as e:
                error_str = str(e)
                if "413" in error_str or "rate_limit" in error_str or "tokens" in error_str:
                    wait_time = 30 * (attempt + 1)
                    print(f"[Orchestrator] Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e

        raise Exception("Groq rate limit exceeded after 3 retries. Try again in a minute.")

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point — Telegram will call this.

        Flow:
            1. Add user message to conversation history
            2. Send history to LLM (with tools bound)
            3. If LLM wants to call a tool → execute it → send result back to LLM
            4. LLM generates final response
            5. Return response text
        """
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))

        # Keep conversation history under control (Groq free tier = 12k TPM)
        # Estimate tokens and trim oldest messages until under budget
        self._trim_history(max_tokens=3000)

        # Build messages: system prompt + conversation history
        messages = [self.system_prompt] + self.conversation_history

        # First LLM call — might return text or a tool call
        response = self._safe_llm_invoke(messages, use_tools=True)

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            # LLM decided to call a tool
            tool_call = response.tool_calls[0]  # Handle first tool call
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"[Orchestrator] Tool call: {tool_name}({tool_args})")

            # Execute the tool
            tool_result = self._execute_tool(tool_name, tool_args)

            # Store a SHORT version in conversation history to save tokens.
            # The full data stays in session_store — tools can access it anytime.
            short_result = tool_result[:500] if len(tool_result) > 500 else tool_result

            # Add the AI's tool call and the SHORT result to history
            self.conversation_history.append(response)
            self.conversation_history.append(
                ToolMessage(content=short_result, tool_call_id=tool_call["id"])
            )

            # Second LLM call — now with the tool result, generate final response
            messages = [self.system_prompt] + self.conversation_history
            final_response = self._safe_llm_invoke(messages, use_tools=False)

            self.conversation_history.append(final_response)
            return self._sanitize_response(final_response.content)

        else:
            # LLM just responded with text (no tool call needed)
            self.conversation_history.append(response)
            return self._sanitize_response(response.content)
