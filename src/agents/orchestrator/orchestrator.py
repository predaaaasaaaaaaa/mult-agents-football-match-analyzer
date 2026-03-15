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


TOOLS = [analyze_video, get_match_summary, get_player_stats, get_team_stats, list_analyses]


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

        self.system_prompt = SystemMessage(content="""You are a football match analysis assistant.
You help users analyze football match videos and answer questions about the results.

RULES:
- When the user wants to analyze a video, use the analyze_video tool.
- When the user asks about match results, use get_match_summary or get_team_stats.
- When the user asks about a specific player, use get_player_stats.
- When the user asks what's been analyzed, use list_analyses.
- If only one video has been analyzed, assume the user is asking about that one.
- Be concise and use football terminology naturally.
- When presenting stats, focus on insights, not just raw numbers.
- All distances/speeds are in pixels — compare players relatively, don't state absolute values as meaningful.""")

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
            return self.session_store[name].get("report", "No report available.")

        elif tool_name == "get_player_stats":
            name = self._normalize_video_name(tool_args["video_name"])
            track_id = tool_args["track_id"]
            if name not in self.session_store:
                return f"No analysis found for '{name}'."
            player_stats = self.session_store[name].get("player_stats", {})
            # track_id might be int or str depending on how it was stored
            stats = player_stats.get(track_id) or player_stats.get(str(track_id))
            if not stats:
                available = sorted([str(k) for k in player_stats.keys()])[:20]
                return f"No player with track ID {track_id}. Available IDs: {', '.join(available)}"
            return str(stats)

        elif tool_name == "get_team_stats":
            name = self._normalize_video_name(tool_args["video_name"])
            if name not in self.session_store:
                return f"No analysis found for '{name}'."
            return str(self.session_store[name].get("team_stats", {}))

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

        # Build messages: system prompt + conversation history
        messages = [self.system_prompt] + self.conversation_history

        # First LLM call — might return text or a tool call
        response = self.llm_with_tools.invoke(messages)

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            # LLM decided to call a tool
            tool_call = response.tool_calls[0]  # Handle first tool call
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"[Orchestrator] Tool call: {tool_name}({tool_args})")

            # Execute the tool
            tool_result = self._execute_tool(tool_name, tool_args)

            # Add the AI's tool call and the result to history
            self.conversation_history.append(response)
            self.conversation_history.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

            # Second LLM call — now with the tool result, generate final response
            messages = [self.system_prompt] + self.conversation_history
            final_response = self.llm.invoke(messages)

            self.conversation_history.append(final_response)
            return final_response.content

        else:
            # LLM just responded with text (no tool call needed)
            self.conversation_history.append(response)
            return response.content