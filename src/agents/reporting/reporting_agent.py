"""
Reporting Agent
Takes match statistics from Analytics Agent and generates
a natural language match report using Groq LLM (Llama 3.3 70B).
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


class ReportingAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        self.system_prompt = """You are an expert football match analyst writing a post-match report.

FORMAT YOUR REPORT EXACTLY LIKE THIS:

## Match Overview
One short paragraph (2-3 sentences) summarizing the overall story of the match.

## Key Statistics
| Stat | Team A | Team B |
|------|--------|--------|
| Possession | X% | Y% |
| Passes | X | Y |
| Tackles Won | X | Y |
| Interceptions | X | Y |
| Turnovers Lost | X | Y |

## Tactical Analysis
One paragraph (3-4 sentences) analyzing what the numbers reveal about each team's approach — pressing, dominance, defensive solidity.

## Standout Performers
Bullet list of 3-5 players with a one-line note on what they did well. Use track IDs (e.g., "Player #96").

## Data Notes
One sentence acknowledging any data limitations.

RULES:
- Be concise. Total report should be under 250 words.
- Write like a Sky Sports or BBC Sport pundit — sharp, insightful, no filler.
- Focus on PATTERNS, not just numbers.
- Do NOT repeat raw numbers the reader can see in the table."""

    def generate_report(self, analytics_agent):
        """
        Generate a match report from the Analytics Agent's stats.

        Args:
            analytics_agent: AnalyticsAgent instance with processed stats
        """
        # Build stats summary string for the LLM
        stats_text = self._build_stats_text(analytics_agent)

        print("[ReportingAgent] Sending stats to Groq LLM...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Here are the match statistics:\n\n{stats_text}\n\nWrite a post-match analysis report."),
        ]

        response = self.llm.invoke(messages)
        report = response.content

        print("\n" + "=" * 50)
        print("MATCH REPORT")
        print("=" * 50)
        print(report)

        return report

    def _build_stats_text(self, analytics_agent):
        """Convert analytics data into a text summary for the LLM."""
        lines = []

        total_poss = sum(
            t["possession_frames"] for t in analytics_agent.team_stats.values()
        )

        # Team stats
        for team_name in ["A", "B"]:
            ts = analytics_agent.team_stats[team_name]
            poss_pct = (ts["possession_frames"] / total_poss * 100) if total_poss > 0 else 0

            lines.append(f"Team {team_name}:")
            lines.append(f"  Possession: {poss_pct:.1f}%")
            lines.append(f"  Passes completed: {ts['total_passes']}")
            lines.append(f"  Turnovers lost: {ts['total_turnovers_lost']}")
            lines.append(f"  Turnovers won: {ts['total_turnovers_won']}")
            lines.append(f"  Tackles won: {ts['total_tackles']}")
            lines.append(f"  Interceptions: {ts['total_interceptions']}")
            lines.append("")

        # Top players
        lines.append("Top Players (by involvement):")
        sorted_players = sorted(
            analytics_agent.player_stats.values(),
            key=lambda p: p["passes_made"] + p["turnovers_won"] + p["tackles_made"] + p["possession_frames"],
            reverse=True,
        )

        for p in sorted_players[:10]:
            lines.append(
                f"  Player #{p['track_id']} (Team {p['team']}): "
                f"{p['passes_made']} passes made, "
                f"{p['passes_received']} received, "
                f"{p['tackles_made']} tackles, "
                f"{p['interceptions_made']} interceptions, "
                f"{p['possession_frames']} frames on ball"
            )

        # Context
        lines.append("")
        lines.append("NOTE: This data comes from computer vision analysis of broadcast footage.")
        lines.append(f"Ball was detected/interpolated on ~47% of frames, so actual event counts are likely higher.")
        lines.append(f"Track IDs change when players leave/re-enter camera view, so one real player may have multiple track IDs.")

        return "\n".join(lines)