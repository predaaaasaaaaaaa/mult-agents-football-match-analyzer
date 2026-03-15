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
| Avg Distance (px) | X | Y |
| Max Speed (px/s) | X | Y |
| Total Sprints | X | Y |

## Tactical Analysis
One paragraph (3-4 sentences) analyzing what the numbers reveal about each team's approach — pressing, dominance, defensive solidity.

## Physical Analysis
One paragraph (2-3 sentences) analyzing the physical output — which team worked harder, who covered the most ground, sprint patterns.

## Standout Performers
Bullet list of 3-5 players with a one-line note on what they did well. Include physical stats where relevant.

## Data Notes
One sentence acknowledging any data limitations.

RULES:
- Be concise. Total report should be under 300 words.
- Write like a Sky Sports or BBC Sport pundit — sharp, insightful, no filler.
- Focus on PATTERNS, not just numbers.
- Do NOT repeat raw numbers the reader can see in the table.
- All distances/speeds are in pixels — focus on relative comparisons between players, not absolute values."""

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
            lines.append(f"  Avg distance covered (px): {ts['avg_distance_px']}")
            lines.append(f"  Max speed (px/s): {ts['avg_top_speed_px_s']}")
            lines.append(f"  Total sprints: {ts['total_sprints']}")
            lines.append("")

        # Top players by physical output
        lines.append("Top Players (by distance covered):")
        sorted_physical = sorted(
            analytics_agent.player_stats.values(),
            key=lambda p: p["distance_px"],
            reverse=True,
        )

        for p in sorted_physical[:10]:
            lines.append(
                f"  Player #{p['track_id']} (Team {p['team']}): "
                f"distance={p['distance_px']}px, "
                f"top_speed={p['top_speed_px_s']}px/s, "
                f"sprints={p['sprint_count']}, "
                f"{p['passes_made']} passes, "
                f"{p['possession_frames']} frames on ball"
            )

        # Context
        lines.append("")
        lines.append("NOTE: This data comes from computer vision analysis of broadcast footage.")
        lines.append("All distances/speeds are in pixels (not meters) — relative comparisons between players are valid.")
        lines.append(f"Ball was detected/interpolated on ~47% of frames, so actual event counts are likely higher.")
        lines.append(f"Track IDs change when players leave/re-enter camera view, so one real player may have multiple track IDs.")

        return "\n".join(lines)