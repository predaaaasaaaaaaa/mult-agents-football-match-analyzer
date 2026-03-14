"""
Analytics Agent
Aggregates detected events into per-player and per-team statistics.
Takes output from EventsAgent → produces match stats summary.
"""


class AnalyticsAgent:
    def __init__(self):
        self.player_stats = {}
        self.team_stats = {"A": self._empty_team_stats(), "B": self._empty_team_stats()}

    def _empty_player_stats(self, track_id, team):
        return {
            "track_id": track_id,
            "team": team,
            "passes_made": 0,
            "passes_received": 0,
            "turnovers_lost": 0,
            "turnovers_won": 0,
            "tackles_made": 0,
            "tackles_suffered": 0,
            "interceptions_made": 0,
            "interceptions_suffered": 0,
            "possession_frames": 0,
        }

    def _empty_team_stats(self):
        return {
            "total_passes": 0,
            "total_turnovers_lost": 0,
            "total_turnovers_won": 0,
            "total_tackles": 0,
            "total_interceptions": 0,
            "possession_frames": 0,
        }

    def _get_player(self, track_id, team):
        """Get or create player stats entry."""
        if track_id not in self.player_stats:
            self.player_stats[track_id] = self._empty_player_stats(track_id, team)
        return self.player_stats[track_id]

    def process(self, possession_log, passes, tackles, interceptions):
        """
        Aggregate all events into stats.

        Args:
            possession_log: Frame-by-frame possession data
            passes: List of pass events
            tackles: List of tackle events
            interceptions: List of interception events
        """
        # Possession frames per player/team
        for entry in possession_log:
            tid = entry["track_id"]
            team = entry.get("team")
            if team:
                player = self._get_player(tid, team)
                player["possession_frames"] += 1
                self.team_stats[team]["possession_frames"] += 1

        # Passes
        for p in passes:
            from_team = p["from_team"]
            to_team = p["to_team"]

            passer = self._get_player(p["from_track_id"], from_team)
            passer["passes_made"] += 1

            receiver = self._get_player(p["to_track_id"], to_team)
            receiver["passes_received"] += 1

            if from_team:
                self.team_stats[from_team]["total_passes"] += 1

        # Tackles
        for t in tackles:
            # from_track_id lost the ball, to_track_id won it
            loser = self._get_player(t["from_track_id"], t["from_team"])
            loser["tackles_suffered"] += 1
            loser["turnovers_lost"] += 1

            winner = self._get_player(t["to_track_id"], t["to_team"])
            winner["tackles_made"] += 1
            winner["turnovers_won"] += 1

            if t["from_team"]:
                self.team_stats[t["from_team"]]["total_turnovers_lost"] += 1
            if t["to_team"]:
                self.team_stats[t["to_team"]]["total_tackles"] += 1
                self.team_stats[t["to_team"]]["total_turnovers_won"] += 1

        # Interceptions
        for i in interceptions:
            loser = self._get_player(i["from_track_id"], i["from_team"])
            loser["interceptions_suffered"] += 1
            loser["turnovers_lost"] += 1

            winner = self._get_player(i["to_track_id"], i["to_team"])
            winner["interceptions_made"] += 1
            winner["turnovers_won"] += 1

            if i["from_team"]:
                self.team_stats[i["from_team"]]["total_turnovers_lost"] += 1
            if i["to_team"]:
                self.team_stats[i["to_team"]]["total_interceptions"] += 1
                self.team_stats[i["to_team"]]["total_turnovers_won"] += 1

        self._print_summary()

    def _print_summary(self):
        """Print match summary."""
        total_poss = sum(t["possession_frames"] for t in self.team_stats.values())

        print("\n" + "=" * 50)
        print("MATCH STATISTICS")
        print("=" * 50)

        # Team stats
        for team_name in ["A", "B"]:
            ts = self.team_stats[team_name]
            poss_pct = (ts["possession_frames"] / total_poss * 100) if total_poss > 0 else 0

            print(f"\n--- Team {team_name} ---")
            print(f"  Possession: {poss_pct:.1f}%")
            print(f"  Passes: {ts['total_passes']}")
            print(f"  Turnovers lost: {ts['total_turnovers_lost']}")
            print(f"  Turnovers won: {ts['total_turnovers_won']}")
            print(f"  Tackles won: {ts['total_tackles']}")
            print(f"  Interceptions: {ts['total_interceptions']}")

        # Top players
        print(f"\n--- Top Players (by involvement) ---")
        sorted_players = sorted(
            self.player_stats.values(),
            key=lambda p: p["passes_made"] + p["turnovers_won"] + p["tackles_made"],
            reverse=True,
        )

        for p in sorted_players[:10]:
            print(
                f"  Track #{p['track_id']} (Team {p['team']}): "
                f"{p['passes_made']} passes, "
                f"{p['tackles_made']} tackles, "
                f"{p['interceptions_made']} interceptions, "
                f"{p['possession_frames']} frames on ball"
            )