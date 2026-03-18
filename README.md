# ⚽ Football Match Analyzer — Multi-Agent AI System

A multi-agent AI system that watches football match footage and automatically generates professional-grade performance analytics — the same kind of data that companies like **Opta** and **StatsBomb** charge clubs thousands for.

No cloud GPUs. No enterprise APIs. Just Python, a consumer GPU, and a free-tier LLM.

**Send a video to a Telegram bot → get a full match analysis → ask follow-up questions in natural language.**

---

## What it does

Feed the system a broadcast match video and it will:

- **Detect and track** every player across thousands of frames using YOLOv8s + ByteTrack
- **Classify teams** automatically using KMeans clustering on jersey colors (HSV)
- **Detect events** — passes, tackles, interceptions — using rule-based logic
- **Compute physical stats** — distance covered, top speed, sprint count, heatmap positions per player
- **Generate a match report** using Groq LLM (Llama 3.3 70B) that reads like a real analyst wrote it
- **Answer follow-up questions** via Telegram: "Who was the fastest player?", "Compare the teams", "How did player 96 perform?"

---

## Architecture

Six agents orchestrated by LangGraph, controlled through a Telegram bot powered by LangChain tool calling.

```
MP4 Video
    │
    ▼
┌──────────────────────────────────┐
│  Vision Agent                    │  YOLOv8s + ByteTrack
│  Detect & track every player     │  → 73,485 detections
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Ball Interpolation              │  Linear interpolation
│  Fill ball detection gaps        │  → 47.2% ball coverage
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Team Classifier                 │  KMeans on HSV jersey colors
│  Auto-assign Team A / Team B     │  → 100 sample frames
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Events Agent                    │  Rule-based detection
│  Passes, tackles, interceptions  │  → 20 passes, 5 tackles, 13 int.
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Analytics Agent                 │  Distance, speed, sprints
│  Per-player & per-team stats     │  → 228 players tracked
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Reporting Agent                 │  Groq LLM (Llama 3.3 70B)
│  Natural language match report   │  → Professional-grade report
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Orchestrator (LangGraph)        │  LLM tool calling
│  + Telegram Bot                  │  Session store + token-efficient
└──────────────────────────────────┘
```

**Key design decisions:**
- Agents 1–3 are **deterministic Python** — no LLM, no randomness, no API costs
- Only the Reporting Agent and Orchestrator use an LLM — keeping total API calls to ~5–10 per match
- Per-video caching — YOLO runs once per video, second analysis is instant
- Token-efficient architecture — LLM only sees short summaries, tools fetch specific data on demand

---

## Tech Stack

| Component | Technology |
|---|---|
| Player detection | YOLOv8s (Ultralytics) |
| Player tracking | ByteTrack (via supervision) |
| Team classification | KMeans clustering (scikit-learn) |
| Event detection | Rule-based Python |
| Physical stats | Custom engine (distance, speed, sprints) |
| Match reports | Groq API (Llama 3.3 70B) |
| Agent orchestration | LangGraph |
| Orchestrator | LangChain tool calling |
| Interface | Telegram bot (python-telegram-bot) |
| Language | Python 3.12+ |

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── vision/          # YOLOv8s + ByteTrack detection & tracking
│   │   ├── events/          # Rule-based event detection (passes, tackles)
│   │   ├── analytics/       # Per-player & per-team stats aggregation
│   │   ├── reporting/       # Groq LLM match report generation
│   │   └── orchestrator/    # LLM tool calling + session store
│   ├── graph/
│   │   ├── nodes.py         # LangGraph node wrappers for each agent
│   │   └── match_graph.py   # StateGraph definition (wires all agents)
│   ├── services/
│   │   └── telegram_bot.py  # Telegram interface
│   ├── state/
│   │   └── match_state.py   # Shared LangGraph state (TypedDict)
│   └── utils/
│       ├── ball_interpolation.py  # Fill ball detection gaps
│       └── team_classifier.py     # KMeans jersey color clustering
├── data/                    # Match videos + cached results
├── config.py                # Central configuration
├── run_pipeline.py          # Standalone pipeline runner (no Telegram)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option 1: Docker Hub (fastest)
```bash
# 1. Pull the image
docker pull samymetref/football-match-analyzer:latest

# 2. Create a .env file with your keys
echo "GROQ_API_KEY=your_key_here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env
echo "LLM_PROVIDER=groq" >> .env
echo "TELEGRAM_BOT_TOKEN=your_token_here" >> .env

# 3. Run it
docker run --env-file .env -v $(pwd)/data:/app/data samymetref/football-match-analyzer:latest
```

### Option 2: Docker (build from source)
```bash
# 1. Clone the repo
git clone https://github.com/predaaaaasaaaaaaa/mult-agents-football-match-analyzer.git
cd mult-agents-football-match-analyzer

# 2. Create your .env file
cp .env.example .env
# Edit .env with your Groq API key and Telegram bot token

# 3. Place a match video in data/
# (broadcast camera angle, 1080p, MP4 format)

# 4. Build and run
docker-compose up --build
```

### Option 3: Manual setup

```bash
# 1. Clone the repo
git clone https://github.com/predaaaaasaaaaaaa/mult-agents-football-match-analyzer.git
cd mult-agents-football-match-analyzer

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For GPU support (recommended), install PyTorch with CUDA:
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 5. Create your .env file
cp .env.example .env
# Edit .env with your Groq API key and Telegram bot token

# 6. Place a match video in data/
# (broadcast camera angle, 1080p, MP4 format)

# 7. Run the Telegram bot
python -m src.services.telegram_bot
```

### Getting your API keys

**Groq API key** (free):
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up and create an API key
3. Add it to `.env` as `GROQ_API_KEY`

**Telegram bot token** (free):
1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the token and add it to `.env` as `TELEGRAM_BOT_TOKEN`

---

## Usage

### Via Telegram bot

Start the bot, then open Telegram and chat:

```
You: Analyze the match clip
Bot: The analysis is complete. 228 players tracked, 20 passes, 5 tackles...

You: Show me the match report
Bot: Team A dominated possession with 68.5%...

You: How did player 96 perform?
Bot: 📊 PLAYER #96 — Team A
     Distance covered: 5267.2px
     Top speed: 1116.4px/s
     Sprints: 328...

You: Compare the two teams
Bot: ⚽ TEAM COMPARISON
     Possession    68.5%    31.5%
     Passes           18        2
     ...
```

### Via standalone pipeline (no Telegram)

```bash
python run_pipeline.py
```

This runs the full pipeline and prints results to the terminal. Useful for testing or scripting.

---

## How it works — Token-Efficient Architecture

Most AI agent frameworks load everything into context on every message — memory files, conversation history, full datasets. This burns tokens fast and makes local LLM deployment impractical.

This project uses a different pattern:

1. **Data lives externally** — all analysis results are stored in a session store (Python dict), not in the LLM context
2. **LLM sees summaries only** — tool results are truncated to ~500 characters in conversation history
3. **Tools fetch on demand** — when the LLM needs specific data (player stats, team comparison), it calls a tool that fetches just that slice
4. **History auto-trims** — conversation history is trimmed by estimated token budget before each call

**Result:** ~2K tokens per message instead of 10K+. Full multi-turn conversations on Groq's free tier (12K TPM). Could run on a local 8B model with 4GB VRAM.

---

## Video Requirements

For best results, use broadcast match footage with:
- **Camera angle:** Wide tactical view (the standard TV broadcast angle)
- **Resolution:** 1080p recommended
- **Format:** MP4
- **Duration:** Any length (longer = more processing time)
- **Note:** Close-ups, replays, and camera switches will reduce detection quality

Telegram bots can only download files under 20MB. For full match videos, place them in the `data/` folder manually and tell the bot: `"Analyze data/your_video.mp4"`

---

## Caching

The system caches expensive computations per video:
- `data/cache/{video_name}_tracking.json` — YOLO + ByteTrack detections
- `data/cache/{video_name}_teams.json` — Team classification results

First analysis of a video runs the full CV pipeline (~2 min for 5000 frames on an RTX 2050). Every subsequent analysis loads from cache instantly.

To re-run detection on a video, delete its cache files.

---

## Known Limitations

- **Ball detection:** ~47% coverage after interpolation. YOLO detects the ball inconsistently on broadcast footage. V2 will fine-tune on a football-specific dataset.
- **Track ID fragmentation:** ByteTrack assigns new IDs when players leave and re-enter the camera view. One real player may have multiple track IDs. V2 will add per-track majority voting.
- **Pixel distances:** All distances and speeds are in pixels, not meters. Relative comparisons between players are valid. V2 will add pitch homography for real-world units.
- **Event detection:** Currently detects passes, tackles, and interceptions. Shot detection, dribbles, and fouls are planned for V2.
- **Single user:** The bot runs one analysis at a time. V2 will add Redis for job queuing and multi-user support.

---

## V2 Roadmap

- [ ] Redis for persistence + job queue + multi-user support
- [ ] Fine-tune YOLO on Roboflow football dataset
- [ ] Per-track majority voting for team classification
- [ ] Groq → local model option (zero cloud cost)
- [ ] Shot detection, dribble detection, foul detection
- [ ] Pixel → meter conversion via pitch homography
- [ ] Real-time live match support
- [ ] Web dashboard alternative to Telegram

---

## Contributing

This project is open source. If you want to contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/shot-detection`)
3. Commit your changes
4. Push and open a PR

Priority areas: ball detection improvement, new event types, pitch homography, Redis integration.

---

## License

MIT

---

## Author

**Samy Metref** — [GitHub](https://github.com/predaaaaasaaaaaaa) · [LinkedIn](www.linkedin.com/in/samy-metref-77744133b)

Built from scratch. If you're a football club that needs affordable analytics — or a developer interested in token-efficient agent architecture — reach out.