# FPL Analytics Assistant 🏆

> A machine-learning powered Fantasy Premier League assistant that replaces gut-feel with gradient boosting.

Built and battle-tested over 7 seasons — top **30k in India**, top **1.5M globally** this season.

---

## What it does

| Feature | Details |
|---|---|
| **Live data ingestion** | Pulls all player stats, fixtures, and your squad directly from the FPL API |
| **Custom FDR** | Replaces FPL's often-misleading difficulty ratings with a tiered model (`SUPER_EASY` → `SUPER_HARD`) |
| **Position-specific analysis** | Separate metrics for GK / DEF / MID / FWD (saves/90, xGI%, xG conversion, etc.) |
| **ML recommendations** | `GradientBoostingRegressor` trained on 20 FPL features predicts best transfer targets |
| **Fixture swing scoring** | Identifies players whose upcoming fixtures are meaningfully better or worse |
| **Transfer candidate flags** | Severity-scored list of players to consider moving on |
| **Excel export** | Formatted `.xlsx` with colour-coded performance + transfer sheets |
| **LLM chat layer** | Optional OpenAI integration for conversational analysis and per-player sentiment |

---

## Architecture

```
FPL API
   ├── bootstrap-static  →  FPLDataIngestion  →  players_df / teams_df
   ├── entry/{id}/event  →  FPLDataIngestion  →  team_data (GW history)
   └── fixtures/         →  FixtureDifficultyAnalyzer

FPLTrendAnalyzerEnhanced
   ├── Position-specific stats (GK/DEF/MID/FWD)
   └── Fixture swing overlay

FPLRecommendationEngine
   ├── GradientBoostingRegressor (20 features)
   └── Optional: ChatGPTIntegration (sentiment multiplier 0.7 – 1.3)

Output
   ├── fpl_analysis.xlsx
   ├── gameweek_history.csv
   └── Interactive chat (if OPENAI_API_KEY set)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export FPL_TEAM_ID=your_team_id        # Find in the FPL URL: fantasy.premierleague.com/entry/XXXXXXX/
export OPENAI_API_KEY=sk-...           # Optional — enables chat + sentiment features
```

### 3. Run

```bash
python fpl_assistant.py
```

Output files are written to `./output/`.

---

## Finding your Team ID

Go to [fantasy.premierleague.com](https://fantasy.premierleague.com), open **Points** or **Transfer** page, and look at the URL:

```
https://fantasy.premierleague.com/entry/2115807/event/35
                                         ^^^^^^^
                                         This is your Team ID
```

---

## Metrics explained

### Fixture Difficulty Rating (custom)

The official FPL FDR updates slowly and ignores current form. This model classifies every fixture using:

1. **Base difficulty** — determined by league position tiers:

   | Scenario | Difficulty |
   |---|---|
   | Same tier as opponent | `EASY` |
   | Opponent in top 5, you're not | `HARD` |
   | Opponent in bottom 5, you're not | `EASY` |
   | All other cases | `MODERATE` |

2. **Form escalation** — if the opponent has been in the top 5 in all of their last 5 games:
   - `HARD` → `SUPER_HARD`
   - `EASY` + consistently bottom 5 opponent → `SUPER_EASY`

**Fixture swing** = `upcoming_avg_difficulty - past_avg_difficulty`. Negative = improving run.

### Transfer candidate flags

| Position | Flag triggered when… |
|---|---|
| Goalkeeper | saves/90 < 2.0 or clean sheet rate < 20% |
| Defender | goals conceded > xGC by more than 2 |
| Midfielder | xGI underperformance > 20% |
| Forward | xG conversion rate < 60% |
| All | minutes < 500, PPG < 2.5, form < 2.0, fixture swing > 1.5 |

Severity score accumulates across flags; higher = more urgent transfer.

---

## ML model

- **Algorithm:** `GradientBoostingRegressor` (sklearn) — handles non-linear feature interactions well
- **Target:** `total_points` (current season accumulated)
- **Features (20):** form, PPG, total_points, xG, xA, xGI, xGC, goals, assists, clean sheets, bonus, BPS, influence, creativity, threat, ICT index, minutes, cost, starts, saves
- **Sentiment adjustment:** if OpenAI is configured, each candidate's ML score is multiplied by a sentiment weight (0.7 – 1.3) derived from GPT analysis of their stats + fixture difficulty

> **Note:** The model is trained on the current season's accumulated data — it learns which combination of stats correlates with high total points. It is NOT a future-points predictor trained on historical seasons; treat outputs as a ranked shortlist, not hard projections.

---

## Project structure

```
fpl-analytics/
├── fpl_assistant.py      # Main module (all classes + CLI entry point)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
├── LICENSE               # MIT
└── output/               # Generated at runtime (Excel, CSV)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data wrangling |
| `numpy` | Numeric operations |
| `scikit-learn` | GradientBoosting + StandardScaler |
| `requests` | FPL API calls |
| `openpyxl` | Excel export |
| `openai` *(optional)* | Chat + sentiment features |

---

## Licence

MIT — free to use, fork, and build on. Attribution appreciated but not required.

---

## Contributing

PRs welcome. Particularly interested in:
- Historical season training data integration
- Better minutes probability modelling
- Web UI wrapper (Streamlit / Gradio)
- Better home/away FDR splits

Open an issue first for substantial changes.
