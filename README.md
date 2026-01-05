# veltora-supply-chain

Nine autonomous AI agents coordinating data centre supply chain decisions.

The core problem: a hyperscale data centre build involves 50,000+ components across
500+ vendors, with lead times ranging from 2 weeks to 18 months. A single missed
delivery can delay an entire project by months. Each day of delay costs ~$500K-$2M
in lost revenue for the operator.

Traditional approach: spreadsheets, phone calls, and reactive firefighting once delays
are already happening.

This system: a network of specialised AI agents that monitor the global supply chain
continuously, predict risks 60-90 days out, and coordinate procurement decisions
autonomously, acting before problems become delays.

## The 9 Agents

| Agent | Role | Key capability |
|-------|------|----------------|
| `ProcurementOptimizer` | Auto-source components, generate RFQs | Cost-carbon trade off optimisation |
| `SupplyChainPredictor` | Forecast shortages 60-90 days out | Physics-informed lead time model |
| `ProjectScheduler` | Optimise construction sequencing | Critical path + delivery coordination |
| `SustainabilityOptimizer` | Carbon footprint, ESG sourcing | Scope 3 emissions tracking |
| `VendorCoordinator` | Manage 500+ supplier relationships | Relationship scoring, term negotiation |
| `QualityAssurance` | Monitor component specs | Deviation flagging, hold/release decisions |
| `LogisticsCoordinator` | Route optimisation, consolidation | Multi modal shipping, customs prediction |
| `RiskMitigation` | Single points of failure, backup suppliers | Graph based dependency analysis |
| `CostOptimizer` | Continuous price monitoring | Alternative sourcing, bulk buy signals |

All 9 agents share a common memory layer and are orchestrated by a coordinator that
resolves conflicts and makes final decisions when agents disagree.

## Architecture

```
                    ┌──────────────────────────┐
                    │   Supply Chain           │
                    │   Coordinator            │
                    │   (conflict resolution,  │
                    │    final decisions)      │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼──────┐     ┌──────────▼──────┐     ┌──────────▼──────┐
│  Prediction  │     │   Procurement   │     │  Risk & Quality  │
│  cluster     │     │   cluster       │     │  cluster         │
│              │     │                 │     │                  │
│ - Predictor  │     │ - Procurement   │     │ - RiskMitigation │
│ - Scheduler  │     │   Optimizer     │     │ - QA Agent       │
│ - Logistics  │     │ - Cost Optim.   │     │ - Sustainability │
└───────┬──────┘     └──────────┬──────┘     └──────────┬──────┘
        │                        │                        │
        └────────────────────────▼────────────────────────┘
                        ┌────────────────┐
                        │ Shared Memory  │
                        │ (Redis + graph │
                        │  component DB) │
                        └────────────────┘
```

Agents within a cluster run in parallel. Clusters run sequentially:
prediction first (establish risk landscape), then procurement (act on it),
then risk/quality (validate decisions). The coordinator arbitrates conflicts.

## Physics informed risk model

The `SupplyChainPredictor` uses a physics informed lead time model rather than
pure ML. The intuition: manufacturing lead times are not arbitrary. They are
constrained by physical production rates, transport physics and queue dynamics.

```
lead_time_total = t_queue + t_manufacture + t_test + t_transport + t_customs

t_queue        ~ capacity utilisation curve (M/M/1 queue approximation)
t_manufacture  ~ lot_size / production_rate, modulated by yield loss
t_transport    ~ distance / mode_speed, with weather/port congestion factors
t_customs      ~ stochastic, estimated from historical clearing times by HS code
```

Pure ML models can predict plausible but physically impossible outcomes (e.g.
air freight from Shenzhen in 6 hours). The physics constraints prevent this.
On the held out test set, the physics informed model has 23% lower MAE than
a pure gradient boosting baseline.

## Benchmark results

Tested against a rule based baseline (current industry standard: fixed reorder
points, manual escalation, no predictive capability) on 18 months of simulated
procurement data generated from real project templates.

```
Metric                          Rule based    This system    Improvement
─────────────────────────────────────────────────────────────────────────
Delay prediction accuracy
  (60-day horizon)              31%           84%            +171%
  (90-day horizon)              18%           71%            +294%

Mean procurement lead time      67 days       41 days        -39%
Emergency order rate            23%           4%             -83%
Component stockout events       11/project    2/project      -82%
Cost vs budget                  +18%          +3%            -83%
Carbon per $1M spend (tCO2e)    142           98             -31%

Agent coordination latency      N/A           340ms median
Decisions made autonomously     0%            68%
```

Simulation uses 500 project runs with randomised demand, vendor failures,
geopolitical disruptions, and weather events. See `benchmarks/` for methodology.

## Running it

```bash
pip install -r requirements.txt

# run the simulation benchmark
python -m benchmarks.run --projects 50 --horizon 90

# run the agent system against a single project scenario
python -m coordinator.orchestrator --project examples/hyperscale_build.json

# train the RL procurement policy
python -m rl.train --timesteps 1000000 --n-envs 8
```

## Key design decisions

**Why 9 separate agents instead of one?**
Each domain (logistics, sustainability, quality) requires different tools and
context. A single agent trying to reason about all of them simultaneously runs
into context window limits and attention dilution. Specialised agents stay
focused and can run in parallel.

**Why physics informed instead of pure ML?**
Supply chain data is noisy and non stationary. A model trained on 2019-2022
data had no concept of COVID disruptions. Physics constraints keep predictions
grounded in what's actually possible regardless of training distribution.

**Why RL for procurement policy?**
Procurement is a sequential decision problem with delayed rewards. You don't
know if a buying decision was good until months later when the components
arrive (or don't). RL handles this naturally. Supervised learning can't because
you'd need labelled examples of optimal decisions, which don't exist.

## File layout

```
agents/               - all 9 agent implementations
  procurement.py      - ProcurementOptimizer
  predictor.py        - SupplyChainPredictor (physics informed)
  scheduler.py        - ProjectScheduler
  sustainability.py   - SustainabilityOptimizer
  vendor.py           - VendorCoordinator
  quality.py          - QualityAssurance
  logistics.py        - LogisticsCoordinator
  risk.py             - RiskMitigation
  cost.py             - CostOptimizer
coordinator/          - orchestrator that runs all 9 agents
  orchestrator.py     - main coordination loop
  conflict.py         - conflict resolution between agents
simulation/           - supply chain simulation environment
  env.py              - Gymnasium environment for RL training
  components.py       - component catalogue and lead time model
  disruptions.py      - random disruption generator
  physics.py          - physics informed lead time calculator
rl/                   - reinforcement learning for procurement policy
  train.py            - PPO training script
  policy.py           - policy network definition
benchmarks/           - benchmark harness and results
  run.py              - benchmark runner
  baseline.py         - rule based baseline
  results.json        - latest benchmark results
```
