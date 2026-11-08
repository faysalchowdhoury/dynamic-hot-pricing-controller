My reinforcement-learning toll controller learned to break the express lane on purpose. It was doing exactly what I told it to.

I built a dynamic-pricing testbed for a freeway HOT lane: a cell transmission model with a GP bottleneck, ingress weaving and an egress merge; logit lane choice over a value-of-time distribution; and four controllers evaluated on the same 10 held-out demand days.

Peak-period results:
🔹 Time-of-day ($1/$4): HOT ≥ 45 mph 95% of the time, GP lanes at 45.9 mph
🔹 MnPASS-style density lookup: 93%, with bang-bang toll oscillation
🔹 Tuned PID: 99% reliability, GP at 61.7 mph, average toll $2.38 (40% lower)
🔹 Q-learning: 97%, but over-priced and highly variable day to day

The RL story took three versions:
1️⃣ State without the current toll → 50% reliability. Incremental actions mean different things at $1 and $9.
2️⃣ Added the toll, breakdown penalty of 5 → 37%. It learned to under-price and trade breakdowns for throughput, because that's what the reward said.
3️⃣ Penalty aligned with the evaluation standard → 97%, still behind a two-gain PID.

Lessons:
• When the target state is clear (density just under critical), classical feedback is hard to beat.
• Revenue and operations pull in different directions: the flat $4 plan earned the most and moved traffic the worst.
• Monitor the model behind the price. A Page-Hinkley detector on predicted-vs-observed buy-in caught a 40% drop in willingness to pay on 8 of 10 days, with no false alarms.

Caveat: synthetic corridor with illustrative choice parameters; first-order traffic model.

Repo: [link]

Managed-lane folks: how do you tune dynamic pricing in practice, lookup tables or feedback?

#ManagedLanes #CongestionPricing #TrafficEngineering #ReinforcementLearning #ControlSystems #ITS #Python
