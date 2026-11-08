"""dynamic-hot-pricing-controller.

Dynamic tolling for a freeway HOT lane: a two-facility cell transmission model with weaving
capacity loss at the ingress, logit lane choice over a value-of-time distribution, four pricing
controllers (time-of-day, density lookup, PID, tabular Q-learning) and a Page-Hinkley drift
detector on the price-response model.
"""

__version__ = "1.0.0"
