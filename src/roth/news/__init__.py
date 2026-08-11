"""Live news monitoring for a fixed watchlist of large-cap equities.

This subsystem is deliberately separate from the research harness. The harness
is on-demand, offline, and reproducible; this is a long-running process that
talks to the public internet and tells you things while the market is open.
They share the trading calendar and nothing else.
"""
