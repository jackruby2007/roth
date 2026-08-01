"""Strategies.

Adding a hypothesis means adding one class here and nothing else.
"""

from roth.strategies.reference import MondayCallReference

STRATEGIES = {
    MondayCallReference.name: MondayCallReference,
}

__all__ = ["STRATEGIES", "MondayCallReference"]
