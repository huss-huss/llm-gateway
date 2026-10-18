from __future__ import annotations

from llm_gateway.breaker import BreakerState, CircuitBreaker


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_opens_after_threshold_and_recovers_via_half_open():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=3, cooldown_s=10, clock=clock)
    for _ in range(2):
        b.record_failure()
    assert b.state is BreakerState.CLOSED and b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN
    assert not b.allow()

    clock.now = 10
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow()  # the single probe
    assert not b.allow()  # no second concurrent probe
    b.record_success()
    assert b.state is BreakerState.CLOSED and b.allow()


def test_failed_probe_reopens():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, cooldown_s=5, clock=clock)
    b.record_failure()
    clock.now = 5
    assert b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN
    clock.now = 9
    assert not b.allow()
    clock.now = 10
    assert b.allow()


def test_success_resets_failure_count():
    b = CircuitBreaker(failure_threshold=2)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state is BreakerState.CLOSED
