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


def test_failures_outside_the_window_are_forgotten():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=clock, window_s=5)
    b.record_failure()
    clock.now = 5
    b.record_failure()  # the first failure has aged out: a new window starts here
    assert b.state is BreakerState.CLOSED
    clock.now = 9
    b.record_failure()
    assert b.state is BreakerState.OPEN


def test_probe_lease_expires_and_release_hands_it_back():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, cooldown_s=5, clock=clock, probe_lease_s=3)
    b.record_failure()
    clock.now = 5
    assert b.allow()
    assert not b.allow()
    clock.now = 8  # the probe never reported back: its lease is up
    assert b.allow()
    b.release()
    assert b.allow()


def test_late_failure_while_open_does_not_extend_cooldown():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, cooldown_s=10, clock=clock)
    b.record_failure()
    clock.now = 6
    b.record_failure()  # from a call that started before the circuit opened
    clock.now = 10
    assert b.state is BreakerState.HALF_OPEN


def test_transitions_are_reported():
    clock = Clock()
    seen: list[tuple[str, str]] = []
    b = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=1,
        clock=clock,
        on_transition=lambda o, n: seen.append((o, n)),
    )
    b.record_failure()
    clock.now = 1
    assert b.allow()
    b.record_success()
    assert seen == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]
