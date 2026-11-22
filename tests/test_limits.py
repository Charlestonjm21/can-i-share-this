from datetime import date, timedelta

from cish import limits


def test_daily_counter_stops_at_the_cap():
    c = limits.DailyCounter()
    d = date(2026, 1, 1)
    assert [c.try_take(2, d) for _ in range(3)] == [True, True, False]


def test_daily_counter_resets_the_next_day():
    c = limits.DailyCounter()
    d = date(2026, 1, 1)
    c.try_take(1, d)
    assert not c.try_take(1, d)
    assert c.try_take(1, d + timedelta(days=1))


def test_zero_means_unlimited():
    c = limits.DailyCounter()
    assert all(c.try_take(0, date(2026, 1, 1)) for _ in range(1000))


def test_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("CISH_MAX_QUESTIONS", raising=False)
    assert limits.max_questions() == 20
    monkeypatch.setenv("CISH_MAX_QUESTIONS", "5")
    assert limits.max_questions() == 5
    monkeypatch.setenv("CISH_MAX_QUESTIONS", "not a number")
    assert limits.max_questions() == 20
