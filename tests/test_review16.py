"""Regression tests added after the review of Version 16.

The review found that the quiet half minute of the service component was
silent by construction, because its two messages belong to one request, so
that the printed bound was a calculation under a model that had not been
named; that an empty record was called verified for its whole nominal span
with no evidence that the recorder had been live; and that the first
positive count was left with a symmetric interval that reaches below zero.
It suggested a model in which messages come in pairs, a worked case on a
search for an empty window, and a scope note on a judge whose verdicts go
against the reference.

Each test computes what it expects independently of the code under test: the
Poisson probability of silence for the union of the start times that leave a
message, seeded simulations of the paired process and of the search, the
lines of the trace themselves, closed forms of the exact limits for counts
of 0, 1 and 2, exact coverage by summation, published limits, the Poisson
tail in 50-digit decimal arithmetic, and the algebra of exchanged verdicts.
"""
import math
import random
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch20_judge, ch23_case
from swmetro.stats import poisson_at_most, quiet_bound, rate_limits, rate_upper

ROOT = Path(__file__).resolve().parent.parent
START = (12 * 3600 + 26 * 60) * 1000          # 12:26:00.000 in ms since midnight
SPAN = (START, START + 60_000)
COUNTER = (1, 800)
D = 2.3456                                    # the duration of request 17 in the generator, s


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def fixture():
    text = (ROOT / "data" / "app_trace.log").read_text(encoding="utf-8")
    return [trace.parse(line) for line in text.splitlines()]


def plain(n, mean):
    """P(X <= n) for a Poisson count as the plain sum of its terms, for a small mean."""
    return sum(math.exp(-mean) * mean ** i / math.factorial(i) for i in range(n + 1))


def reference(n, mean):
    """The same in 50-digit decimal arithmetic, for any mean."""
    with localcontext() as ctx:
        ctx.prec = 50
        mean = Decimal(mean)
        term, total = (-mean).exp(), Decimal(0)
        for i in range(n + 1):
            total += term
            term = term * mean / (i + 1)
        return float(total)


class PairedMessages(unittest.TestCase):
    def test_the_two_service_messages_belong_to_one_request(self):
        svc = [((ms - START) / 1000, text) for _, ms, comp, text in fixture() if comp == "svc"]
        self.assertEqual(svc, [(10.123, "request 17 start"), (12.469, "request 17 end")])
        source = (ROOT / "data" / "make_trace.py").read_text(encoding="utf-8")
        self.assertIn('out.append((10.1234, 2204, "svc", "request 17 start"))', source)
        self.assertIn('out.append((10.1234 + 2.3456, 2204, "svc", "request 17 end"))', source)
        writes = [x for x in source.splitlines() if "out.append(" in x and '"svc"' in x]
        self.assertEqual(len(writes), 2)                           # nothing else writes for the component
        # The Time Delta that the listing measures is the duration to the millisecond.
        self.assertEqual(trace.time_delta(fixture()).value, 2346)

    def test_the_bound_of_the_review(self):
        # No request may start from d before the window to its end: an exposure of 30 + d for the
        # requests, and two messages to a request.
        self.assertEqual(f"{2 * math.log(20) / (30 + D):.5f}", "0.18523")
        self.assertEqual(f"{silence.paired_bound([(0, 30)], D):.5f}", "0.18523")
        self.assertEqual(f"{silence.paired_bound([(0, 30)], 2.346):.5f}", "0.18523")
        self.assertAlmostEqual(silence.paired_bound([(0, 30)], D), 2 * quiet_bound(30 + D), places=15)
        # At that message rate the window is silent 5 % of the time, and at the bound for single
        # messages, 19.9 %.
        for rate, want in ((silence.paired_bound([(0, 30)], D), "5.0"), (quiet_bound(30), "19.9")):
            self.assertEqual(f"{100 * math.exp(-rate / 2 * (30 + D)):.1f}", want)
        # (Since Version 18 the bound takes the verified stretches, here the one window.)
        # Without a duration the pairs fall together, and the bound is twice the single one.
        self.assertAlmostEqual(silence.paired_bound([(0, 30)], 0), 2 * quiet_bound(30), places=15)

    def test_the_union_of_the_start_times(self):
        # By brute force on a grid of start times: which leave a start or an end in [0, 30).
        step = 0.0004
        starts = [-5 + step * i for i in range(int(40 / step))]
        hit = [s for s in starts if 0 <= s < 30 or 0 <= s + D < 30]
        self.assertAlmostEqual(min(hit), -D, delta=step)
        self.assertAlmostEqual(max(hit), 30, delta=step)
        self.assertAlmostEqual(len(hit) * step, 30 + D, delta=2 * step)        # one interval, no hole
        ends = [s for s in starts if 0 <= s + D < 30]
        self.assertAlmostEqual(min(ends), -D, delta=step)
        self.assertAlmostEqual(max(ends), 30 - D, delta=step)

    def test_a_simulated_paired_process_is_silent_that_often(self):
        rng = random.Random(1617)
        for rate, want in ((silence.paired_bound([(0, 30)], D), 0.05), (quiet_bound(30), 0.1989)):
            trials, silent = 30_000, 0
            for _ in range(trials):
                t, seen = -20 + rng.expovariate(rate / 2), False      # requests from 20 s before the window
                while t < 30 and not seen:
                    seen = 0 <= t < 30 or 0 <= t + D < 30
                    t += rng.expovariate(rate / 2)
                silent += not seen
            self.assertAlmostEqual(silent / trials, want, delta=0.008, msg=rate)

    def test_the_figure_is_drawn_from_the_measured_duration(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        # (Since Version 18 the figure has a second panel and takes the intervals from starts().)
        for piece in ("from examples import ch16_trace_measurements as c16", "c16.time_delta(rows).value / 1000",
                      '("either", c16s.starts(whole, d))', 'save(fig, "fig-16-1-request-starts.png")'):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


class VerifiedByEvidence(unittest.TestCase):
    def test_the_span_alone_verifies_nothing(self):
        # Version 16 returned 60 s for an empty record, whatever had become of its messages.
        self.assertEqual(trace.verified([], 0, 60, SPAN), 0.0)
        self.assertEqual(trace.verified([], 0, 60, SPAN, None), 0.0)
        # A counter that did not move: nothing was assigned, so nothing was lost.
        for reading in (1, 41, 800):
            self.assertEqual(trace.verified([], 0, 60, SPAN, (reading, reading)), 60.0)
            self.assertEqual(trace.verified([], 15, 35, SPAN, (reading, reading)), 20.0)
        # A counter that moved by 40 with no line to show for it: every message was dropped.
        self.assertEqual(trace.verified([], 0, 60, SPAN, (1, 41)), 0.0)
        with self.assertRaises(ValueError):
            quiet_bound(trace.verified([], 0, 60, SPAN))             # no exposure, no bound
        self.assertEqual(f"{quiet_bound(20):.5f}", "0.14979")
        self.assertEqual(f"{-math.log(0.05) / 20:.5f}", "0.14979")

    def test_lines_verify_the_time_between_them_and_readings_the_ends(self):
        rows = fixture()
        first, last = (rows[0][1] - START) / 1000, (rows[-1][1] - START) / 1000
        self.assertEqual((rows[0][0], rows[-1][0], first, last), (1, 799, 0.022, 59.531))
        self.assertEqual(trace.COUNTER, COUNTER)
        # With the readings: everything but the stretch of the loss.
        self.assertEqual(trace.verified(rows, 0, 30, SPAN, COUNTER), 27.912)
        self.assertEqual(trace.verified(rows, 30, 60, SPAN, COUNTER), 30.0)
        # Without them the time before the first line and after the last is not verified.
        self.assertAlmostEqual(trace.verified(rows, 0, 30, SPAN), 27.912 - first, places=9)
        self.assertAlmostEqual(trace.verified(rows, 30, 60, SPAN), last - 30, places=9)
        self.assertAlmostEqual(trace.verified(rows, 0, 60, SPAN), last - first - 2.088, places=9)

    def test_a_reading_shows_a_loss_at_either_end(self):
        rows = fixture()
        second, before_last = (rows[1][1] - START) / 1000, (rows[-2][1] - START) / 1000
        # The first line lost: the counter says a message came before the first that is there.
        self.assertAlmostEqual(trace.verified(rows[1:], 0, 30, SPAN, COUNTER), 27.912 - second, places=9)
        # The last line lost: the counter at the stop is two beyond the last line.
        self.assertAlmostEqual(trace.verified(rows[:-1], 30, 60, SPAN, COUNTER), before_last - 30, places=9)
        # Readings that fit the shortened record verify the ends again.
        self.assertEqual(trace.verified(rows[1:], 0, 30, SPAN, (2, 800)), 27.912)
        self.assertEqual(trace.verified(rows[:-1], 30, 60, SPAN, (1, 799)), 30.0)

    def test_the_transcripts(self):
        text = expected("ch16_silence")
        for line in ("svc, 30-60 s: 0 messages, 30 s verified", "  20 s, 100 %: 20 s,\n    at most 0.14979 msg/s at 95 %",
                     "  counter at 1, and at 1:\n    60 s verified", "  counter at 1, and at 41:\n    exposure unknown, no bound",
                     "  no counter readings:\n    exposure unknown, no bound"):
            self.assertIn(line + "\n", text)
        # Every bound in the three transcripts carries its level.
        for name in ("ch16_silence", "ch16_trace_measurements", "ch23_case"):
            lines = [x for x in expected(name).splitlines() if "at most" in x]
            self.assertTrue(lines, name)
            for line in lines:
                if line.endswith("at 97.5 %:"):               # since Version 24: the limit for a share, at its own level
                    self.assertEqual(line, "  at most 31.699 % of 10 s, at 97.5 %:")
                    continue
                if line.endswith("at 97.5 %"):                # since Version 60: the upper end by ranks, at its level
                    self.assertEqual(line, "  after, at most 185.70 ms at 97.5 %")
                    continue
                self.assertTrue(line.endswith("at 95 %"), line)


class SearchForSilence(unittest.TestCase):
    def test_the_three_rows_of_the_review(self):
        self.assertEqual(f"{quiet_bound(30):.5f}", "0.09986")
        self.assertEqual(f"{silence.searched_bound(30, 10):.5f}", "0.17584")
        self.assertEqual(f"{quiet_bound(300):.5f}", "0.00999")
        # The definitions: the probability of each observation is 5 % at its bound.
        rate = silence.searched_bound(30, 10)
        self.assertAlmostEqual(1 - (1 - math.exp(-30 * rate)) ** 10, 0.05, places=12)
        self.assertAlmostEqual(math.exp(-300 * quiet_bound(300)), 0.05, places=12)
        # At the bound for a window chosen in advance the search succeeds 40.13 % of the time.
        self.assertEqual(f"{100 * (1 - (1 - math.exp(-30 * quiet_bound(30))) ** 10):.2f}", "40.13")
        self.assertEqual(f"{100 * (1 - 0.95 ** 10):.2f}", "40.13")
        # A search of one window is a window chosen in advance, and more windows raise the bound.
        self.assertAlmostEqual(silence.searched_bound(30, 1), quiet_bound(30), places=14)
        bounds = [silence.searched_bound(30, m) for m in (1, 2, 5, 10, 100)]
        self.assertEqual(bounds, sorted(bounds))

    def test_a_simulated_search_succeeds_that_often(self):
        rng = random.Random(1618)
        for rate, want in ((silence.searched_bound(30, 10), 0.05), (quiet_bound(30), 0.4013)):
            trials = 30_000
            found = sum(any(rng.expovariate(rate) > 30 for _ in range(10)) for _ in range(trials))
            self.assertAlmostEqual(found / trials, want, delta=0.009, msg=rate)

    def test_the_transcript(self):
        text = expected("ch16_silence")
        for line in ("  chosen in advance:\n    at most 0.09986 msg/s at 95 %", "  found by a search of ten:\n    at most 0.17585 msg/s at 95 %",
                     "  all ten empty:\n    at most 0.00999 msg/s at 95 %", "  finds an empty window: 40.13 %",
                     "    at most 0.18524 msg/s at 95 %", "    silent at the first bound: 19.9 %"):
            self.assertIn(line + "\n", text)


class FirstPositiveCount(unittest.TestCase):
    def test_one_message_in_30_s(self):
        c = trace.current(1, 30)
        self.assertEqual(f"{c.value:.5f}", "0.03333")
        # Since Version 62 the limits are fractions, which Python 3.11 formats only through a float.
        self.assertEqual(tuple(f"{float(x):.5f}" for x in c.interval()), ("-0.03200", "0.09867"))
        self.assertEqual(tuple(f"{x:.5f}" for x in rate_limits(1, 30)), ("0.00084", "0.18572"))
        self.assertEqual(f"{rate_upper(1, 30):.5f}", "0.15813")
        text = expected("ch16_silence")
        # Until Version 69 the ends of an interval were printed to the nearest digit. Since Version 70 the
        # lower end is rounded downward and the upper one upward, so that the printed interval holds the
        # computed one (tests/test_review69.py). The exact interval read 0.00084 to 0.18572: its upper limit is 0.185721...
        for line in ("one message in 30 s: 0.033(33) msg/s", "  normal, 95 %: -0.03200 to 0.09867",
                     "  exact, 95 %: 0.00084 to 0.18573", "  at most 0.15813 msg/s at 95 %"):
            self.assertIn(line + "\n", text)

    def test_closed_forms_for_counts_of_0_1_and_2(self):
        # n = 0: exp(-m) is the whole tail. n = 1: exp(-m)(1 + m). n = 2: exp(-m)(1 + m + m^2 / 2).
        self.assertEqual(rate_limits(0, 30)[0], 0.0)
        # In Version 70 an upper limit was moved upward by 1e-12 of itself, and this test asked for 12 places and
        # for a limit of -log(0.025) / 30 or more. Since Version 71 a limit is the float next to the limit of the
        # model on its outer side, for the level as the number that it is (tests/test_review70.py). The float 0.95
        # lies below 95 %, its half tail above 0.025, and the limit for no event two floats below that quotient,
        # which is the limit of another level, rounded: the floor of Version 70 was a wrong expectation. The
        # places are 14 again, and the side is tested with fifty digits, at the float and at the float before it.
        self.assertAlmostEqual(rate_limits(0, 30)[1], -math.log(0.025) / 30, places=14)
        self.assertEqual(rate_upper(0, 30), quiet_bound(30))
        self.assertAlmostEqual(rate_limits(1, 30)[0], -math.log(0.975) / 30, places=14)
        half = (1 - Decimal(0.95)) / 2
        for n, tail, rate in ((0, half, rate_limits(0, 30)[1]), (1, half, rate_limits(1, 30)[1]),
                              (1, 2 * half, rate_upper(1, 30))):
            with localcontext() as ctx:
                ctx.prec = 50
                tail = 1 * tail
                at, before = (30 * Decimal(r) for r in (rate, math.nextafter(rate, 0.0)))
                self.assertLessEqual((-at).exp() * (1 + n * at), tail, (n, rate))
                self.assertGreater((-before).exp() * (1 + n * before), tail, (n, rate))
            m = 30 * rate
            self.assertAlmostEqual(math.exp(-m) * (1 + n * m), float(tail), places=14)
        low, high = (7 * x for x in rate_limits(2, 7))
        self.assertAlmostEqual(1 - math.exp(-low) * (1 + low), 0.025, places=14)
        self.assertAlmostEqual(math.exp(-high) * (1 + high + high * high / 2), 0.025, places=14)
        # Version 70 also asked that this float be 0.025 or less. It is a rounded sum set against the tail of
        # another level, as above; the side of the limit is tested with fifty digits against the tail of 0.95.
        with localcontext() as ctx:
            ctx.prec = 50
            for rate, more in ((rate_limits(2, 7)[1], False), (math.nextafter(rate_limits(2, 7)[1], 0.0), True)):
                m = 7 * Decimal(rate)
                self.assertEqual((-m).exp() * (1 + m + m * m / 2) > half, more, rate)
        # The exposure only scales the limits.
        self.assertAlmostEqual(rate_limits(2, 7)[1] * 7, rate_limits(2, 1)[1], places=12)

    def test_published_limits(self):
        # Half the chi-squared quantiles of the classical tables: 9.591 and 36.781 for a count of
        # 10 (20 and 22 degrees of freedom), 11.143 for the upper limit of a count of 1 (4).
        low, high = rate_limits(10, 1)
        self.assertEqual((f"{low:.3f}", f"{high:.3f}"), ("4.795", "18.390"))
        self.assertEqual(f"{2 * low:.3f} {2 * high:.3f}", "9.591 36.781")
        self.assertEqual(f"{2 * rate_limits(1, 1)[1]:.3f}", "11.143")
        self.assertEqual(f"{2 * rate_upper(0, 1):.3f}", "5.991")        # chi-squared, 2 degrees, 95 %

    def test_the_limits_cover_at_their_level(self):
        # Exact coverage: the probability of the counts whose interval holds the true mean.
        limits = [rate_limits(n, 1) for n in range(60)]
        uppers = [rate_upper(n, 1) for n in range(60)]
        for mean in (0.02, 0.3, 1.0, 2.5, 4.7, 9.0, 15.5, 20.0):
            pmf = [math.exp(-mean) * mean ** n / math.factorial(n) for n in range(60)]
            self.assertGreater(sum(pmf), 1 - 1e-12)
            two_sided = sum(p for p, (low, high) in zip(pmf, limits) if low <= mean <= high)
            one_sided = sum(p for p, top in zip(pmf, uppers) if mean <= top)
            self.assertGreaterEqual(two_sided, 0.95, mean)
            self.assertGreaterEqual(one_sided, 0.95, mean)
        # The level is reached, not only exceeded: a mean just above the bound for a count of zero
        # is missed by that count alone, which has a probability of just under 5 %.
        mean = uppers[0] * (1 + 1e-9)
        missed = math.exp(-mean)
        self.assertTrue(0.0499 < missed < 0.05)
        self.assertGreater(uppers[1], mean)

    def test_the_tail_is_the_poisson_sum(self):
        for n, mean in ((0, 0.5), (1, 4.74), (3, 0.01), (10, 12.0), (25, 20.0), (0, 30.0)):
            self.assertAlmostEqual(poisson_at_most(n, mean), plain(n, mean), places=14, msg=(n, mean))
        for n, mean in ((564, 600.0), (1905, 2000.0), (1905, 1800.0), (40, 700.5), (9000, 9300.0)):
            want = reference(n, mean)
            self.assertAlmostEqual(poisson_at_most(n, mean) / want, 1.0, places=11, msg=(n, mean))
        self.assertEqual((poisson_at_most(-1, 3.0), poisson_at_most(5, 0.0)), (0.0, 1.0))
        # Large counts: the exact limits approach the estimate with its standard uncertainty.
        low, high = rate_limits(1905, 100)
        self.assertAlmostEqual(low, 19.05 - 1.96 * 0.4365, delta=0.02)
        self.assertAlmostEqual(high, 19.05 + 1.96 * 0.4365, delta=0.03)
        # The 564 messages of the first half minute, as Chapter 16 quotes them.
        exact, c = rate_limits(564, 30), trace.current(564, 30)
        self.assertEqual(tuple(f"{x:.2f}" for x in exact), ("17.28", "20.42"))
        self.assertEqual(tuple(f"{float(x):.2f}" for x in c.interval()), ("17.25", "20.35"))
        self.assertLess(max(abs(a - b) for a, b in zip(exact, c.interval())), c.u / 10)

    def test_what_is_refused(self):
        for call in (rate_upper, rate_limits):
            for exposure in (0, -1, float("nan")):
                with self.assertRaises(ValueError):
                    call(1, exposure)
            for level in (0, 1, 1.5):
                with self.assertRaises(ValueError):
                    call(1, 30, level)

    def test_the_case_of_chapter_23(self):
        self.assertEqual(ch23_case.arrival(0), "at most 0.02996 req/s at 95 %")
        self.assertEqual(ch23_case.arrival(1), "0.00025 to 0.05572 req/s at 95 %")
        self.assertEqual(ch23_case.arrival(1), "{:.5f} to {:.5f} req/s at 95 %".format(*rate_limits(1, 100)))
        # Exact limits as long as the symmetric interval would reach below zero: up to a count of 3.
        self.assertTrue(ch23_case.arrival(3).endswith("req/s at 95 %"))
        self.assertEqual(ch23_case.arrival(4), "0.040(20) req/s")
        self.assertLess(3 - 1.96 * math.sqrt(3), 0)
        self.assertGreater(4 - 1.96 * math.sqrt(4), 0)
        self.assertEqual(ch23_case.arrival(1905), "19.05(44) req/s")
        text = expected("ch23_case")
        for line in ("  its arrival rate at most 0.02996 req/s at 95 %", "  one request: 0.00025 to 0.05572 req/s at 95 %",
                     "arrival rate before 19.05(44) req/s"):
            self.assertIn(line + "\n", text)


class ReversedJudge(unittest.TestCase):
    def test_the_two_cases_are_named(self):
        for x, message in (([0.4, 0.5, 0.5], "uninformative judge"), ([0.4, 0.3, 0.7], "uninformative judge"),
                           ([0.4, 0.10, 0.15], "reversed judge"), ([0.4, 0.5, 0.25], "reversed judge")):
            with self.assertRaises(ValueError) as caught:
                ch20_judge.corrected(x)
            self.assertEqual(str(caught.exception), message)
        self.assertAlmostEqual(ch20_judge.corrected([0.4, 0.90, 0.85]), (0.4 - 0.15) / 0.75, places=12)

    def test_the_example_of_the_review(self):
        k, n, sens, spec = ch20_judge.mirrored(40, 100, 0.10, 0.15)
        self.assertEqual((k, n), (60, 100))
        self.assertAlmostEqual(sens, 0.90, places=15)
        self.assertAlmostEqual(spec, 0.85, places=15)
        y, seen, ends = ch20_judge.few_outputs(k, n, sens, spec)
        self.assertAlmostEqual(y, 0.60, places=12)
        self.assertEqual(tuple(f"{100 * x:.2f}" for x in ends), ("46.29", "72.89"))
        # The same ends from the exact interval of 60 in 100, cut to the model and converted.
        low, high = ch20_judge.exact_interval(60, 100)
        self.assertAlmostEqual(ends[0], (low - 0.15) / 0.75, places=12)
        self.assertAlmostEqual(ends[1], (high - 0.15) / 0.75, places=12)

    def test_exchanged_verdicts_are_the_mirror_judge(self):
        # The mean judged rate of a judge is theta sens + (1 - theta)(1 - spec). Fails read as
        # passes have the rate of a judge with 1 - sens and 1 - spec, for every true rate.
        for sens, spec in ((0.10, 0.15), (0.30, 0.20), (0.45, 0.50)):
            for theta in (0.0, 0.04, 0.5, 0.6, 1.0):
                judged = theta * sens + (1 - theta) * (1 - spec)
                mirror = theta * (1 - sens) + (1 - theta) * spec
                self.assertAlmostEqual(1 - judged, mirror, places=15)
        # At a sum of 1 the judged rate does not depend on the true rate at all.
        rates = {round(theta * 0.3 + (1 - theta) * (1 - 0.7), 12) for theta in (0.0, 0.2, 0.5, 1.0)}
        self.assertEqual(len(rates), 1)

    def test_the_transcript(self):
        text = expected("ch20_judge")
        # Until Version 69 the ends of an interval were printed to the nearest digit. Since Version 70 the
        # lower end is rounded downward and the upper one upward, so that the printed interval holds the
        # computed one (tests/test_review69.py). The interval read 0.463 to 0.729.
        for line in ("sens 0.10, spec 0.15: reversed judge", "  verdicts exchanged, 60 of 100:", "  corrected +0.6000,",
                     "  95 % interval 0.462 to 0.729"):
            self.assertIn(line + "\n", text)


if __name__ == "__main__":
    unittest.main()
