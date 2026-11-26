"""Draw the figures of the book from the same code as the examples.

    python figures/make_figures.py          (needs matplotlib)

The figures are black-and-white line art, 224 pt wide (the text width of
the book), written to figures/out/ as 1200 dpi PNG files. Figures 16.4, 16.7
and 20.1 have one gray band each besides, Figure 16.6 has two, and
Figure 10.2 draws one of its two curves in gray. Figure 6.1 hatches the bars
of one of its two models, and Figure 6.3 shades one region. Each is drawn from the
functions of the example that its chapter prints, or from the numbers of its
caption.
"""
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from examples import ch06_exchange as c6x  # noqa: E402
from examples import ch06_ordering as c6  # noqa: E402
from examples import ch08_aliasing as c8  # noqa: E402
from examples import ch08_counter as c8c  # noqa: E402
from examples import ch08_sampling as c8s  # noqa: E402
from examples import ch10_coordinated_omission as c10  # noqa: E402
from examples import ch12_calibration as c12  # noqa: E402
from examples import ch15_control_chart as c15  # noqa: E402
from examples import ch16_silence as c16s  # noqa: E402
from examples import ch16_trace_measurements as c16  # noqa: E402
from examples import ch17_accuracy as c17  # noqa: E402
from examples import ch18_calibration as c18  # noqa: E402
from examples import ch20_judge as c20  # noqa: E402
from examples import ch21_psi as c21  # noqa: E402
from examples import ch22_record_check as c22  # noqa: E402
from examples import ch23_mix as c23  # noqa: E402
from swmetro.dist import phi  # noqa: E402
from swmetro.record import guard  # noqa: E402
from swmetro.stats import ecdf_quantile, quantile, wilson  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
WIDTH = 224 / 72  # inches
BLACK, GREY, PALE = "#000000", "#8c8c8c", "#cfcfcf"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.7,
    "lines.markersize": 3,
    "legend.frameon": False,
    "savefig.dpi": 1200,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    try:  # store as greyscale, as line art should be
        from PIL import Image
        Image.open(path).convert("L").save(path, dpi=(1200, 1200))
    except ImportError:
        pass
    print("wrote", path.relative_to(ROOT))


def reversed_pairs():
    """The number of pairs recorded in reverse order among ten pairs that are
    truly 1 ms apart, for the two models of Listing 6.2: clock errors drawn
    anew for every pair, and one offset kept by each clock through the
    capture. The bars are reversed_of(); the chance for one pair is that of
    the transcript."""
    p = phi(-1.0 / (c6.SIGMA * math.sqrt(2)))
    anew = [100 * q for q in c6.reversed_of(10, p, False)]
    kept = [100 * q for q in c6.reversed_of(10, p, True)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    half = 0.36
    ax.bar([k - half / 2 for k in range(11)], anew, width=half, color=BLACK, lw=0,
           label="errors drawn anew for each pair")
    ax.bar([k + half / 2 for k in range(11)], kept, width=half, color="white", edgecolor=BLACK,
           lw=0.5, hatch="/////", label="one offset kept through the capture")
    for k in (0, 10):
        ax.text(k + half / 2, kept[k] + 1.5, f"{kept[k]:.2f} %", ha="center", va="bottom")
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(0, 78)
    ax.set_xticks(range(11))
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("pairs recorded in reverse order, of ten")
    ax.set_ylabel("captures, %")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.24)
    save(fig, "fig-06-1-reversed-pairs.png")


def exchange_gap():
    """The true time from an event stamped on host A to one stamped 3 ms
    earlier on host B, as each of the two exchanges of Listing 6.3 leaves it
    and as both do: gap() for the offsets of offset() and of both(). The
    lowest line is the last case of the transcript, with every stamp within
    0.5 ms of its event."""
    spans = [c6x.offset(*stamps) for stamps in c6x.EXCHANGES]
    wide = c6x.both([c6x.offset(*stamps, err=0.5) for stamps in c6x.EXCHANGES])
    rows = (("exchange 1", c6x.gap(0, -3, spans[0])), ("exchange 2", c6x.gap(0, -3, spans[1])),
            ("both", c6x.gap(0, -3, c6x.both(spans))),
            ("both, stamps\nwithin 0.5 ms", c6x.gap(0, -3, wide, err=0.5)))
    rows = [(name, tuple(map(float, span))) for name, span in rows]    # the listing returns exact fractions
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3))
    ax.axvline(0, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    for row, (name, (lo, hi)) in enumerate(rows):
        y = 3 - row
        ax.plot([lo, hi], [y, y], color=BLACK, lw=1.8 if name == "both" else 0.8,
                marker="o", ms=3.2, solid_capstyle="butt", ls=(0, (3, 1.5)) if row == 3 else "-")
    ax.text(-0.14, 3.72, "B first", ha="right", va="center")
    ax.text(0.14, 3.72, "A first", ha="left", va="center")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([name for name, _ in rows], linespacing=1.1)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 4.0)
    ax.set_xlim(-1.9, 5.4)
    ax.set_xticks(range(-1, 6))
    ax.set_xticklabels(["\u22121", "0", "+1", "+2", "+3", "+4", "+5"])
    ax.set_xlabel("event on B after event on A, ms")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.98, bottom=0.28)
    save(fig, "fig-06-2-exchange-gap.png")


def calibration_age():
    """The time since the two exchanges of Listing 6.3 and the bound on the
    error of each stamp up to which the event on B is still the later one, if
    the offset moves by RATE of the listing at most: holds() for bounds from 0
    to 0.150 ms. The three marked points are the cases of the transcript. On
    the line itself the two events can be simultaneous."""
    def seconds(bound_ms):
        span = c6x.both([c6x.offset(*stamps, err=bound_ms) for stamps in c6x.EXCHANGES])
        return float(c6x.holds(0, -3, span, bound_ms))
    bounds = [k / 1000 for k in range(0, 151)]                 # ms
    ages = [seconds(e) for e in bounds]
    top = next(e for e, s in zip(bounds, ages) if s == 0)      # the order is open from here, ms
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3))
    edge = [(s, e) for e, s in zip(bounds, ages) if e <= top]
    ax.fill([x for x, _ in edge] + [0], [y for _, y in edge] + [0], color=PALE, lw=0)
    ax.plot(*zip(*edge), color=BLACK, lw=0.9)
    marks = [(seconds(e), e) for e in (0, 0.1, top)]
    ax.plot(*zip(*marks), "o", color=BLACK, ms=3.4, mfc="white", mew=0.7, clip_on=False, zorder=3, ls="none")
    ax.annotate(f"{seconds(0.1):.0f} s, 0.100 ms", xy=(seconds(0.1), 0.1), xytext=(8.4, 0.127),
                ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, shrinkB=2.5))
    ax.text(2.2, 0.03, "B is the later", fontweight="bold", ha="left", va="center")
    ax.text(21.5, 0.118, "either order", ha="left", va="center")
    slope = math.degrees(math.atan2(-top, ages[0]))            # of the line, in the units of the axes
    ax.text(15.6, 0.0565, "simultaneous events possible", ha="center", va="bottom", rotation=slope,
            rotation_mode="anchor", transform_rotates_text=True)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.158)
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks([k / 40 for k in range(7)])
    ax.set_yticklabels([f"{k / 40:.3f}" for k in range(7)])
    ax.set_xlabel("time since the exchanges, s")
    ax.set_ylabel("bound for each\nstamp, ms", linespacing=1.1)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.27)
    save(fig, "fig-06-3-calibration-age.png")


def third_host():
    """Three hosts of Listing 6.3. Left: the offset of the clock at the head of
    each arrow from the clock at its tail, as the exchanges between the two
    hosts leave it: both() of the two exchanges of A and B, and offset() of
    B_TO_C and of A_TO_C. Right: the true time from an event stamped 300 ms on
    A to one stamped 299 ms on C, gap() for the offset of C from A by their
    own exchange, by way of B with chain(), and by both."""
    a_b = c6x.both([c6x.offset(*stamps) for stamps in c6x.EXCHANGES])
    b_c, direct = c6x.offset(*c6x.B_TO_C), c6x.offset(*c6x.A_TO_C)
    via_b = c6x.chain(a_b, b_c)
    rows = [(name, tuple(map(float, c6x.gap(300, 299, span))))
            for name, span in (("direct", direct), ("via B", via_b), ("both", c6x.both([direct, via_b])))]

    def said(span):
        return " to ".join(f"{float(x):+g}".replace("-", "\u2212") for x in span)
    fig = plt.figure(figsize=(WIDTH, 1.25))
    left = fig.add_axes([0.0, 0.0, 0.44, 1.0])
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.axis("off")
    at = {"A": (0.16, 0.24), "B": (0.5, 0.84), "C": (0.84, 0.24)}
    for tail, head, span, dx, dy, turn in (("A", "B", a_b, -0.045, 0.03, 52.0), ("B", "C", b_c, 0.045, 0.03, -52.0),
                                           ("A", "C", direct, 0.0, -0.085, 0.0)):
        left.annotate("", xy=at[head], xytext=at[tail],
                      arrowprops=dict(arrowstyle="-|>", lw=0.7, color=BLACK, shrinkA=6.5, shrinkB=6.5, mutation_scale=7))
        x, y = (at[tail][0] + at[head][0]) / 2 + dx, (at[tail][1] + at[head][1]) / 2 + dy
        left.text(x, y, said(span), ha="center", va="center", rotation=turn, rotation_mode="anchor")
    for name, (x, y) in at.items():
        left.text(x, y, name, ha="center", va="center", fontweight="bold",
                  bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=BLACK, lw=0.7))
    left.text(0.5, 0.03, "clock offsets, ms", ha="center", va="bottom")
    ax = fig.add_axes([0.6, 0.27, 0.37, 0.65])
    ax.axvline(0, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    for row, (name, (lo, hi)) in enumerate(rows):
        y = 2 - row
        ax.plot([lo, hi], [y, y], color=BLACK, lw=1.8 if name == "both" else 0.8, marker="o", ms=3.2, solid_capstyle="butt")
    ax.text(-0.14, 2.75, "C first", ha="right", va="center")
    ax.text(0.14, 2.75, "A first", ha="left", va="center")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([name for name, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 3.05)
    ax.set_xlim(-2.6, 4.5)
    ax.set_xticks(range(-2, 5, 2))
    ax.set_xticklabels(["\u22122", "0", "+2", "+4"])
    ax.set_xlabel("event on C after A, ms")
    save(fig, "fig-06-4-third-host.png")


def three_host_bound():
    """The three hosts of Listing 6.3 with every stamp, of the exchanges and of
    the two events, within a bound of its event: routes() and gap() for bounds
    from 0 to 0.3 ms. Solid: the least time from the event stamped 300 ms on A
    to the one stamped 299 ms on C. Dashed: with 62 ms for the last stamp of A,
    the least offset of C by the exchange of A and C less its greatest offset by
    way of B, which is how far the two are apart where it is above zero and how
    far they overlap where it is below. Each circle is where a line reaches
    zero."""
    from fractions import Fraction
    other = c6x.A_TO_C[:3] + (62,)
    bounds = [Fraction(k, 400) for k in range(121)]

    def least(e):
        return c6x.gap(300, 299, c6x.routes(c6x.A_TO_C, e)[2], e)[0]

    def apart(e):
        direct, via_b, _ = c6x.routes(other, e)
        return direct[0] - via_b[1]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3))
    ax.axhline(0, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    for value, style, name, y in ((least, "-", "event on C after A, at least", 1.47),
                                  (apart, (0, (4, 2)), "the two offsets of C apart,", 1.2)):
        ax.plot([float(e) for e in bounds], [float(value(e)) for e in bounds], color=BLACK, lw=0.9, ls=style)
        step = bounds[4]
        zero = value(0) / ((value(0) - value(step)) / step)          # the line is straight
        assert value(zero) == 0
        ax.plot([float(zero)], [0], "o", color=BLACK, ms=3.4, mfc="white", mew=0.7, zorder=3)
        ax.annotate(f"{float(zero):.3f} ms", xy=(float(zero), 0), xytext=(float(zero) + 0.012, 0.33),
                    ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, shrinkB=2.5))
        ax.plot([0.118, 0.146], [y, y], color=BLACK, lw=0.9, ls=style)   # the key, where no line passes
        ax.text(0.153, y, name, ha="left", va="center")
    ax.text(0.153, 0.95, "below zero overlapping", ha="left", va="center")   # the second line of that entry
    ax.set_xlim(0, 0.305)
    ax.set_ylim(-0.45, 1.6)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["0", "0.5", "1.0", "1.5"])
    ax.set_xlabel("bound for each stamp, ms")
    ax.set_ylabel("ms")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.27)
    save(fig, "fig-06-5-three-host-bound.png")


def stamps_at_fault():
    """How many stamps of the four exchanges of Listing 6.3 can each be the one
    at fault, against the last stamp of A in its exchange with C: suspects() for
    last stamps from 59 to 65 ms in steps of 1/8 ms. A filled circle belongs to
    its line and an open one does not. Where the count is zero the exchanges
    agree, and that part is shaded."""
    from fractions import Fraction
    lasts = [59 + Fraction(k, 8) for k in range(49)]
    counts = [len(c6x.suspects(c6x.A_TO_C[:3] + (last,))) for last in lasts]
    runs, first = [], 0
    for k in range(1, len(lasts) + 1):
        if k == len(lasts) or counts[k] != counts[first]:
            runs.append((lasts[first], lasts[min(k, len(lasts) - 1)], counts[first]))
            first = k
    assert [(float(a), n) for a, _, n in runs] == [(59, 1), (61, 2), (62, 3), (62.5, 6), (63.5, 0)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.35))
    for a, b, n in runs:
        if not n:
            ax.axvspan(float(a), 65.25, color=PALE, lw=0)
            ax.text((float(a) + 65.25) / 2, 3.4, "the four\nexchanges\nagree", ha="center", va="center")
            continue
        ax.plot([float(a), float(b)], [n, n], color=BLACK, lw=0.9)
        ax.plot([float(a)], [n], "o", color=BLACK, ms=3.4, zorder=3)
        ax.plot([float(b)], [n], "o", color=BLACK, ms=3.4, mfc="white", mew=0.7, zorder=3)
    ax.text(62.5, 6.62, "62.5", ha="center", va="bottom")      # the two ends that no tick names
    ax.text(63.45, 6.62, "63.5", ha="right", va="bottom")
    ax.set_xlim(58.75, 65.25)
    ax.set_ylim(0.3, 7.6)
    ax.set_xticks([59, 60, 61, 62, 63, 64, 65])
    ax.set_yticks([1, 2, 3, 6])
    ax.set_xlabel("last stamp of A, ms")
    ax.set_ylabel("stamps")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.26)
    save(fig, "fig-06-6-stamps-at-fault.png")


def aliasing():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.65))
    ts = [i * 0.25 for i in range(4 * 240 + 1)]
    ax.plot(ts, [c8.gauge(t) for t in ts], color=GREY, lw=0.35)
    scrape = [i * c8.SCRAPE for i in range(17)]
    ax.plot(scrape, [c8.gauge(t) for t in scrape], color=BLACK, lw=0.9,
            marker="o", ms=2.6)
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 118)
    ax.set_xticks(range(0, 241, 60))
    ax.set_yticks([10, 50, 90])
    ax.set_xlabel("time, s")
    ax.set_ylabel("queue depth")
    ax.text(240, 108, "signal, period 16 s", color="#555555", ha="right")
    ax.annotate("scraped every 15 s: period 240 s", xy=(150, 78.3),
                xytext=(2, 108), fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.4, shrinkB=2,
                                relpos=(0.75, 0.0)))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.97, bottom=0.22)
    save(fig, "fig-08-1-aliasing.png")


EMPTY_ERRORS, EMPTY_RATE = 350, 0.01


def counter_readings():
    """Two streams of bytes at constant rates that leave the same two readings
    of a 32-bit counter 100 s apart, and the reading at 50 s that tells them
    apart: the readings and the totals are those of Listing 8.2. The slower
    rate is 7.05032704 MB/s, and its label, to two places, says about."""
    first, last = c8c.wrapping(100, 2)
    middle = c8c.wrapping(50, 2)[1]
    slow, fast = c8c.increments(first, last, 60_000_000 * 100)
    wrap = 100 * c8c.WRAP / fast  # the faster stream is back at zero here, s
    gb = 1e9
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    ax.plot([0, 100], [first / gb, (first + slow) / gb], color=BLACK,
            ls=(0, (4, 2)))
    ax.plot([0, wrap], [first / gb, c8c.WRAP / gb], color=BLACK, lw=0.9)
    ax.plot([wrap, 100], [0, last / gb], color=BLACK, lw=0.9)
    ax.plot([wrap, wrap], [0, c8c.WRAP / gb], color=GREY, lw=0.4, ls=":")
    ax.plot([wrap], [c8c.WRAP / gb], "o", color=BLACK, ms=2.8, mfc="white",
            mew=0.6)
    ax.plot([wrap], [0], "o", color=BLACK, ms=2.2)
    ax.plot([0, 100], [first / gb, last / gb], "o", color=BLACK, ms=3.4,
            clip_on=False)
    ax.plot([50], [middle / gb], "s", color=BLACK, ms=3.4, mfc="white", mew=0.7)
    ax.text(47, middle / gb + 0.18, f"reading at 50 s:\n{middle / gb:.1f} GB",
            ha="right", va="bottom")
    ax.text(83.5, last / gb + 0.3, f"both end at\n{last / gb:.3f} GB",
            ha="right", va="bottom")
    ax.text(64, 3.55, f"{fast / 1e8:.0f} MB/s", ha="right", va="bottom")
    ax.text(40, 0.42, f"\u2248 {slow / 1e8:.2f} MB/s", ha="right", va="bottom")
    ax.text(wrap - 3, c8c.WRAP / gb + 0.05, "wrap", ha="right", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.15, 4.75)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("time, s")
    ax.set_ylabel("counter reading, GB")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.97, bottom=0.23)
    save(fig, "fig-08-2-counter-readings.png")


def empty_record():
    """The chance that a sampled record holds none of 350 failures, against the
    probability with which a failure is kept: one independent decision for each
    failure, and one for each of 35 traces of ten failures. The curves and the
    two marks at 1 % are none_kept() of Listing 8.3."""
    rates = [i / 2000 for i in range(201)]                # 0 to 10 %, in steps of 0.05 points
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    cases = ((10, "-", "35 decisions, ten failures each: {:.2f} %", (2.1, 66)),
             (1, (0, (4, 2)), "350 decisions,\none failure each: {:.2f} %", (1.45, 13.5)))
    for group, style, label, at in cases:
        ys = [100 * c8s.none_kept(EMPTY_ERRORS, rate, group) for rate in rates]
        ax.plot([100 * rate for rate in rates], ys, color=BLACK, lw=0.9, ls=style)
        y = 100 * c8s.none_kept(EMPTY_ERRORS, EMPTY_RATE, group)
        ax.plot([100 * EMPTY_RATE], [y], marker="o", ms=3.2, mfc=BLACK, mec=BLACK, zorder=3)
        ax.text(at[0], at[1], label.format(y), ha="left", va="center", linespacing=1.15)
    ax.axvline(100 * EMPTY_RATE, color=BLACK, lw=0.35, ls=(0, (1, 2)), zorder=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 104)
    ax.set_xticks([0, 1, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("probability that a failure is kept, %")
    ax.set_ylabel("records with no failure, %")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.25)
    save(fig, "fig-08-3-empty-record.png")


def omission():
    qs = [0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    labels = ["50", "90", "95", "97", "98", "99", "99.5", "99.9"]
    measured = c10.closed_loop()
    rows = (("as measured", measured, "o", "-"),
            ("back-filled", c10.back_fill(measured), "s", "--"),
            ("from intended start", c10.open_loop(), "D", "-"))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    xs = range(len(qs))
    for name, data, marker, style in rows:
        ys = [quantile(data, q) for q in qs]
        ax.plot(xs, ys, color=BLACK, marker=marker, ls=style, ms=2.8,
                mfc="white" if marker == "s" else BLACK, mew=0.6)
        dy = {"as measured": 1.25, "back-filled": 0.42,
              "from intended start": 1.5}[name]
        ax.text(7.0, ys[-1] * dy, name, ha="right",
                va="bottom" if dy > 1 else "top",
                fontweight="bold" if name == "as measured" else "normal")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 4000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_xlabel("selected percentiles, equally spaced")
    ax.set_ylabel("latency, ms")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.21)
    save(fig, "fig-10-1-coordinated-omission.png")


def two_conventions():
    """The upper percentiles of the measured requests of Listing 10.1 under the
    two conventions that its transcript compares at the 99.97th."""
    measured = c10.closed_loop()
    n = len(measured)
    qs = [0.9995 + k * 0.0004 / 800 for k in range(801)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    ax.axhline(500, color=GREY, lw=0.5, ls=":")
    ax.text(99.9503, 520, "objective, 500 ms", ha="left", va="bottom")
    ax.plot([100 * q for q in qs], [quantile(measured, q) for q in qs],
            color=GREY, lw=1.6, solid_capstyle="butt")
    # the inverse of the empirical distribution steps at the share of the
    # requests that took 1 ms: it has the lower value at that share itself
    step = measured.count(measured[0]) / n
    low, high = ecdf_quantile(measured, 0.9995), ecdf_quantile(measured, 0.9999)
    assert ecdf_quantile(measured, step) == low < high
    ax.plot([99.95, 100 * step], [low, low], color=BLACK, ls=(0, (4, 2)))
    ax.plot([100 * step, 99.99], [high, high], color=BLACK, ls=(0, (4, 2)),
            clip_on=False)
    ax.plot([100 * step] * 2, [low, high], color=BLACK, lw=0.4, ls=":")
    ax.plot([100 * step], [low], "o", color=BLACK, ms=2.2)
    ax.plot([100 * step], [high], "o", color=BLACK, ms=2.6, mfc="white",
            mew=0.6)
    a, b = quantile(measured, 0.9997), ecdf_quantile(measured, 0.9997)
    ax.plot([99.97], [a], "o", color=BLACK, ms=3.6)
    ax.plot([99.97], [b], "s", color=BLACK, ms=3.4)
    ax.text(99.9707, a - 30, f"{a:.1f} ms", ha="left", va="top")
    ax.text(99.9707, b - 60, f"{b:.0f} ms", ha="left", va="top")
    ax.text(99.9899, b + 45, "inverse of the empirical distribution",
            ha="right", va="bottom")
    ax.text(99.9762, 450, "interpolated", ha="left", va="top")
    ax.set_xlim(99.95, 99.99)
    ax.set_ylim(-70, 1190)
    ax.set_xticks([99.95, 99.96, 99.97, 99.98, 99.99])
    ax.set_xticklabels(["99.95", "99.96", "99.97", "99.98", "99.99"])
    ax.set_yticks([1, 500, 1001])
    ax.set_yticklabels(["1", "500", "1001"])
    ax.set_xlabel("percentile of the measured requests")
    ax.set_ylabel("latency, ms")
    fig.subplots_adjust(left=0.14, right=0.94, top=0.97, bottom=0.23)
    save(fig, "fig-10-2-two-conventions.png")


def calibration_readings():
    """The standard uncertainty of the corrected delay of Listing 12.1 against
    the number of new indications whose mean is corrected, for the calibration
    of the listing and for five times its readings at the same settings:
    measure() for 1 to 100 indications. The circles are the six values of the
    transcript. The dotted line is the uncertainty of the reference delays,
    which is common to all readings and which no repeat reduces."""
    xs, fit = c12.calibrate(random.Random(12))
    ms = list(range(1, 101))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    ax.axhline(c12.U_REF, color=BLACK, lw=0.6, ls=(0, (1, 2)))
    ax.text(1.05, c12.U_REF - 0.004, f"reference delays, {c12.U_REF:.3f} ms", ha="left", va="top")
    for times, style, y in ((1, "-", 0.150), (5, (0, (4, 2)), 0.128)):
        us = [c12.measure(37.90, xs, fit, m, times).u for m in ms]
        ax.plot(ms, us, color=BLACK, lw=0.9, ls=style)
        ax.plot([1, 5, 30], [us[0], us[4], us[29]], "o", color=BLACK, ms=3.2, mfc="white", mew=0.7, zorder=3, ls="none")
        ax.plot([6.5, 11], [y, y], color=BLACK, lw=0.9, ls=style)          # the key, where no curve passes
        ax.text(12.5, y, f"{30 * times} calibration readings", ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.9, 100)
    ax.set_ylim(0, 0.17)
    ax.set_xticks([1, 2, 5, 10, 30, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0", "0.05", "0.10", "0.15"])
    ax.set_xlabel("new indications in the mean")
    ax.set_ylabel("standard uncertainty, ms")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.97, bottom=0.25)
    save(fig, "fig-12-1-calibration-readings.png")


def control_chart():
    xs = c15.series(random.Random(15))
    centre, sigma = c15.xmr(xs[:30])
    zs = [(x - centre) / sigma for x in xs]
    hi, cus = 0.0, []
    for z in zs[30:]:
        hi = max(0.0, hi + z - 0.5)
        cus.append(hi)
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 2.5), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1]})
    n = range(1, len(xs) + 1)
    a.plot(n, xs, color=BLACK, marker="o", ms=1.8, lw=0.5)
    for y, style in ((centre, "-"), (centre + 3 * sigma, "--"),
                     (centre - 3 * sigma, "--")):
        a.axhline(y, color=BLACK, lw=0.4, ls=style)
    a.axvline(30.5, color=GREY, lw=0.4)
    a.text(31.5, centre - 3 * sigma + 0.5, "mean shifts by 1.5 ms", va="bottom")
    a.text(70.5, centre + 3 * sigma + 0.3, "upper limit", ha="right",
           va="bottom")
    a.set_ylabel("benchmark, ms")
    a.set_ylim(centre - 4.2 * sigma, centre + 4.4 * sigma)
    b.plot(range(31, 71), cus, color=BLACK, marker="o", ms=1.8, lw=0.5)
    b.axhline(5, color=BLACK, lw=0.4, ls="--")
    b.axvline(30.5, color=GREY, lw=0.4)
    first = next(i for i, c in enumerate(cus) if c > 5)
    b.plot([31 + first], [cus[first]], marker="o", ms=5.5, mfc="none",
           mec=BLACK, mew=0.6)
    b.annotate("signal", xy=(31 + first, cus[first]),
               xytext=(31 + first - 9, cus[first] + 3.5), fontweight="bold",
               arrowprops=dict(arrowstyle="-", lw=0.4))
    b.text(1, 5.4, "decision interval h = 5", va="bottom")
    b.set_ylabel("CUSUM, sigma")
    b.set_xlabel("nightly run")
    b.set_xlim(0, 71)
    b.set_ylim(0, max(cus) * 1.08)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.14,
                        hspace=0.12)
    save(fig, "fig-15-1-control-chart.png")


def request_starts():
    """The start times of a request, relative to a window of 30 s, at which
    its start message, its end message, or either falls into the window:
    above for the Time Delta that Listing 16.1 measures, below for a request
    of 60 s, which is longer than the window. Filled ends belong to an
    interval and open ends do not."""
    text = c16.LOG.read_text(encoding="utf-8")
    rows = [c16.parse(line) for line in text.splitlines()]
    whole = [(0.0, 30.0)]

    def label(x):
        return f"{x:.3f}".rstrip("0").rstrip(".").replace("-", "\u2212")

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2.55))
    for ax, d in zip(axes, (c16.time_delta(rows).value / 1000, 60.0)):
        ends = [(x - d, y - d) for x, y in whole]
        bars = (("either", c16s.starts(whole, d)), ("end\nmessage", ends), ("start\nmessage", whole))
        bars = [(name, [(float(x), float(y)) for x, y in pieces]) for name, pieces in bars]   # starts() are fractions
        left, right = (-8, 37) if d < 30 else (-80, 38)
        width = right - left
        for edge in (0, 30):
            ax.axvline(edge, color=BLACK, lw=0.4, ls="--")
        ax.text(15, 2.78, "window", ha="center", va="center")
        ax.set_title(f"a request of {label(d)} s", loc="left", fontsize=6.5, pad=2)
        for row, (_, pieces) in enumerate(bars):
            for lo, hi in pieces:
                ax.plot([lo, hi], [row, row], color=BLACK, lw=1.6, solid_capstyle="butt")
                ax.plot([lo], [row], marker="o", ms=3.2, mfc=BLACK, mec=BLACK, mew=0.7, zorder=3)
                ax.plot([hi], [row], marker="o", ms=3.2, mfc="white", mec=BLACK, mew=0.7, zorder=3)
                gap = 0.016 * width
                ax.text(lo - gap, row + 0.34, label(lo), ha="right", va="center")
                inside = 0 < hi < 30  # keep the label clear of the edge of the window
                ax.text(hi - gap if inside else hi + gap, row + 0.34, label(hi),
                        ha="right" if inside else "left", va="center")
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([name for name, _ in bars])
        ax.set_ylim(-0.6, 3.15)
        ax.set_xlim(left, right)
        ax.set_xticks([0, 10, 20, 30] if d < 30 else [-60, -30, 0, 30])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[1].set_xticklabels(["\u221260", "\u221230", "0", "30"])
    axes[1].set_xlabel("start of a request that leaves a message in the window, s")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.14, hspace=0.5)
    save(fig, "fig-16-1-request-starts.png")


def mean_duration():
    """The start times covered by a window of 30 s against the duration of a
    request, reach() of Listing 16.4, and two sources with a mean duration
    of 30 s: one duration, and two durations in equal shares, for which
    mean_reach() gives the middle of the chord."""
    whole = [(0, 30)]
    ds = [d / 2 for d in range(0, 111)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    ax.plot(ds, [c16s.reach(whole, d) for d in ds], color=BLACK, lw=1.2)
    ends = [(d, c16s.reach(whole, d)) for d in (10, 50)]
    ax.plot(*zip(*ends), color=BLACK, lw=0.6, ls=(0, (3.2, 1.6)), marker="o", ms=2.2)
    one = c16s.mean_reach(whole, [(1, 30)])
    two = c16s.mean_reach(whole, [(0.5, 10), (0.5, 50)])
    ax.plot([30, 30], [two, one], color=GREY, lw=0.5, ls=":")
    ax.plot([30], [one], marker="o", ms=3.6, color=BLACK, lw=0, zorder=3)
    ax.plot([30], [two], marker="o", ms=3.6, mfc="white", mec=BLACK, mew=0.8, lw=0, zorder=3)
    ax.text(28.5, one + 1.2, "all requests 30 s", ha="right", va="bottom")
    ax.text(31.5, two - 1.2, "half 10 s, half 50 s", ha="left", va="top")
    ax.set_xlim(0, 55)
    ax.set_ylim(28, 65)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([30, 40, 50, 60])
    ax.set_xlabel("duration of a request, or mean duration, s")
    ax.set_ylabel("start times covered, s")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.25)
    save(fig, "fig-16-2-mean-duration.png")


def counted_where():
    """Two records of the half minute from 30 s to 60 s with the same verified
    stretches and one service message each: in the gap, and in verified
    time. The records are those of gap() in Listing 16.4. An end of a stretch
    is filled where a line of its run is stamped: that line is counted with
    the stretch. It is open where the window closes, at 60 s, and the line
    stamped there lies outside the window."""
    fig, ax = plt.subplots(figsize=(WIDTH, 1.15))
    cases = ((35, "message\nat 35 s"), (45, "message\nat 45 s"))
    for row, (at, _) in enumerate(cases):
        rows = c16s.gap(at)
        stamp = {seq: ms for seq, ms, _, _ in rows}
        ax.plot([30, 60], [row, row], color=GREY, lw=0.5, ls=":")
        for lo, hi, own in c16.runs(rows, 30, 60, (0, 60_000)):
            ax.plot([lo / 1000, hi / 1000], [row, row], color=BLACK, lw=1.6, solid_capstyle="butt")
            for end, seq in ((lo, own[0]), (hi, own[-1])):
                fill = BLACK if stamp[seq] == end else "white"
                ax.plot([end / 1000], [row], marker="o", ms=3.2, mfc=fill, mec=BLACK, mew=0.7, zorder=3)
        for _, ms, comp, _ in rows:
            if comp == "svc":
                ax.plot([ms / 1000], [row + 0.32], marker="v", ms=4.2, color=BLACK, lw=0)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([name for _, name in cases])
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlim(28, 62)
    ax.set_xticks([30, 35, 40, 45, 50, 60])
    ax.set_xlabel("time in the capture, s")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.31)
    save(fig, "fig-16-3-counted-where.png")


GAP_MESSAGES, GAP_FROM = (40, 74), 42   # two messages of a record, and where an empty 30 s begin


def gap_anywhere():
    """Two windows of 30 s on a fixed grid, from 30 s and from 60 s, with one
    message in each, at 40 s and at 74 s. A search of the two windows finds
    no empty one; the 30 s from 42 s lie across their boundary and hold no
    message. gap_somewhere() of Listing 16.4 gives the chance of such a gap."""
    grid, length = (30, 60, 90), 30
    held = [sum(lo <= t < hi for t in GAP_MESSAGES) for lo, hi in zip(grid, grid[1:])]
    band = (GAP_FROM, GAP_FROM + length)
    assert held == [1, 1] and not any(band[0] <= t < band[1] for t in GAP_MESSAGES)

    fig, ax = plt.subplots(figsize=(WIDTH, 1.1))
    ax.axvspan(*band, color=PALE, lw=0)
    ax.text(sum(band) / 2, 1.38, f"{length} s without\na message", ha="center", va="center")
    ax.text(band[0] - 1.2, 1.38, f"{band[0]}", ha="right", va="center")
    ax.text(band[1] + 1.2, 1.38, f"{band[1]}", ha="left", va="center")
    for edge in grid:
        ax.plot([edge, edge], [-0.5, 0.8], color=BLACK, lw=0.5, ls=":")
    ax.plot([grid[0], grid[-1]], [0, 0], color=BLACK, lw=0.6)
    for t in GAP_MESSAGES:
        ax.plot([t], [0], marker="o", ms=4.2, mfc=BLACK, mec=BLACK, zorder=3)
        ax.text(t, 0.52, f"{t}", ha="center", va="center")
    ax.set_ylim(-0.5, 1.9)
    ax.set_xlim(26, 94)
    ax.set_xticks(range(30, 91, 10))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time in the record, s")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.34)
    save(fig, "fig-16-4-gap-anywhere.png")


GAP_COUNT, GAP_RECORD, GAP_MARKS = 60, 300, (30, 35, 40, 45)


def gap_given_count():
    """A homogeneous Poisson process, observed for 300 s, given exactly 60
    messages: the share of such records in which the largest gap, the two at
    the ends of the record included, is of a length or more. The curve and
    its four marks are gap_given() of Listing 16.4."""
    def share(length):
        return 100 * c16s.gap_given(GAP_COUNT, GAP_RECORD, length)
    xs = [GAP_MARKS[0] + i / 8 for i in range(8 * (GAP_MARKS[-1] - GAP_MARKS[0]) + 1)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.35))
    ax.plot(xs, [share(x) for x in xs], color=BLACK, lw=0.9)
    for x in GAP_MARKS:
        y = share(x)
        ax.plot([x], [y], marker="o", ms=3.2, mfc=BLACK, mec=BLACK, zorder=3)
        ax.text(x + 0.45, y + 0.75, f"{y:.2f} %", ha="left", va="center")
    ax.set_xlim(29, 48)
    ax.set_ylim(0, 12.5)
    ax.set_xticks(GAP_MARKS)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xlabel("a gap of this length or more, s")
    ax.set_ylabel("share of records, %")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.27)
    save(fig, "fig-16-5-gap-given-count.png")


def changing_rate():
    """A record of 300 s whose reference rate is 0.10 msg/s for 150 s and
    0.30 msg/s after that: the messages expected from its start up to each
    time, expected() of Listing 16.4. The two shaded windows of 30 s, QUIET
    of the listing, span 3 and 9 expected messages."""
    turn = c16s.RISE[0][0]
    total = sum(seconds for seconds, _ in c16s.RISE)

    def count(t):
        return float(c16s.expected(t))
    spans = [count(hi) - count(lo) for lo, hi in c16s.QUIET]
    assert (count(turn), count(total), spans) == (15, 60, [3, 9])

    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    for lo, hi in c16s.QUIET:
        ax.axvspan(lo, hi, color=PALE, lw=0)
    ax.plot([turn, turn], [0, count(total)], color=BLACK, lw=0.5, ls=":")
    ts = (0, turn, total)
    ax.plot(ts, [count(t) for t in ts], color=BLACK, lw=0.7)
    for (lo, hi), span, above in zip(c16s.QUIET, spans, (True, False)):
        ends = [count(lo), count(hi)]
        ax.plot([lo, hi], ends, color=BLACK, lw=1.9, solid_capstyle="butt")
        ax.plot([lo, hi], ends, ls="none", marker="o", ms=3.2, mfc=BLACK, mec=BLACK, zorder=3)
        y = ends[1] + 11 if above else ends[0] - 13
        ax.text((lo + hi) / 2, y, f"{span:g} expected\nmessages", ha="center", va="center")
    ax.set_xlim(0, total)
    ax.set_ylim(0, 63)
    ax.set_xticks([0, turn, total] + [t for span in c16s.QUIET for t in span])
    ax.set_yticks([0, 15, 30, 45, 60])
    ax.set_xlabel("time in the record, s")
    ax.set_ylabel("messages expected\nfrom the start")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.25)
    save(fig, "fig-16-6-changing-rate.png")


def shared_starts():
    """Two windows of 30 s that follow one another and requests of 10 s: the
    start times from which a request leaves a message in each, starts() of
    Listing 16.4, and the start times that the two have in common, from
    which a request writes its start into the first and its end into the
    second."""
    windows = (("message in\n0 to 30 s", [(0, 30)]), ("message in\n30 to 60 s", [(30, 60)]))
    found = [[(float(x), float(y)) for x, y in c16s.starts(w, 10)] for _, w in windows]
    (a, b), (c, d) = found[0][0], found[1][0]
    both = (max(a, c), min(b, d))

    def label(x):
        return f"{x:g}".replace("-", "\u2212")

    fig, ax = plt.subplots(figsize=(WIDTH, 1.25))
    ax.axvspan(*both, color=PALE, lw=0)
    ax.text(sum(both) / 2, 1.62, "shared\nstarts", ha="center", va="center")
    for row, pieces in zip((1, 0), found):
        for lo, hi in pieces:
            ax.plot([lo, hi], [row, row], color=BLACK, lw=1.6, solid_capstyle="butt")
            ax.plot([lo], [row], marker="o", ms=3.2, mfc=BLACK, mec=BLACK, mew=0.7, zorder=3)
            ax.plot([hi], [row], marker="o", ms=3.2, mfc="white", mec=BLACK, mew=0.7, zorder=3)
            ax.text(lo - 1.2, row + 0.3, label(lo), ha="right", va="center")
            ax.text(hi + 1.2, row + 0.3, label(hi), ha="left", va="center")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([name for name, _ in windows])
    ax.set_ylim(-0.55, 2.05)
    ax.set_xlim(-16, 66)
    ax.set_xticks([-10, 0, 10, 20, 30, 40, 50, 60])
    ax.set_xticklabels([label(t) for t in (-10, 0, 10, 20, 30, 40, 50, 60)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("start of a request of 10 s, s")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.97, bottom=0.3)
    save(fig, "fig-16-7-shared-starts.png")


def joint_outcomes():
    """The exact two-sided probability of Listing 17.1 for two models with the
    scores of its transcript, against the number of items that both answer
    correctly: two_sided() for every number that the two scores allow. The
    square is the pair of lists of two_models(), the circle that of
    same_scores()."""
    marks = []
    for a, b in (c17.two_models(2000, random.Random(17)), c17.same_scores()):
        right_a, right_b, n = sum(a), sum(b), len(a)
        marks.append(sum(x and y for x, y in zip(a, b)))
    least = right_a + right_b - n                              # no item that both get wrong

    def p(both):
        return float(c17.two_sided(right_a - both, right_b - both)[1])
    ks = range(least, min(right_a, right_b) + 1)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    ax.axhline(0.05, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    ax.text(ks[-1], 0.054, "5 % level", ha="right", va="bottom")
    ax.plot(ks, [p(k) for k in ks], color=BLACK, lw=0.9)
    for both, marker, name, dx in zip(marks, ("s", "o"), ("first lists", "second lists"), (10, 10)):
        ax.plot([both], [p(both)], marker, color=BLACK, ms=3.6, mfc="white", mew=0.7, zorder=3)
        ax.text(both + dx, p(both) + 0.004, f"{name}, {p(both):.3f}", ha="left", va="bottom")
    ax.set_xlim(least - 4, ks[-1] + 4)
    ax.set_ylim(0, 0.135)
    ax.set_xticks([least, 1520, 1560, 1600, 1640, 1680, ks[-1]])
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_yticklabels(["0", "0.05", "0.10"])
    ax.set_xlabel("items that both models answer correctly, of 2000")
    ax.set_ylabel("exact two-sided p")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.25)
    save(fig, "fig-17-1-joint-outcomes.png")


def unlinked_pairs():
    """Every exact two-sided probability of Listing 17.1 that the scores of its
    transcript allow when 1950 pairs are verified and 50 are not: partly() with
    the counts of main() as they are, with six more pairs verified as wrong for
    both models, and with ten more verified as right for A alone. One dot for
    each table of joint outcomes."""
    rows = (("50 not verified", (0, 0)), ("6 wrong for both", (0, 6)), ("10 for A alone", (10, 0)))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.15))
    ax.axvline(0.05, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    ax.text(0.0502, 2.62, "5 % level", ha="left", va="center")
    counts = []
    for y, (name, (only_a, neither)) in zip((2, 1, 0), rows):
        known = (1560, 132 + only_a, 163, 95 + neither)
        ps = [float(p) for p in c17.partly(known, 1722, 1758, 2000)]
        counts.append(len(ps))
        ax.plot([ps[0], ps[-1]], [y, y], color=GREY, lw=0.7)
        ax.plot(ps, [y] * len(ps), "o", color=BLACK, ms=2.6, zorder=3)
    assert counts == [16, 10, 6]
    ax.set_xlim(0.0425, 0.0545)
    ax.set_ylim(-0.5, 3.0)
    ax.set_xticks([0.044, 0.046, 0.048, 0.050, 0.052, 0.054])
    ax.set_xticklabels(["0.044", "0.046", "0.048", "0.050", "0.052", "0.054"])
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([name for name, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("exact two-sided p for all 2000 items")
    fig.subplots_adjust(left=0.29, right=0.97, top=0.97, bottom=0.32)
    save(fig, "fig-17-2-unlinked-pairs.png")


def batch_labels():
    """What a batch label carries in Listing 17.1: the 50 pairs that are not
    verified lie in two batches of 25, A right on 23 and 7 of them, and up to
    so many answers of B in each batch may carry the label of the other batch.
    For the export with 23 and 12 right answers of B the greatest exact
    two-sided probability that partly() returns for any counts of B that such
    labels can hide, and for the export with 13 and 22 the least."""
    known = (1560, 132, 163, 95)

    def ends(b1, b2, most):
        ps = []
        for j in range(-most, most + 1):
            if 0 <= b1 + j <= 25 and 0 <= b2 - j <= 25:
                rows = [(25, 23, b1 + j), (25, 7, b2 - j)]
                ps += c17.partly(known, 1722, 1758, 2000, rows)
        return float(min(ps)), float(max(ps))

    most = list(range(11))
    first = [ends(23, 12, s)[1] for s in most]
    second = [ends(13, 22, s)[0] for s in most]
    whole = [float(p) for p in c17.partly(known, 1722, 1758, 2000)]
    assert first[0] < 0.05 < first[1] and second[1] < 0.05 < second[0]
    assert (first[0], first[1]) == (second[1], second[0])
    assert first.index(max(whole)) == 6 and second.index(min(whole)) == 10
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    ax.axhline(0.05, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    ax.text(10.2, 0.0503, "5 % level", ha="right", va="bottom")
    ax.plot(most, first, color=GREY, lw=0.7)
    ax.plot(most, second, color=GREY, lw=0.7)
    ax.plot(most, first, "o", color=BLACK, ms=3.2, zorder=3)
    ax.plot(most, second, "s", color=BLACK, ms=3.0, mfc="white", mew=0.7, zorder=3)
    ax.text(6.0, 0.05415, "B right on 23 and 12: greatest p", ha="center", va="bottom")
    ax.text(3.6, 0.0452, "B right on 13 and 22: least p", ha="center", va="top")
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(0.0424, 0.0556)
    ax.set_xticks(most)
    ax.set_yticks([0.044, 0.046, 0.048, 0.050, 0.052, 0.054])
    ax.set_yticklabels(["0.044", "0.046", "0.048", "0.050", "0.052", "0.054"])
    ax.set_xlabel("labels of B in doubt in each batch, at most")
    ax.set_ylabel("exact two-sided p")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.24)
    save(fig, "fig-17-3-batch-labels.png")


def reliability():
    rng = random.Random(18)
    cal = c18.make(2000, 2.0, rng)
    test = c18.make(5000, 2.0, rng)
    temp = c18.fit_temperature(cal)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.74, WIDTH * 0.74))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.4)
    for t, marker, fill in ((1.0, "o", BLACK), (temp, "s", "white")):
        bins = [b for b in c18.bins_of(test, t) if b[0]]
        pts = [(s / n, k / n) for n, s, k, _ in bins]
        for n, s, k, _ in bins:  # 95 % interval of each observed frequency
            lo, hi = wilson(k, n)
            ax.plot([s / n, s / n], [lo, hi], color=BLACK, lw=0.5)
        ax.plot(*zip(*pts), color=BLACK, marker=marker, mfc=fill, ms=3,
                mew=0.6, ls="-" if t == 1.0 else "--")
    ax.text(0.83, 0.42, "as reported", ha="center", fontweight="bold")
    ax.text(0.50, 0.84, f"corrected,\nT = {temp:.2f}", ha="right")
    ax.text(0.33, 0.22, "stated = observed", color="#555555", rotation=45,
            ha="center", va="center", rotation_mode="anchor")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("stated probability")
    ax.set_ylabel("observed frequency")
    fig.subplots_adjust(left=0.19, right=0.96, top=0.97, bottom=0.17)
    save(fig, "fig-18-1-reliability.png")


def temperature_interval():
    """What a hundred labels at one stated logit of 40 settle of the fitted
    temperature, for one, two, three and five negative labels among them."""
    n, logit, limit = 100, 40, 10.0
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    ax.axvline(limit, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    ax.text(limit + 0.25, 3.62, "upper end of the search", ha="left", va="center")
    for row, negative in enumerate((1, 2, 3, 5)):
        y = 3 - row
        low, high = c20.exact_interval(n - negative, n)
        a, b = c18.at_one_logit(logit, high), c18.at_one_logit(logit, low)
        t = c18.at_one_logit(logit, (n - negative) / n)
        ax.plot([a, b], [y, y], color=BLACK, lw=0.8)
        for end in (a, b):
            ax.plot([end, end], [y - 0.14, y + 0.14], color=BLACK, lw=0.8)
        ax.plot([t], [y], "o", color=BLACK, ms=3.2)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["1 negative of 100", "2 negative", "3 negative", "5 negative"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 3.95)
    ax.set_xlim(4, 20)
    ax.set_xticks(range(4, 21, 2))
    ax.set_xlabel("fitted temperature and its exact 95 % interval")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.98, bottom=0.25)
    save(fig, "fig-18-2-temperature-interval.png")


def judge_model():
    """Exact 95 % intervals for the judged pass rate of three samples, against
    the judged rates that the judge of Listing 20.2 can produce on average:
    1 - specificity to sensitivity."""
    samples = ((40, 40), (2, 20), (0, 25))
    low, high = 1 - c20.SPEC, c20.SENS
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3))
    ax.axvspan(low, high, color=PALE, lw=0)
    for edge in (low, high):  # the band keeps its ends if the tint prints pale
        ax.axvline(edge, color=BLACK, lw=0.4, ls="--")
    ax.text((low + high) / 2, 2.62, f"judge model: {low:.2f} to {high:.2f}",
            ha="center", va="center")
    for row, (k, n) in enumerate(samples):
        lo, hi = c20.exact_interval(k, n)
        ax.plot([lo, hi], [row, row], color=BLACK, lw=1.6,
                solid_capstyle="butt")
        ax.plot([k / n], [row], marker="o", ms=3.4, mfc="white", mec=BLACK,
                mew=0.7, clip_on=False, zorder=3)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([f"{k} of {n}" for k, n in samples])
    ax.set_ylim(-0.6, 3.0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.15, 0.5, 0.9, 1])
    ax.set_xticklabels(["0", "0.15", "0.5", "0.90", "1"])
    ax.set_xlabel("judged pass rate (observed, exact 95 % interval)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.28)
    save(fig, "fig-20-1-judge-model.png")


def psi_null():
    rng = random.Random(21)
    sizes = (100, 200, 500, 2000)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    for row, n in enumerate(sizes):
        xs = c21.simulate(n, None, rng)
        lo, mid, hi = (quantile(xs, q) for q in (0.05, 0.5, 0.95))
        ax.plot([lo, hi], [row, row], color=BLACK, lw=1.6,
                solid_capstyle="butt")
        ax.plot([mid], [row], marker="o", ms=3.4, mfc="white", mec=BLACK,
                mew=0.7)
    ax.axvline(0.1, color=BLACK, lw=0.4, ls="--")
    ax.text(0.104, 3.25, 'rule of thumb: 0.1 is "drift"', va="center")
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels([f"n = {n}" for n in sizes])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(0, 0.37)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("PSI of two samples from one distribution "
                  "(median, 5 % to 95 %)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.26)
    save(fig, "fig-21-1-psi-null.png")


def guard_values():
    """The estimate plus its guard band for the record of Chapter 22 with an
    estimate of 194.6 ms: one number from the working values, a range from
    the same record rounded to two significant digits."""
    rec = c22.load("measurement_record.json")
    change = {("result", "value"): 194.6}
    working = float(guard(c22.altered(rec, change))[0][0])
    lo, hi = map(float, guard(c22.altered(c22.displayed(rec), change))[0])
    limit = float(rec["decision"]["limit"])
    fig, ax = plt.subplots(figsize=(WIDTH, 1.25))
    ax.axvline(limit, color=BLACK, lw=0.4, ls="--")
    ax.text(limit + 0.006, 1.62, f"limit, {limit:.0f} ms", va="center")
    ax.plot([lo, hi], [0, 0], color=BLACK, lw=1.2, marker="|", ms=7, mew=0.9)
    ax.plot([working], [1], marker="o", ms=3.6, color=BLACK)
    for x, y, ha in ((working, 1, "center"), (lo, 0, "center"),
                     (hi, 0, "center")):
        ax.text(x, y + 0.3, f"{x:.2f}", ha=ha, va="bottom")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["rounded\nrecord", "working\nvalues"])
    ax.set_ylim(-0.55, 1.95)
    ax.set_xlim(199.79, 200.17)
    ax.set_xticks([199.8, 199.9, 200.0, 200.1])
    ax.set_xticklabels(["199.8", "199.9", "200.0", "200.1"])
    ax.set_xlabel("estimate plus guard band, ms")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.3)
    save(fig, "fig-22-1-guard-values.png")


def one_mix():
    """The 95th percentile of the two periods of Listing 23.2, both at the
    mix on the horizontal axis: at_share() for every thousandth of a percent
    of slow requests from 3.8 % to 4.8 %. The steps are those of the recorded
    requests; nothing is smoothed."""
    rng = random.Random(23)
    before = c23.record(c23.BEFORE, rng)
    after = c23.record(c23.AFTER, rng)
    parts = range(3800, 4801)
    share = [k / 1000 for k in parts]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.55))
    ax.axvline(4.1, color=GREY, lw=0.5, ls=":")
    for rec, style, text, at in ((before, "-", "first period", (3.81, 195.5)),
                                 (after, (0, (3.2, 1.6)), "second period", (3.81, 164.5))):
        ys = [c23.at_share(rec, k, 100_000) for k in parts]
        ax.plot(share, ys, color=BLACK, lw=0.9, ls=style, drawstyle="steps-post")
        y = c23.at_share(rec, 41)
        ax.plot([4.1], [y], marker="o", ms=3.4, color=BLACK, lw=0, zorder=3)
        ax.text(4.115, y + 2.2, f"{y:.0f} ms", ha="left", va="bottom")
        ax.text(*at, text, ha="left", va="bottom")
    ax.set_xlim(3.8, 4.8)
    ax.set_ylim(150, 234)
    ax.set_xticks([3.8, 4.0, 4.1, 4.3, 4.5, 4.7])
    ax.set_xticklabels(["3.8", "4.0", "4.1", "4.3", "4.5", "4.7"])
    ax.set_yticks([160, 180, 200, 220])
    ax.set_xlabel("slow requests in the mix of both periods, %")
    ax.set_ylabel("95th percentile, ms")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.24)
    save(fig, "fig-23-1-one-mix.png")


def one_recovered():
    """The most of the 900 slow requests of Listing 23.2 that can be within
    200 ms when their mean is 450 ms, none took longer than 810 ms, and one of
    them is known to have taken x ms: with_one(), which is most_within() for
    the other 899 at the mean that x leaves them, and one more if x is itself
    within the limit. A filled circle belongs to its line and an open one does
    not."""
    from fractions import Fraction

    def most(x):
        return c23.with_one(x, 450)

    ends = [0, 110, 200, 720, 810]
    grid = [Fraction(k, 4) for k in range(810 * 4 + 1)]
    spans = []
    for a, b in zip(ends, ends[1:]):
        inside = {most(x) for x in grid if a < x < b}
        assert len(inside) == 1, (a, b, inside)
        spans.append((a, b, inside.pop()))
    assert [n for _, _, n in spans] == [530, 531, 530, 531]
    assert [most(x) for x in ends] == [530, 531, 531, 531, 531]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.2))
    for a, b, n in spans:
        ax.plot([a, b], [n, n], color=BLACK, lw=0.9)
        for x in (a, b):
            ax.plot([x], [n], "o", color=BLACK, ms=3.4, zorder=3, mew=0.7,
                    mfc=BLACK if most(x) == n else "white")
    ax.set_xlim(-25, 835)
    ax.set_ylim(529.45, 531.6)
    ax.set_xticks(ends + [400, 600])
    ax.set_yticks([530, 531])
    ax.set_yticklabels(["530, missed", "531, unresolved"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("duration of the one recovered slow request, ms")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.34)
    save(fig, "fig-23-2-one-recovered.png")


def mean_as_printed():
    """The recovered durations of Figure 23.2 that settle the objective as
    missed, by what is known of the mean of the 900 slow durations: 450 ms
    exactly, or a mean that prints as 450.00, 450.0 or 450 ms and may lie half
    a unit of its last digit lower. A duration settles it if with_one() of
    Listing 23.2 at that least mean is 530 at most. A filled circle belongs to
    its line and an open one does not."""
    from fractions import Fraction
    rows = (("exactly 450 ms", Fraction(0)), ("450.00 ms", Fraction(5, 1000)),
            ("450.0 ms", Fraction(5, 100)), ("450 ms", Fraction(1, 2)))
    grid = [Fraction(k, 4) for k in range(810 * 4 + 1)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    seen = []
    for y, (name, half) in zip((3, 2, 1, 0), rows):
        def settled(x):
            return c23.with_one(x, 450 - half) <= 530

        low, high = 110 - 900 * half, 720 - 900 * half  # where 531 first fit
        bars = ([(0, low)] if low > 0 else []) + [(200, high)]
        for x in grid:
            inside = any(a < x < b or x == a == 0 for a, b in bars)
            assert settled(x) == inside, (name, x)
        assert not settled(high) and not settled(200) and (low <= 0 or not settled(low))
        for a, b in bars:
            ax.plot([float(a), float(b)], [y, y], color=BLACK, lw=0.9)
            ax.plot([float(a)], [y], "o", color=BLACK, ms=3.4, zorder=3, mew=0.7,
                    mfc=BLACK if a == 0 else "white")
            ax.plot([float(b)], [y], "o", color=BLACK, ms=3.4, zorder=3, mew=0.7, mfc="white")
            ax.text(float(b), y + 0.22, f"{float(b):g}", ha="center", va="bottom")
            seen.append(float(b))
    assert seen == [110, 720, 105.5, 715.5, 65, 675, 270]
    ax.set_xlim(-25, 835)
    ax.set_ylim(-0.5, 3.75)
    ax.set_xticks([0, 200, 400, 600, 810])
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([name for name, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("duration of the one recovered slow request, ms")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.25)
    save(fig, "fig-23-3-mean-as-printed.png")


def one_or_the_longest():
    """The numbers of the 900 slow requests of Listing 23.2 that can be within
    200 ms, their mean 450 ms, when 720 ms is the duration of one request and
    none is above 810 ms, and when 720 ms is the longest of all 900. The ends
    are with_one() and most_within() of the listing, and every whole number
    from none to each end is tried here against the sums that it allows. The
    dashed line is the 531 requests that the objective needs."""
    total = 900 * 450

    def fits(k, known, cap):
        # k of the other durations at 200 ms at most, the rest above it and at most `cap`
        rest = 900 - len(known) - k
        left = total - sum(known)
        return rest * 200 < left <= k * 200 + rest * cap

    rows = (("one request at 720 ms", c23.with_one(720, 450), [720], 810),
            ("none above 720 ms", c23.most_within(900, 450, 200, 720), [720], 720))
    assert [most for _, most, _, _ in rows] == [531, 467]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.2))
    ax.axvline(531, color=BLACK, lw=0.5, ls=(0, (4, 2)))
    ax.text(524, 1.55, "531 needed", ha="right", va="center")
    for y, (name, most, known, cap) in zip((1, 0), rows):
        assert all(fits(k, known, cap) for k in range(most + 1)) and not fits(most + 1, known, cap)
        ax.plot([0, most], [y, y], color=BLACK, lw=0.9)
        ax.plot([0, most], [y, y], "o", color=BLACK, ms=3.4, zorder=3)
        if most != 531:  # the first line ends on the dashed one
            ax.text(most, y + 0.2, str(most), ha="center", va="bottom")
    ax.set_xlim(-18, 560)
    ax.set_ylim(-0.5, 1.95)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_yticks([1, 0])
    ax.set_yticklabels([name for name, _, _, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("slow requests that can be within 200 ms")
    fig.subplots_adjust(left=0.37, right=0.97, top=0.97, bottom=0.34)
    save(fig, "fig-23-4-one-or-the-longest.png")


if __name__ == "__main__":
    reversed_pairs()
    exchange_gap()
    calibration_age()
    third_host()
    three_host_bound()
    stamps_at_fault()
    aliasing()
    counter_readings()
    empty_record()
    omission()
    two_conventions()
    calibration_readings()
    control_chart()
    request_starts()
    mean_duration()
    counted_where()
    gap_anywhere()
    gap_given_count()
    changing_rate()
    shared_starts()
    joint_outcomes()
    unlinked_pairs()
    batch_labels()
    reliability()
    temperature_interval()
    judge_model()
    psi_null()
    guard_values()
    one_mix()
    one_recovered()
    mean_as_printed()
    one_or_the_longest()
