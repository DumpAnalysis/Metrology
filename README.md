# Metrology in Software Diagnostics, Observability, and Debugging

Code examples for the book *Metrology in Software Diagnostics, Observability,
and Debugging: With Applications to Machine Learning and AI* by Dmitry Vostokov
(OpenTask, ISBN 978-1-919135-15-1). Book Version 61, November 2026.

Repository: https://github.com/DumpAnalysis/Metrology

The listings of the book are these programs, and they print its
transcripts. A few values that the discussion uses beside them, such as a
median read from a figure, a percentile between two printed ones or a
confidence bound on a simulated count, are checked in the tests.
The programs are seeded, use only the Python standard library, and are stored
with their transcripts so that they can be replayed.

A successful replay shows that the book and the programs agree. It does not
show that a model fits your system; each chapter states the model on which its
numbers rest.

## Requirements

Python 3.11 or later. No third-party packages for the examples, the replay
and the tests. Two optional checks compare the Student t quantile with
high-precision arithmetic and need mpmath:
`tools/certify_coverage_factor.py`, for the coverage factor as the
record checker computes it, and `tools/certify_t_quantile.py`, for the
quantile function with a float tail.

Tested with CPython 3.11, 3.12 and 3.13 on Linux (x86-64). Under all three,
every `exact` example reproduces its transcript within the tolerance below and
the unit tests pass. The `live` examples are checked for the form of their
output only; the numbers in their transcripts come from one run on one
machine.

Drawing the figures is separate and needs matplotlib and Pillow, pinned in
`figures/requirements.txt`:

```
python -m pip install -r figures/requirements.txt
python figures/make_figures.py
```

## Running

From the repository root:

```
python -m examples.ch05_propagation     # run one example
python run_all.py                       # replay all examples against expected/
python run_all.py ch08 ch09             # replay the examples whose names match
python run_all.py --update              # rewrite the transcripts in expected/
python run_all.py --lint                # check the 50-column rule
python -m unittest discover -s tests    # unit and regression tests
```

## Two modes of replay

Each example declares its mode in its first line.

| Mode | Meaning | What `run_all.py` compares |
|------|---------|----------------------------|
| `# replay: exact` | Seeded and deterministic. | Text identical; every number written in the notation of the transcript (an exponent or none, no added plus sign or leading zeros) and printed to the same place; a whole number identical, and a number with decimals within one unit of its last place. |
| `# replay: live` | Measures the machine it runs on. | The form of the output only: numbers are masked, and lines starting with `env:` are ignored. |

The tolerance of the exact mode allows for differences in the last bit of
elementary functions between mathematical libraries. It applies to numbers
printed with decimals. A count is printed as a whole number and must be
identical, so that one event and none are never the same transcript. The live examples
(`ch06_clock`, `ch07_probe_effect`, `ch11_measurand`, `ch19_inference_timing`)
will print different numbers on your machine; that is what they are for.

## Contents

| Path | Contents |
|------|----------|
| `swmetro/` | Measurement results, Type A and Type B evaluation, uncertainty budgets (`core`); normal and Student t helpers (`dist`); law of propagation and Monte Carlo (`propagate`); small statistics (`stats`); units with decimal and binary prefixes (`units`); checks of a measurement record (`record`). Printed in Appendix B. |
| `examples/` | One or more programs for each of Chapters 2 to 23. |
| `expected/` | The transcript of each example, as printed in the book. |
| `data/` | The trace fixture of Chapter 16 (`app_trace.log`) and its seeded generator (`make_trace.py`); the measurement record of Chapter 22 (`measurement_record.json`), a blank template with the same fields (`record_template.json`), a description of the form, of the decision procedure and of what the checker examines (`RECORD_FORMAT.md`), and the program that writes both (`make_record.py`). |
| `figures/` | `make_figures.py` draws the book's thirty-two figures from the same functions as the examples. Needs matplotlib and Pillow at the versions in `requirements.txt`; output goes to `figures/out/`. |
| `tests/` | `test_swmetro.py`: unit tests of the package. `test_review.py`, `test_review2.py` to `test_review5.py`, `test_review7.py` to `test_review46.py` and `test_review48.py` to `test_review60.py`: regression tests, numbered by the version of the book whose review they followed; each checks the code against an independent fact (a closed form, a definition evaluated by brute force, a published table, a simulated coverage), not against saved output. `test_release.py`: the license text is the published one, and the record and this file name the repository and the release. |
| `tools/` | Two optional checks of the Student t quantile against high-precision arithmetic. They need mpmath. |
| `run_all.py` | The replay program. |
| `LICENSE` | PolyForm Strict License 1.0.0. |
| `.github/workflows/replay.yml` | Runs the tests, the line-width check and the replay on Python 3.11 to 3.13. |

## Examples by chapter

| Chapter | Example | Mode | Subject |
|---------|---------|------|---------|
| 2 | `ch02_result` | exact | From ten indications to a result with a budget |
| 2 | `ch02_targets` | exact | One record, three targets: record, process, next observation (with the coverage of the prediction procedure) |
| 3 | `ch03_scales_units` | exact | Ordinal means; binary and decimal prefixes |
| 4 | `ch04_accuracy` | exact | Trueness and precision; the zero-variance trap; coverage of a rectangular distribution |
| 5 | `ch05_propagation` | exact | Law of propagation and Monte Carlo, with a factor shared by two durations as two correlated inputs and as one, one constant error of the rate of a clock shared by two stamps, which leaves one part in 10**9 of the interval between them wherever it lies, and an exact check of the nonlinear case; three correlated stages whose correlations have to hold together |
| 6 | `ch06_clock` | live | Declared and observed properties of clocks |
| 6 | `ch06_exchange` | exact | The clock offsets that matched exchanges between two hosts allow, what they leave of the order of two events, for how long after the exchanges, and with stamps that are written late; in exact fractions, whatever the origin of each clock; a third host that settles an order or contradicts the two, the bound on every stamp that each bears, and the stamps of which one alone can explain a record that fits no offsets, with their number as one stamp moves |
| 6 | `ch06_ordering` | exact | Ordering events stamped by two clocks; ten pairs under clock errors drawn anew for each pair and under one offset kept through a capture |
| 7 | `ch07_probe_effect` | live | Probe indication, blank indication, overhead |
| 8 | `ch08_aliasing` | exact | Aliasing of a scraped gauge |
| 8 | `ch08_counter` | exact | A counter that restarts, with and without a visible decrease, and one that wraps around; the increases that two readings of a counter of fixed width allow, also when the gap between them was rounded |
| 8 | `ch08_sampling` | exact | Profiler shares; counts from sampled traces, with one retention decision for each event or for each unit of several, equal or not; the chance of an empty record; the count and its standard uncertainty estimated from one capture; the number of units that an empty record still allows, with independent decisions and with a fixed quota |
| 8 | `ch08_selection` | exact | Sampling that depends on the outcome; what weighting recovers, and how much the weighted estimates vary over retention draws |
| 9 | `ch09_histogram` | exact | Quantiles by two conventions, from buckets, and from a sketch; an empty histogram has no quantile (NaN), and a NaN or a negative value is no observation |
| 9 | `ch09_dependent_tail` | exact | A tail quantile of a bursty trace: iid and block bootstrap, with the sensitivity to block length; how many independent values a finite upper end for a percentile needs |
| 10 | `ch10_coordinated_omission` | exact | Latency from actual and intended send times, in whole nanoseconds, for a tester that sends at the first scheduled moment not before the response, with interpolated percentiles; one percentile under two conventions; the requests counted against an objective, also for a server that is slower while a request of the run waits |
| 10 | `ch10_censoring` | exact | A latency report that drops timeouts; a percentile that is only a bound |
| 11 | `ch11_measurand` | live | One list, four sizes |
| 11 | `ch11_leak_rate` | exact | Slope of a memory series and its detection limit |
| 12 | `ch12_calibration` | exact | Calibrating a latency probe; the three sources of the uncertainty of a corrected delay, and what further indications and further calibration readings would each buy |
| 13 | `ch13_variance_components` | exact | Within-launch and between-launch variance for a balanced design, and for launches of unequal length; designs within a budget of time |
| 14 | `ch14_decisions` | exact | Guarded acceptance; regression rules |
| 15 | `ch15_control_chart` | exact | Shewhart and CUSUM charts, a confirmation rule; average run lengths |
| 15 | `ch15_run_length` | exact | Exact run lengths of the Shewhart rule, with known limits and with the limits that the baseline of `ch15_control_chart` gave |
| 15 | `ch15_detection_limit` | exact | Critical value and detection limit |
| 16 | `ch16_trace_measurements` | exact | Time Delta of a request with one start and one end, Statement Current in windows measured from the start of the capture, an upper bound for the current of a window without messages, period, completeness counted against the readings of the tracer's counter, and the stretches of time for which sequence numbers, counter readings and time stamps that do not run backwards verify the record, each with the sequence numbers of the lines that belong to it |
| 16 | `ch16_timing_models` | exact | Truncation error of a duration; period estimators under two noise models |
| 16 | `ch16_silence` | exact | What a silence establishes: the bound under two models of what writes the messages, the start times of requests that leave a message in the verified stretches (requests longer than a window, records with gaps, requests of several durations, durations sampled under four rules of selection, and a mix of durations estimated from a sample of requests, with the risk divided between the sample and the silence), the count of messages among the lines of the verified runs (by sequence number; from time stamps alone, the count where a line is alone with its stamp and a range where lines share one), the bound for one or more messages of a paired source, stretches that overlap, the first count above zero with exact limits, verified time and retention, an empty record with and without counter readings, a recorder that keeps one message in four by the message or by the request, an empty window chosen in advance or found by a search, for single messages and for requests that write two, a gap found anywhere in the record, and the chance of a gap given the number of messages, for a Poisson process of constant rate and for one whose reference rate changes within the record, a silence judged against a second, independent record in place of a rate, and the same requests read in two runs |
| 16 | `ch16_bounds` | exact | Bounds on a failure share from an incomplete capture, and the decision they allow; the ranks that bound a percentile when some latencies are unknown; a percentile objective decided from completions, timeouts and lost records; what one more histogram bound would establish |
| 17 | `ch17_accuracy` | exact | Intervals for accuracy; paired comparison in percentage points, with the correlation of the right answers and the exact test for the pairs that differ, one-sided and two-sided, for two pairs of lists with the same scores; the limits of that test that two scores alone allow, for a lead of either model and for a tie, with some pairs verified, with batches that keep the other pairs together, and with the counts that one doubtful label in each batch can hide; probabilities as exact fractions |
| 17 | `ch17_seeds` | exact | Variation of a score across seeds and splits |
| 18 | `ch18_calibration` | exact | Temperature scaling, with the loss taken from the logit; bias of the ECE estimator; a fit with one confident error, and the interval that a hundred labels leave for its temperature, for a logit of either sign |
| 19 | `ch19_float` | exact | Floating-point resolution, among it stamps that count nanoseconds from 1970 and lie 1 ns apart; accumulation, order |
| 19 | `ch19_inference_timing` | live | Cold start and steady state |
| 20 | `ch20_error_bars` | exact | Clustered and paired standard errors; exact tests of a gain with the labels of the two models exchanged by question and by passage |
| 20 | `ch20_judge` | exact | A judge model with known error rates; a corrected rate below zero by sampling alone, shown unconfined, with an exact interval; samples whose interval has no rate in common with the judge model; a judge whose verdicts go against the reference; Cohen's kappa at another share of reference passes |
| 20 | `ch20_raters` | exact | Reference labels that err: rater noise, task sampling, shared error |
| 21 | `ch21_psi` | exact | The population stability index under no drift, with its shares taken over the bins of the two samples; its asymptotic critical value checked by simulation |
| 22 | `ch22_budget` | exact | A budget for a percentile; what bucket counts establish |
| 22 | `ch22_record_check` | exact | Checking a record: missing inputs, values that cannot be, careless edits and the stale outcome that each leaves behind, working values and rounded ones near the limit, comparability with a second record |
| 22 | `ch22_runs` | exact | Which p95 across runs: long-run mean, next run, pooled requests; three sets of runs with one equally weighted mean of run percentiles, and what weighting by requests and pooling give instead |
| 23 | `ch23_case` | exact | A diagnostic case: workload, instrument, or service; replay through four configurations; an empty window shown as no observations, with an upper bound for its arrival rate, and exact limits for a single request; power, bound, and the decision against a declared margin |
| 23 | `ch23_mix` | exact | A change of request mix: observed and standardized means and percentiles; the change of the mean and of the 95th percentile divided between the classes and the mix, in both orders; a class of the reference mix without observations, and what such records still decide, alone, with a mean and an upper limit of the lost durations, with one of them recovered, with the mean known only as it prints, and with a duration known as that of one request or as the longest of all; a request whose duration is not known, or known only by a bound; both periods at one mix that neither had, and what one request decides there; a class that only the later period has, the one order that its records determine for the mean, and a percentile that is determined without its durations |

## Quantile conventions

A quantile of a finite sample is defined by a convention, and
`swmetro/stats.py` has three. An objective on a percentile must name one, and
the raw data, the histogram and the record must then follow it.

| Function | Convention | Used by |
|----------|------------|---------|
| `ecdf_quantile`, `ecdf_rank` | Inverse empirical CDF: the smallest value with at least a share q of the values at or below it. The quantile that cumulative counts determine. | `ch02_targets`, `ch10_censoring`, the last line of `ch10_coordinated_omission`, `ch22_budget`, `ch22_runs`, `ch23_case`, the measurement record. With weights it is `weighted_quantile`, which sums them as fractions: `ch08_selection`, `ch23_mix` |
| `quantile` | Linear interpolation around rank q(n - 1). | `ch09_histogram` (first column), `ch09_dependent_tail`, `ch10_coordinated_omission`, `ch21_psi` |
| `order_stat` | Lower order statistic at rank floor(q(n - 1)), the definition under which the sketch guarantee is stated. | `ch09_histogram` (second column) |

A probability is read exactly before any rank is computed (`exact_q`). A
float is taken as the decimal number that Python prints for it, so that
`0.95` means 95/100 and not the binary number nearest to it; a `Fraction`, a
`Decimal`, an `int` or a string is taken as it stands. Ranks are then found
in integer arithmetic. The two readings differ at boundaries:
`ecdf_rank(0.9, 10)` is 9, and `ecdf_rank(Fraction(0.9), 10)` is 10, because
the stored number for 0.9 lies above nine tenths.

A duration enters the examples through `known()` of `swmetro/stats.py`: a
finite real number (an `int`, a `float`, a `Fraction` or a `Decimal`), kept
in its type. A NaN, an infinity and a number beyond the range of a float are
refused as missing, and a Boolean flag, a text and `None` as no numbers.
Where an entry that is no duration has a meaning, the listing says so:
`None` is a timeout in `ch10_censoring`, whose completions lie below the
deadline of the client, and `None`, a NaN, an infinity and a bound such as
`(">=", 1000)` keep their meanings in `ch23_mix`.

A bound that an example prints is rounded to its safe side, an upper bound
upward and a lower bound downward, by `upward()` and `downward()` of
`swmetro/stats.py`, in exact arithmetic. A limit is compared with the bound
as it was computed, not with its printed form, and the bound is computed
without rounding: in `ch23_case` a change of a percentile and an upper bound
are exact differences and sums, a float only where a float holds the result,
so that a bound equal to a margin, or just above it, is not reported as below
it. The two fields of a row are added as floats where both are floats, since
that sum is the duration a configuration records, and exactly otherwise, so
that a decimal duration on a bucket bound stays on it.

A `Result` of `swmetro/core.py` takes its value, its uncertainty, a limit and
a level of coverage as an `int`, a `float`, a `Fraction` or a `Decimal`. The
distance of a limit from the value, in units of the uncertainty, is formed
from the exact numbers and rounded once, so that one limit gives one
probability in whatever type it is written. The limits of an interval are
floats where the value is a float or a float holds it, and exact fractions
otherwise.

## Using the measurement record

`data/record_template.json` is a blank record, and `data/RECORD_FORMAT.md`
describes the form. A `null`, an empty string or a prompt in angle brackets
stands for unknown, and `swmetro/record.py` never reads any of them as a
number:

```
from pathlib import Path
from swmetro.record import read, missing, invalid, combined, guard, decide, stale, comparable, in_unit
record = read(Path("data/measurement_record.json").read_text(encoding="utf-8"))
missing(record)         # inputs of the decision that are unknown
invalid(record)         # values that cannot be; statements that contradict
combined(record)        # bounds of the combined u that the whole record supports
guard(record)           # bounds of the estimate plus, and less, the guard band
decide(record)          # a status, with reasons when it is not a decision
stale(record)           # the stored outcome, if it is known and is not that status
comparable(a, b)        # reasons why two records are not one quantity; unit notes
comparable(a, b, across=("conditions",))   # the same, with the load under study; nothing else can be
in_unit(record, "ms")   # value and u of the result, converted explicitly
```

Open a record with `read`, not with `json.load`. The default reader turns a
number written with a decimal point or an exponent into the nearest float
before any check can see it: a reading written as `9007199254740993.0`
comes back as `9007199254740992.0`, and against a limit of
9007199254740992 it would be accepted. `read` keeps such a number as the
decimal that is written (`decimal.Decimal`) and an integer as an integer,
and it refuses a name that occurs twice in one object and the bare words
`NaN`, `Infinity` and `-Infinity`, which are not JSON. `combined`, `guard`
and `in_unit` return decimals too, so that what they show is the number of
the record and not the float nearest to it.

`decide` returns `accept`, `reject` or `undecided` under the one procedure it
supports, a guarded decision against an upper limit with a Student t
coverage factor. It returns `invalid` for a record with a value that cannot
be, such as a negative uncertainty, and `unresolved` for a record that
supports no decision as it stands: an input is unknown, the procedure is
another one, components are correlated, a required component was never
evaluated, or records were lost.

The guard band uses the uncertainty that the checker combines from the
components of the budget (`combined(record)`); a combined uncertainty stated
in the record is checked against it and never used in its place. A record
holds working values (`"significant_digits": "all"`). One that holds only
rounded numbers says how many significant digits they have and to what
increment its estimate was rounded (`value_step`), and the checker
then reads each number as the interval of the values that round to it,
keeps the range in which the components and the stated combined uncertainty
both hold, and gives a decision only if the whole range agrees; otherwise the
status is `unresolved`, with a request for the working values. The estimate and the
degrees of freedom are taken as stated, and the outcome that a record states
is not read by `decide`: `stale` reports it where it is not the status found.

The comparison with the limit is made on exact fractions of the numbers as
the record states them, and never on sums of floats, so that a record
decides the same in milliseconds and in seconds: 100 ms with a guard band of
50 ms meets a limit of 150 ms, and 0.1 s with 0.05 s meets 0.15 s, although
`0.1 + 0.05` is `0.15000000000000002`. Only the coverage factor is computed.
Above a risk of one quarter it is computed from the distance of the risk
from one half, which is kept exactly, so that a risk such as
0.49999999999999999999 has a small factor and not, as its float would give,
none. The checker allows one part in a million for the accuracy of the
factor, and a working record whose guarded value comes that close to the
limit is `unresolved` (`at the limit: the coverage factor decides`).
`data/RECORD_FORMAT.md` has the rules, and a note on records of earlier
formats.

These are checks of consistency. A record that passes them is not thereby a
record whose model fits the system.

## The 50-column rule

Lines in `swmetro/`, `examples/`, `expected/` and
`data/measurement_record.json` are at most 50 characters long, so that
listings and transcripts print without wrapping on the book's page.
`run_all.py --lint` checks it. The listings in the book are these files; the
two-line `if __name__ == "__main__":` guard that ends each example is omitted
in print.

## Data

All data are synthetic and generated by the programs themselves. The trace
`data/app_trace.log` can be regenerated with `python data/make_trace.py`, and
the measurement record with `python data/make_record.py`. The record carries a
hash of the source files that produced it, the writing program included, and
a short hash of each file, so it must be regenerated after any change to
them; a test fails otherwise.

## Versions

The files of this repository are those of one version of the book, which the
first lines of this file name; the history of the repository holds the files
of the versions before it. Seeds alone do not specify a result: the stored
transcripts, the trace fixture and the measurement record are part of the
repository for that reason.

These are the files for Version 61 of the book.

## License

The contents of this repository are licensed under the
PolyForm Strict License 1.0.0. The full text is in `LICENSE`, unchanged from
the text published by the PolyForm Project at
https://polyformproject.org/licenses/strict/1.0.0.
