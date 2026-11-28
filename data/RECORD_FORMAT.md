# The measurement record and its checker

`measurement_record.json` is the record of the Chapter 22 result,
`record_template.json` is the same form with nothing filled in, and
`swmetro/record.py` checks records of this form. This page says what the
form is, which decision procedure the checker supports, and what it does and
does not examine. The format is `measurement record, version 5`; the checker
decides nothing for a record that names another.

## Conventions

- **One unit.** Every dimensional value in a record is in `measurand.unit`:
  the estimate, the uncertainties, the stored step, the corrections, the
  limit. Risk, degrees of freedom and counts have no unit.
- **Numbers as written.** The checker reads each number as the decimal that
  the file states, and an integer as it stands, and keeps it as an exact
  fraction: `0.15` is 3/20 and not the binary number nearest to it, and
  `9007199254740993` keeps its last unit. This begins when the file is
  opened (see *Opening a record*). A record converted to another unit must
  state the converted decimals. The product of two floats can differ from
  them in its last digit, and it is then another record.
- **Unknown.** A `null`, an empty string, or a prompt in angle brackets
  stands for unknown. The checker never reads unknown as a number. Zero is a
  statement: `"records_lost": 0` says that nothing was lost.
- **Working values.** A record is passed between programs, and it holds the
  numbers as they were computed: `result.significant_digits` is `"all"`.
  Rounding belongs to the report that a person reads. A record that holds
  only rounded numbers must say so, by giving the number of significant
  digits instead (see below).
- **Kinds.** Each group of values says whether it was `observed`, `derived`
  from other values in the record, or `stipulated`.
- **One clock.** `capture.clock` names the one clock that stamps both events
  of the interval. A relation to another clock is an entry of
  `capture.clock_relations` that names both clocks, says whether it concerns
  their `rate` or their `offset`, and says whether it is `established`. An
  interval whose two events are stamped by different clocks is not described
  by this form.

## Opening a record

Open a record with the reader of the module:

    from swmetro.record import read
    record = read(path.read_text(encoding="utf-8"))

`json.load` and `json.loads` with their defaults turn every number that is
written with a decimal point or an exponent into the nearest float, and they
do it before any check can see the number. A float is a binary number and
may be another number than the one in the file:

| Written in the file | Default reader | `read` |
|---------------------|----------------|--------|
| `9007199254740993` | `9007199254740993` | `9007199254740993` |
| `9007199254740993.0` | `9007199254740992.0` | `Decimal('9007199254740993.0')` |
| `9.007199254740993e15` | `9007199254740992.0` | `Decimal('9007199254740993')` |
| `0.10000000000000001` | `0.1` | `Decimal('0.10000000000000001')` |
| `1e-400` | `0.0` | `Decimal('1E-400')` |

Against a limit of 9007199254740992 the first three readings are one unit
too high, and a checker that was handed the float of the second or third
would accept them. It cannot see what the reader dropped. `read` keeps such
a number as a `decimal.Decimal`, the decimal that is written, and an integer
as an integer. Validation and the arithmetic of the decision take these
decimals as they take integers and floats.

Five further points.

- A record is an object. A text that is JSON and no object, `[]` or `null`
  or a number, is refused with a `ValueError`: the grammar of JSON has such
  texts (RFC 8259, section 2), and this form does not. A value built in a
  program that is no object is `invalid` (`the record is not an object`).
- A record built in a program may hold floats. A float is read as the
  decimal that Python prints for it, the shortest that identifies it, so a
  record that Python wrote and `read` opens again holds the same numbers.
- The digits of a number are read exactly, but its range is not extended. A
  number must be of a size that a float can hold: at most about 1.8e308,
  and, unless it is zero, at least about 5e-324. A number outside that range
  is reported as not a finite number, and the record is `invalid`. The last
  row of the table is such a number.
- A name that occurs twice in one object is refused with a `ValueError`. The
  default reader would keep the last value and say nothing of the first.
- The bare words `NaN`, `Infinity` and `-Infinity` are refused with a
  `ValueError`. The default reader takes them for numbers, and the grammar
  of JSON has no such values (RFC 8259, section 6). In a record that was
  built in a program and never was a file, such a value is `invalid` where a
  number is expected and where a name is.

The module reads records and does not write them. `json.dumps` writes a
float as Python prints it, which `read` returns unchanged, but it does not
accept a `Decimal`. A program that alters a record it has read and stores
it again has to write each decimal as it stands.

## Sections

| Section | Contents |
|---------|----------|
| `measurand` | Quantity, quantile convention, target across runs, conditions, start and end events, unit. |
| `data` | Origin of the data: capture or simulation, runs, requests, and how the runs are weighted (`run_weighting`). |
| `capture` | The clock and its relations to other clocks, the stored step, the sampling rule, the count of lost records. |
| `components` | The budget: for each component its name, evaluation type (A or B), kind, standard uncertainty `u`, degrees of freedom where finite, and source. Each `u` is the component's contribution to the result, in the unit of the record. A correction is a component that also has a `value`. |
| `covariances` | Correlations between components. An empty list says that none were entered; in the template the list is `null`, which says nothing. |
| `result` | Estimate, combined standard uncertainty, effective degrees of freedom, the significant digits of these numbers, a 95 % interval, and the method. |
| `decision` | The procedure, the limit, the rule and coverage factor in words, the risk and what kind of probability it is, and the outcome. |
| `assumed` | What the result takes for granted. The decision is conditional on these. |
| `not_claimed` | What lies outside the target or outside the claim. |
| `not_evaluated` | Required components that were never evaluated. |
| `sources`, `replay` | Repository, release, the files that produced the record with their hashes, and how to reproduce it. |

The three lists at the end are different statements and are treated
differently. An entry of `assumed` or `not_claimed` qualifies a decision
without preventing it. An entry of `not_evaluated` prevents it.

## Working values and rounded values

`result.significant_digits` is either `"all"` or a whole number from 1 to 17.

- `"all"`: the uncertainties, the estimate and the degrees of freedom are
  working values. The checker combines the components and requires
  `result.u` to be that number, up to the last bits of the floating-point
  arithmetic that produced it (a relative difference of 1e-12).
- A number *n*: the uncertainty of every component, `result.u` and
  `result.dof` were each rounded to *n* significant digits, and the estimate
  was rounded to the increment that the record states in
  `result.value_step`. The checker then reads each of these numbers as the
  interval of the values that round to it, and carries the intervals through
  the calculation.

A rounded record must state `result.value_step`: 0.1 if the estimate was
rounded to tenths, and 0 if it was not rounded at all. The increment is a
number of the format like any other, and an increment above zero that is
too small to differ from zero in a float is not one: an estimate rounded at
a place below 1e-323 cannot be recorded as rounded. A report rounds an
estimate to the place of the last digit of its uncertainty, and an estimate
with an uncertainty of zero has no such place: it is either left as it is,
with an increment of 0, or rounded to some place that the record then has to
name. An estimate of 100.000 with an increment of 0.001 stands for 99.9995
to 100.0005, and against a limit of 100 it decides nothing. A record of
working values states no increment, or 0.

The interval of an uncertainty or of the degrees of freedom runs from the
midpoint to the next lower number of *n* digits up to the midpoint to the
next higher one. That is half a unit of the last digit either way, except at
a power of ten, below which the numbers of *n* digits lie ten times closer:

| Written, two digits | Stands for |
|---------------------|------------|
| 9.6 | 9.55 to 9.65 |
| 9.9 | 9.85 to 9.95 |
| 10 | 9.95 to 10.5 |
| 11 | 10.5 to 11.5 |
| 0.50 | 0.495 to 0.505 |

The estimate is rounded to a fixed increment, and its interval is half that
increment either way. The ends of an interval are included, since which
neighbor a midpoint rounds to depends on the rule for ties.

These bounds are decimal numbers, and the checker keeps them as exact
fractions. Whether two statements can both be true is decided on the squares
of the bounds, without a square root and without a tolerance, so that a gap
in the twelfth digit is a gap. The decision is made on the same fractions
(see *The comparison at the limit*). Floating-point arithmetic enters in one
place only, the coverage factor.

Two statements bound the combined uncertainty of a rounded record: the
components, whose intervals combine to a range, and `result.u`, which has an
interval of its own. If the two do not meet, they cannot both be true and the
record is `invalid`: a component shown as 9.6 cannot have a combined
uncertainty shown as 10. If they meet, the combined uncertainty lies where
both hold, and that intersection, not the wider range of the components
alone, is carried into the guard band. The decision is given only if every
combination of values in the intervals leads to the same outcome; otherwise
the status is `unresolved` with the reason
`rounding decides: wants working values`. A rounded record can therefore
still decide, and when it does, more digits could not change the outcome.

The limit and the risk are requirements, not results, and are read as
written in both cases. A value of exactly zero is read as zero.

In both cases the guard band uses the uncertainty that the checker combines
from the components. A stated `result.u` is checked against it. It never
takes its place, and in a rounded record it serves only to narrow the range
that the components allow.

## The supported decision procedure

The checker supports one procedure, named in `decision.procedure`:

    upper limit, t guard band, v1

It decides whether a quantity is at most a limit, with a guard band:

- `accept` if `value + k u <= limit`;
- `reject` if `value - k u > limit`;
- `undecided` otherwise.

Here `value` and `dof` are those of `result`, `u` is the root sum of squares
of the components, `limit` and `risk` are those of `decision`, and `k` is the
quantile of Student's t distribution at probability `1 - risk` for `dof`
degrees of freedom. The risk is a probability under the model assigned to the
result. It is not a long-run rate of false acceptance.

A record that names this procedure must also state it in words, and the
checker compares the words exactly:

| Field | Required text |
|-------|---------------|
| `result.method` | `budget, Student t interval` |
| `decision.rule` | `accept if value + k u <= limit` |
| `decision.k` | `t quantile at 1 - risk, result dof` |
| `decision.risk_is` | `probability under the model` |

The degrees of freedom must be finite. A result that rests on the normal
approximation would need another procedure, and there is none here.

## The comparison at the limit

The rule accepts at equality, so a record whose guarded value is the limit
must be accepted, in whatever unit it is written. Sums of floats do not
provide that: `0.1 + 0.05` is `0.15000000000000002`, which is above `0.15`,
while `100 + 50` is `150`. The checker therefore never adds floats. It forms
the headroom, `limit - value`, and the square of the combined uncertainty as
exact fractions of the numbers as written, and compares the headroom with
the guard band on the squares:

- `accept` if the headroom is at least zero and `headroom^2 >= k^2 u^2`;
- `reject` if the headroom is below zero and `headroom^2 > k^2 u^2`;
- `undecided` otherwise.

A change of unit multiplies both sides by the same power of ten and cannot
change the outcome. With `u = 0` the comparison is of the estimate with the
limit, digit for digit, and an integer of any size is compared as an
integer. In a rounded record the two ends of each interval are compared in
the same way, and a decision is given only if both ends agree.

The coverage factor `k` is the one number that is not stated. It is
computed, and how it is computed depends on the risk, which the checker
holds as an exact decimal.

- Up to a risk of one quarter the factor is found from the upper tail,
  `t_isf(risk, dof)`.
- Above a risk of one quarter it is found from the distance of the risk
  from one half, `t_mid(1/2 - risk, dof)`. That distance is formed exactly
  before any float is made. A float of the risk itself can lose it
  altogether: 0.49999999999999999999 is 0.5 as a float, and a factor
  computed from that float is zero, which would turn a guard band into
  none. From the distance, the factor for that risk and one degree of
  freedom is about 3.14e-20, and a record at its limit with a positive
  uncertainty is `undecided`, as the rule requires.

Wherever `swmetro/dist.py` was checked, it returns the factor within parts
in 10^13 for up to a hundred degrees of freedom and within 1.3 parts in
10^9 at worst, at 10^6 degrees of freedom, at every risk. The tests compare
it with the closed forms for 1, 2 and 4 degrees of freedom, with the two
methods against each other at a risk of one quarter and at 10^6 degrees of
freedom, and with a series next to the center. The optional program
`tools/certify_coverage_factor.py` brackets what the checker uses in
80-digit arithmetic, for exact risks from 1e-300 to 1/2 - 1e-300.

The checker allows more than that: one part in 10^6 (`DELTA`), at every
risk. (Version 13 divided this allowance by `1 - 2 risk`, because the factor
next to one half was then computed from a float of the risk and known less
well. That reason is gone, and the division with it.) A decision is given
only if it holds for every factor within the allowance. A working record
whose guarded value comes that close to the limit is `unresolved`, with the
reason `at the limit: the coverage factor decides`: the outcome would turn
on the last digits of a quantile. The allowance never admits a record. It
only withholds a decision.

Four cases stand apart.

- With one degree of freedom and a risk of 0.25 the factor is known: it is
  1. The allowance is zero, and equality is decided exactly. An estimate of
  100 ms with a standard uncertainty of 50 ms is accepted against a limit of
  150 ms, and so is 0.1 s with 0.05 s against 0.15 s.
- A factor beyond 10^150 is not computed, and the search for a factor may give
  up from 5 × 10^149 on. Very few degrees of freedom produce such a factor,
  and so does a very small risk with fewer than two degrees of freedom. A
  factor that is not computed is known to exceed 10^149 (`FAR`), and it has no
  upper bound here. (With fewer than one degree of freedom Version 14 gave up
  earlier, at 10^150 sqrt(dof). The optional program verifies next to the end
  of the range that the bound holds and that no search gives up below 5 ×
  10^149.) Nothing is accepted or rejected unless `u` is zero. The status is
  `undecided` where the known part of the band, 10^149 `u`, already reaches
  past the limit on the one side and back to it on the other, which it does
  for any uncertainty of ordinary size. Otherwise the status is `unresolved`,
  with the reason `coverage factor not computed`: the band may be finite and
  small, and arithmetic that reached further could settle the record. With one
  degree of freedom, a risk of 1e-160 and a standard uncertainty of 1e-160 ms,
  the factor is about 3.18e159 and the band 0.318 ms. Against a limit of 100
  ms the rule would accept an estimate of 99 ms and reject one of 101 ms. The
  checker returns `unresolved` for both and `undecided` for an estimate of 100
  ms, which the sign of the band settles. (Version 14 returned `undecided` for
  all three.)
- With fewer than 0.001 degrees of freedom and a risk above one quarter the
  factor is not computed either, because the computation from the distance
  was not checked there. It is then known only to exceed twice that
  distance: no density of Student's t is above 1 / sqrt(2 pi) at zero, so
  the factor is at least sqrt(2 pi), about 2.5, times the distance. That
  is enough to call a record at its limit `undecided`. Further from the
  limit the status is `unresolved`, with the same reason,
  `coverage factor not computed`.
- A risk within 1e-300 of zero or of one half is not supported
  (`risk not supported`). A float is too coarse a probability there.

## What the checker shows

`guard(record)`, `spread(record)`, `combined(record)` and
`in_unit(record, unit)` return `decimal.Decimal` numbers. They are for
display, and `decide` does not use them. `guard` is for a record that has
passed the checks listed below. For any other it raises `ValueError`, since
no decision is formed and no number is compared with the limit:
`stopped(record)` returns the status and the reasons of such a record, as
`decide` does, and `None` for one that has passed.

- A sum or a product of stated numbers is shown exactly, in full, however
  many digits it has. With no guard band the guarded value is the estimate
  as it is written: `9007199254740993` and not the float
  `9007199254740992.0`, and a decimal of 4301 places with all of them.
- The guarded values are bounds of the four numbers that `decide` compares
  with the limit. A computed coverage factor enters with its allowance of
  one part in a million, lower at a lower bound of the band and higher at an
  upper one. With a factor that is known, and an uncertainty whose root is a
  fraction, the band is exact and is added to the estimate exactly: 199 with
  an uncertainty of 1.00000000000000004, one degree of freedom and a risk of
  0.25 gives `200.00000000000000004`. Each bound lies on the side of the
  limit, or on the limit, where `decide` finds it.
- A square root that is a fraction is that fraction, however many digits it
  has. One that is not is bounded with 40 digits, downward for a lower bound
  and upward for an upper one, and with more where a bound next to the limit
  needs them. The root is taken in whole numbers. No float is formed on the
  way, so a single uncertainty of 1e-200 is shown as `1E-200` and one of
  1e200 as `1E+200`; their squares would be 0 and infinity as floats. No
  decimal context rounds on the way either: the decimal settings of the
  caller have no part in what is shown or decided.
- A factor that is not computed is shown by what is known of it: the
  estimate plus the lower bound of the band at one end, and `Infinity` at
  the other; `-Infinity` and the estimate less that bound for the lower
  guarded value.
- A conversion of unit applies the exact ratio of the two units to the
  number as written: 0.2414 s is `241.4` ms. Every ratio is a power of two
  times a power of ten, so the product is a decimal, and it is returned in
  full: 0.2 s and 10^-41 s more is 200 ms and 10^-38 ms more, above a limit
  of 200 ms as it is above 0.2 s.

Formatting works as for floats (`f"{x:.2f}"`), and `float(x)` gives the
nearest float where one is wanted.

## What `decide(record)` returns

A status and a list of reasons.

| Status | Meaning |
|--------|---------|
| `accept`, `reject`, `undecided` | The outcome of the procedure. No reasons. |
| `unresolved` | The record supports no decision as it stands. |
| `invalid` | A value in the record cannot be, or two statements in it contradict each other. |

The checks are made in this order, and the first that fails gives the
status.

1. `invalid`: a number that cannot be.
   - `result.value` or `decision.limit` is not a finite number;
   - `result.u` or the `u` of a component is not a finite number that is at
     least zero;
   - `result.dof` is not a finite number above zero;
   - `decision.risk` is not strictly between 0 and 0.5;
   - `capture.records_lost` is not an integer that is at least zero;
   - `result.significant_digits` is neither `"all"` nor a whole number from
     1 to 17;
   - `result.value_step` is not a finite number that is at least zero, or is
     not zero in a record of working values;
   - a text of the table above differs from the one required;
   - a name that is known is not text: `quantity`, `convention`, `target`,
     `conditions`, `start_event`, `end_event` or `unit` of the measurand,
     `data.run_weighting`, or `capture.clock`. A number or a list in such a
     place names nothing;
   - in a rounded record, and if nothing above was found: `result.u`,
     `result.dof` or the `u` of a component has more significant digits
     than the record declares (`is not rounded to 2 digits`), or
     `result.value` is not a whole multiple of a `result.value_step` above
     zero (`is not a multiple of its step`). Rounding to two digits writes
     1.2 and cannot write 1.234; trailing zeros do not count, so that 1.20
     and 0.30 pass. The interval that the checker would form around such a
     number is that of no number that was rounded.

   Booleans and numbers written as text are not numbers. A number may be an
   integer, a float or a `Decimal`. One beyond the range of a float, too
   large for a float or too small to differ from zero in one, is not a
   finite number here (see *Opening a record*).
2. `unresolved`: a required input is unknown (`wants ...`). In a rounded
   record `result.value_step` is one of them.
3. `unresolved`: the format or the procedure is another one,
   `covariances` has entries (`correlations not supported`), or the risk is
   within 1e-300 of zero or of one half (`risk not supported`). A record with correlated components
   is not examined further, so that a coherent one is not called invalid by
   a formula that does not apply to it.
4. `invalid`: `result.u` and the components, combined as uncorrelated,
   cannot both be true within the rounding that the record declares.
5. `unresolved`: a required component is listed in `not_evaluated`, or
   records were lost.
6. The decision, or `unresolved` if the record is rounded and its rounding
   allows more than one outcome
   (`rounding decides: wants working values`), or if it holds working values
   and the outcome depends on the allowance for the coverage factor
   (`at the limit: the coverage factor decides`), or if the factor was not
   computed and its lower bound does not settle the record
   (`coverage factor not computed`).

## Records of earlier formats

The checker decides nothing for a record that names an earlier format
(`format not supported`), and the name alone must not be edited to make it
pass.

- A record of working values (formats 1 to 4) is regenerated from its
  source: `python data/make_record.py` for the record of the book, the
  program that wrote it for any other. The fields are those of version 5,
  and a working record states no increment.
- A rounded record of format 4 says how many significant digits its
  numbers have and not to what increment its estimate was rounded. Recover
  that increment from the report or the program that rounded it, and state
  it in `result.value_step`. An increment of 0 says that the estimate was
  not rounded and that the working estimate was preserved. If the increment
  cannot be recovered, the record stays `unresolved`
  (`wants result.value_step`), which is what is known about it.

## What the checker does not examine

- Whether the model fits the system. A record that passes is consistent, and
  no more than that.
- `result.value` and `result.dof`, which are taken as stated. The data are
  not in the record, and the degrees of freedom are not recomputed.
- `result.interval_95`, which is not read.
- `decision.outcome` takes no part in the decision. `decide` recomputes the
  outcome and does not read the stated one, and a copied or edited record can
  carry a stale outcome. `stale(record)` returns the stored outcome if it is
  known and is not the status that `decide` finds, so that the two can be
  reported side by side, and `None` otherwise.
- Correlated components. A record with entries in `covariances` is
  `unresolved`.
- Whether a number declared as rounded to *n* digits is written with no more
  than *n*. The declaration is taken at its word.
- The fields of `data` other than `run_weighting`, the fields of `sources`
  and `replay`, the texts of `assumed` and `not_claimed`, the `kind` of any
  group of values, the stored step and the sampling rule of `capture`, and
  the type, the degrees of freedom and the source of a component.

`missing(record)` lists the inputs of the decision that are unknown: the
required fields, the three lists, and in each component its name, its `u`
and, for a correction, its `value`. A count of zero means that these are
known. It does not mean that every field of the record is filled in.

`spread(record)` returns the lower and upper bound of what the components
combine to, and `combined(record)` the bounds that the whole record supports,
with `result.u` taken into account, or `None` if the two do not meet.
`guard(record)` returns the bounds of the estimate plus the guard band and of
the estimate less it. For working values with a factor that is known, each
pair of bounds is one number. With a computed factor the two bounds lie a
millionth of the band on either side of the computed sum.
All three return decimals (see *What the checker shows*).

## Comparing two records

`comparable(a, b)` returns the reasons why two records are not results for
one quantity, and notes on what a comparison would have to convert.

- The quantity, the start and end events, the convention, the target and the
  conditions must be known in both and equal. The conditions, the load among
  them, are part of what defines the quantity: a p95 at 100 requests a second
  and one at 1000 are results for two measurands.
- If the load is the variable under study, say so:
  `comparable(a, b, across=("conditions",))`. A difference in a field named
  there is returned as a note (`across conditions`) and not as a reason. An
  unknown value is still a reason, and so is every other difference.
  A name that is known and is not text, a number or a list in the place of
  a clock or of an event, is a reason as well (`clock is not text`): two
  records with the same invalid name do not thereby describe one quantity.
  Only `conditions` can be named: for any other name `comparable` raises
  `ValueError`. Other events, another convention, another target or another
  weighting make another measurand, and results for two measurands are the
  subject of a different analysis, which this function does not perform.
- `data.run_weighting` must be known in both and equal. A mean of run
  percentiles weighted equally and one weighted by requests are different
  quantities, as the last block of Transcript 22.2 shows.
- The units must be of one kind of quantity. If they differ, the conversion
  is returned as a note.
- Both clocks must be known. If they differ, one of the two records must
  hold an `established` relation of `rate` that names exactly these two
  clocks. The offsets of the clocks do not enter an elapsed interval taken on
  one clock, and are not asked for. A relation stated inside one record for
  other clocks satisfies nothing. Nor does one that names a clock by anything
  but text: the number 7 in a relation does not name the clock `"7"`.

`in_unit(record, unit)` converts the estimate and the standard uncertainty of
the result, exactly, and nothing else. An unknown one stays unknown. A record
of which only these two numbers were converted is not a record in the new
unit.
