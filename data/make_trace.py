"""Generate data/app_trace.log, the trace fixture of Chapter 16.

    python data/make_trace.py

The generator is seeded, so the file it writes is always the same. The
trace covers 60 s and contains, by construction:

  * "svc" messages for request 17, 2.3456 s apart (a Time Delta);
  * "db" messages at about 20 per second for 30 s, then about 5 per
    second (a change of Statement Current);
  * a "net" error every 7.5 s with 50 ms of jitter (a Periodic Error);
  * a "pool" heartbeat once a second;
  * 40 consecutive messages dropped by the tracer near 21 s, visible
    only as a gap in the sequence numbers.

Timestamps are truncated to the millisecond, as many tracers do.

The tracer numbers the 799 messages it is given from 1, before any is
dropped. Its counter of sequence numbers therefore reads 1 when the capture
starts and 800 when it stops, which Listing 16.1 takes as metadata.
"""
import random
from pathlib import Path

T0_MS = (12 * 3600 + 26 * 60) * 1000  # the trace starts at 12:26:00.000
DROP = range(430, 470)                # positions of the dropped messages


def stamp(t):
    """Seconds since the start -> HH:MM:SS.mmm, truncated to the ms."""
    ms = T0_MS + int(t * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def events(rng):
    out = []
    t = 0.0
    while True:
        t += rng.expovariate(20 if t < 30 else 5)
        if t >= 60:
            break
        out.append((t, 3120, "db", f"query {rng.randrange(1000, 9999)} done"))
    for i in range(8):
        t = 3.2 + 7.5 * i + rng.gauss(0, 0.05)
        out.append((t, 2204, "net", "E1042 timeout"))
    for i in range(60):
        t = i + 0.5 + rng.gauss(0, 0.002)
        out.append((t, 1010, "pool", "heartbeat"))
    out.append((10.1234, 2204, "svc", "request 17 start"))
    out.append((10.1234 + 2.3456, 2204, "svc", "request 17 end"))
    out.sort()
    return out


def main():
    rng = random.Random(16)
    lines = []
    for seq, (t, tid, comp, msg) in enumerate(events(rng), 1):
        if seq - 1 in DROP:
            continue
        lines.append(f"{seq:06d} {tid} {stamp(t)} {comp:<4} {msg}")
    path = Path(__file__).with_name("app_trace.log")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} messages written to {path.name}")


if __name__ == "__main__":
    main()
