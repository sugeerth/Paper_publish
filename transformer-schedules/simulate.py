"""Dependency-driven simulator for pipeline-parallel transformer schedules.

The point is that the picture should be *derived* from the scheduling rule, not
drawn by hand. So: define each stage's task order (that is what distinguishes
GPipe from 1F1B), then run an exact list-scheduler over the dependency graph.

Dependencies:
  F(k,i) needs F(k,i-1)   activation flows forward
  B(k,i) needs B(k,i+1)   gradient flows backward
  B(k,p-1) needs F(k,p-1) turnaround at the last stage
  plus: a stage runs one task at a time, in its declared order.
"""
from dataclasses import dataclass


@dataclass
class Task:
    kind: str   # "F" or "B"
    mb: int     # microbatch index
    stage: int
    chunk: int = 0   # virtual stage (interleaved 1F1B)
    start: float = 0.0
    dur: float = 1.0


def gpipe_order(p, m):
    """All forwards, then all backwards, on every stage."""
    return [[Task("F", k, i) for k in range(m)] + [Task("B", k, i) for k in range(m)]
            for i in range(p)]


def one_f_one_b_order(p, m):
    """Warm up with (p-1-i) forwards, then alternate F/B, then drain backwards."""
    orders = []
    for i in range(p):
        warm = min(p - 1 - i, m)
        seq = [Task("F", k, i) for k in range(warm)]
        f_next, b_next = warm, 0
        while f_next < m:                      # steady state
            seq.append(Task("F", f_next, i)); f_next += 1
            seq.append(Task("B", b_next, i)); b_next += 1
        while b_next < m:                      # cooldown
            seq.append(Task("B", b_next, i)); b_next += 1
        orders.append(seq)
    return orders


def schedule(orders, p, dur=1.0):
    """Exact list-schedule: earliest start respecting deps + stage serialisation."""
    done = {}                      # (kind, mb, stage) -> finish time
    tasks = []
    idx = [0] * p                  # next task per stage
    stage_free = [0.0] * p
    remaining = sum(len(o) for o in orders)

    while remaining:
        progressed = False
        for i in range(p):
            if idx[i] >= len(orders[i]):
                continue
            t = orders[i][idx[i]]
            ready = stage_free[i]
            if t.kind == "F":
                if i > 0:
                    dep = done.get(("F", t.mb, i - 1))
                    if dep is None:
                        continue
                    ready = max(ready, dep)
            else:
                if i == p - 1:
                    dep = done.get(("F", t.mb, i))
                else:
                    dep = done.get(("B", t.mb, i + 1))
                if dep is None:
                    continue
                ready = max(ready, dep)
            t.start, t.dur = ready, dur
            done[(t.kind, t.mb, i)] = ready + dur
            stage_free[i] = ready + dur
            idx[i] += 1
            remaining -= 1
            tasks.append(t)
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock — bad task order")
    return tasks


def stats(tasks, p):
    """Two different quantities are both called 'the bubble'. Keep them apart.

    idle_frac  fraction of total device-time (makespan x p) that is idle
               = (p-1)/(m+p-1) for both GPipe and 1F1B
    bubble_ratio  bubble time as a fraction of the *ideal* (zero-bubble) time,
               which is the ratio the Megatron-LM paper reports = (p-1)/m
    """
    makespan = max(t.start + t.dur for t in tasks)
    busy = sum(t.dur for t in tasks)
    ideal = busy / p
    return {
        "makespan": makespan,
        "idle_frac": 1 - busy / (makespan * p),
        "bubble_ratio": (makespan - ideal) / ideal,
    }


def peak_residency(tasks, p):
    """Microbatch activations held per device: +1 when a forward completes,
    -1 when the matching backward starts. This is what separates the two
    schedules -- their timelines are identical."""
    peaks = []
    for i in range(p):
        events = []
        for t in tasks:
            if t.stage != i:
                continue
            events.append((t.start + t.dur, +1) if t.kind == "F" else (t.start, -1))
        events.sort(key=lambda e: (e[0], -e[1]))  # arrivals before releases at equal t
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        peaks.append(peak)
    return peaks


if __name__ == "__main__":
    hdr = f"{'sched':<7}{'p':>3}{'m':>4}{'span':>7}{'idle':>8}{'bubble':>8}{'peak':>7}  per-device peak"
    print(hdr)
    print("-" * len(hdr))
    for p, m in ((4, 4), (4, 8), (4, 16), (8, 16)):
        for name, fn in (("GPipe", gpipe_order), ("1F1B", one_f_one_b_order)):
            ts = schedule(fn(p, m), p)
            s = stats(ts, p)
            pk = peak_residency(ts, p)
            print(f"{name:<7}{p:>3}{m:>4}{s['makespan']:>7.0f}"
                  f"{s['idle_frac']:>8.1%}{s['bubble_ratio']:>8.1%}{max(pk):>7}  {pk}")
        print()
    print("idle   = share of device-time idle; identical for GPipe and 1F1B")
    print("bubble = bubble/ideal, the Megatron-LM ratio (p-1)/m")
    print("peak   = activations held; GPipe grows with m, 1F1B is bounded by p")
