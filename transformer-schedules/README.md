# Depicting the schedule inside a transformer

A visual-encoding argument: transformer visualization is dominated by attention
heatmaps, which answer semantic questions, while nearly every question asked of a
model in production — throughput, latency, memory headroom, idle time — is a
question about *time on contended hardware*. That is a different object and needs
a different picture.

- `index.html` — the interactive explainer. Schedules are simulated in-page, so
  moving the controls recomputes the numbers rather than swapping stored images.
- `simulate.py` — the reference simulator. The JavaScript in the page is a port of
  this; both produce identical results (p=4, m=8: makespan 22, 27.3% idle, peak 8
  under GPipe and 4 under 1F1B).

## The argument in one result

```
sched    p   m   span    idle  bubble   peak  per-device peak
GPipe    4   8     22   27.3%   37.5%      8  [8, 8, 8, 8]
1F1B     4   8     22   27.3%   37.5%      4  [4, 3, 2, 1]
```

GPipe and 1F1B have **identical makespan and identical idle time**. On a pure
timeline they are the same picture rearranged — yet 1F1B is what everyone runs.
The reason is the last column: GPipe's peak activation memory grows with the
number of microbatches, 1F1B's is bounded by pipeline depth.

So a schedule visualization that shows only occupancy cannot explain why the
dominant schedule in the field exists. Occupancy and residency are independent
projections, and for transformers the memory ceiling usually binds first.

## Running it

```sh
python3 simulate.py          # verify the schedules and the memory divergence
python3 -m http.server       # then open index.html
```

## Method and honesty

Schedules are computed by list-scheduling the dependency graph: a forward pass
needs the forward on the previous stage, a backward needs the backward on the
next stage, and each device runs one task at a time in its declared order. The
task order per device is the only thing that distinguishes GPipe from 1F1B.

The model is deliberately idealised — uniform op cost, no communication, no
compute/transfer overlap, backward costing the same as forward. These are the
same idealisations the original pipeline-parallelism papers use to *define* the
schedules, which makes them fair for comparing schedules against each other.
They are not measurements. Real bubbles are generally worse, and nothing here
should be read as a benchmark.

Two quantities both get called "the bubble" and the simulator keeps them apart:
`idle` is the share of total device-time that is idle, `(p−1)/(m+p−1)`; `bubble`
is bubble time over ideal time, `(p−1)/m`, which is the ratio Megatron-LM
reports.
