# VISxAI 2026 submission package

**Deadline: 10 August 2026, anywhere on Earth.** Verified from the workshop's own
site source (`github.com/VISxAI/visxai.github.io`, `src/lib/data/years.ts`:
`{ date: 'August 10, 2026, anywhere', event: 'Submission Deadline', previousDate:
'August 3, 2026' }`, commit dated 2026-07-29). Already extended once from Aug 3;
do not assume a second extension.

9th Workshop on Visualization for AI Explainability, at IEEE VIS 2026,
Boston, November 2026. Camera-ready 23 October 2026.

## How to submit (4 steps, from the workshop's submit page)

1. Log in or create an account at <https://new.precisionconference.com>.
2. Submission tab → Society **VGTC** → Conference **VIS 2026** → Track
   **VIS 2026 VISxAI Workshop Blog Posts**.
3. Fill in title, all authors, abstract. Add the live link in the abstract field
   and/or upload a document. **A link or a document is required** — either alone
   is a complete submission.
4. Press **Record**. (A submission that is not recorded is not submitted.)

## Title

> Where Did the GPU-Seconds Go? Depicting the Execution Schedule Inside a
> Transformer

## Abstract (ready to paste)

> Transformer visualization is dominated by the attention heatmap, an encoding we
> arguably adopted because a softmax output is already a matrix and a matrix is
> already a heatmap. It answers a semantic question: what did this head attend
> to. But most questions asked of a transformer in practice are cost questions —
> where did the GPU-seconds go, why does tail latency spike on a long prompt, why
> does this fit at four-way pipeline depth but not eight. Those are scheduling
> questions, and a schedule is a different object from a dataflow graph: a
> contended resource extended over time.
>
> This explainable argues for depicting that object, and demonstrates it. Every
> schedule shown is computed in the browser by list-scheduling the dependency
> graph, so readers change the pipeline depth and microbatch count and watch the
> claims hold rather than trusting a stored figure. The central demonstration is
> a negative result about encoding: GPipe and 1F1B produce identical makespan and
> identical idle time, so on a pure timeline they are the same picture
> rearranged — yet 1F1B is the schedule the field actually runs. The reason is
> invisible to an occupancy view and obvious in a residency view: GPipe's peak
> activation memory grows with the number of microbatches while 1F1B's is bounded
> by pipeline depth. A schedule visualization showing only occupancy cannot
> explain why the dominant schedule exists.
>
> We extend the same encoding to inference, where static and continuous batching
> are contrasted on identical arrivals, and where logical and wall-clock time
> diverge — that divergence being precisely what latency is. We close with five
> transferable rules: draw resource × time to scale so area reads as cost; make
> idle an explicit mark rather than the colour of the page; treat occupancy and
> residency as independent lanes; attach every gap to the dependency that caused
> it; and always say which clock the axis is showing.

## Link

The submission needs a public URL. Two options, in order of preference:

1. **GitHub Pages** — `transformer-schedules/index.html` is a single
   self-contained file with no external dependencies, so enabling Pages on this
   repository serves it as-is. Most stable and most citable.
2. **The published artifact** — already deployed, but private by default; it must
   be shared from the page's share menu before the reviewers can open it.

Whichever is used, put the URL in the abstract field as the submit page
instructs, and additionally upload a PDF print of the page as the document so
the submission survives link rot.

## What I could not do for you

- **Submitting.** It needs your PCS account, and authorship is yours to decide.
- **Author list and affiliation.** The page currently carries no byline.
- **Making the link public.** Enabling Pages or sharing the artifact is a
  publishing action, and that is your call.

## Honest assessment of fit

Good, not perfect, and worth going in with eyes open.

**For:** the workshop's primary format *is* the interactive web-native
explainable rather than a PDF, so the tool is the deliverable rather than
supporting material competing for a page budget — which is exactly the shape of
this work. There is no page limit. Review is constructive and acceptance odds at
a workshop of this kind are good. There is direct precedent for
transformer-mechanics explainables in their featured list.

**Against:** VISxAI centres on explaining model *behaviour and decisions* —
interpretability, attribution, what a model learned. This work explains model
*execution*. "How does this actually run" is a defensible reading of AI
explainability and the systems angle is underrepresented there, but it is a
reviewer-dependent stretch rather than a bullseye, and a reviewer who reads the
scope narrowly could reasonably call it out of scope.

**The better topical fit was ProTools 26 at SC26** (performance visualization
tools; its CfP explicitly solicits scalable displays of performance data and
interfaces for exploring it, and calls out AI workloads). It shares the same
10 August deadline but wants a 10-page archival IEEE paper, which is not
achievable at quality in the time remaining. If this work continues, ProTools 27
is the natural archival home and the simulator already supports it.

**If both deadlines are missed:** ML for Systems at NeurIPS 2026 closes 29 August
— four pages, not anonymised, and explicitly non-archival, so it can be
presented there and published in full elsewhere afterwards. That is the
lowest-risk fallback and it is three weeks out rather than one day.
