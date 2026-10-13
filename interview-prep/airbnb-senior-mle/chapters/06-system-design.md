# 06. General Backend System Design: Booking, Availability, Search, Messaging, Payments and Feature Store

> **Research note.** This session's web-search budget ran out, and most interview sites (Glassdoor, Blind, LeetCode, Reddit, Medium, IGotAnOffer, TrueInterview, the Airbnb Tech Blog) could not be reached. The chapter therefore rests on the sources that did load: aggregated candidate reports for 2025 to 2026 hosted on GitHub [1][2][3][4], a community prep guide [5], a crowd-sourced question list [6], the Tech Interview Handbook [7] and Airbnb's open-source Chronon repository [8]. A question is tagged [Reported] only when one of these sources reports it for Airbnb. Every other question is [Likely]: an inference from Airbnb's products and the senior bar described in those sources. Before the onsite, re-run searches on Glassdoor, Blind and LeetCode Discuss for "Airbnb system design 2026".

## Format notes

- **Rounds and length.** Candidates report one or two system design rounds of 45 to 60 minutes each. A second design round is sometimes added late in the process when the signal is borderline, and one candidate said that extra round decided the outcome [1][2]. The Tech Interview Handbook lists one "System Design/architecture" round in the general engineering onsite [7]. For the 1-to-3-year level (G8), one report describes a 60-minute design screen plus a 60-minute onsite design round on Miro or Excalidraw [5].
- **Interviewer.** A senior or staff engineer runs the round. MLE loops usually have a separate ML system design round (see chapter 03), but generic backend prompts also appear in engineering loops at senior level [1][2][3].
- **Senior bar.** The aggregated reports say design depth "reads staff-level even for senior candidates." Candidates are expected to "drive requirements yourself, fix a minimal scope in the first few minutes, then go deep where the interviewer leans in" [1].
- **Common failure.** Trying to cover every feature. One candidate lost an in-memory SQL engine round by attempting the whole feature set without first agreeing what "done" meant. The reports say design rounds "reward scoping and negotiated trade-offs" [1].
- **Recurring themes.** Double-booking prevention, availability search, and hot partitions caused by a big event [1][5]. Others are notification fan-out, chat, job scheduling, key-value stores pushed onto CAP trade-offs, and streaming aggregation with Flink and Kafka [1][3][4][6].
- **What this means for an MLE.** Expect a classic backend prompt framed around Airbnb's own products (booking, calendar, search, messaging). Expect the interviewer to probe whatever your background makes natural: feature stores, logging pipelines and experiment assignment.

## Questions

### A. Booking and reservation system

1. Design Airbnb's booking system: guests search for available listings and book them, and the system must never double-book a listing. [Reported] [1][3][5]
   - What are the functional requirements you would commit to in the first five minutes, and what would you explicitly leave out?
   - Is a booking request a strongly consistent write? Which parts of the flow can be eventually consistent?
   - How do you guarantee two guests cannot book overlapping dates on the same listing at the same moment?
   - Would you use pessimistic row locks, optimistic concurrency with version numbers, or a uniqueness constraint on (listing, night)? Why?
2. Design a booking platform that prevents double bookings and also supports search. Follow-up: a big event (for example a festival or a World Cup final) creates a hot partition. How do you handle it? [Reported] [5]
   - Which keys get hot: listing IDs, a city's search shard or the availability index?
   - How do you shed or queue load without letting someone book a listing that is already taken?
   - Would you pre-shard or add caches for a known upcoming event? How would you detect an unplanned spike?
3. Design a hotel booking system (rooms with inventory counts rather than unique homes). [Reported] [3]
   - How does inventory-count booking differ from Airbnb's one-unit-per-listing model?
   - How do you handle overbooking policies, if any are allowed?
4. Walk through the state machine of a reservation: request, pending host approval, accepted, paid, checked in, completed, cancelled or altered. [Likely]
   - Where do you persist state transitions, and how do you make them auditable?
   - How do you expire a booking request the host never answers within 24 hours?
5. How would you implement a temporary hold on dates while the guest is on the checkout page? [Likely]
   - How long should the hold last, and what releases it?
   - What happens if the payment succeeds after the hold has expired?
6. Design Instant Book versus Request to Book. How do the two flows differ in consistency and latency requirements? [Likely]
7. How do you handle reservation alterations, such as changing dates or guest count on an existing booking? [Likely]
   - How do you avoid releasing the old dates before the new dates are secured?
8. Design the cancellation flow, including refunds that depend on the cancellation policy and the time remaining before check-in. [Likely]
9. Design a waitlist system, for example for a sold-out experience or a high-demand listing. [Reported] [3]
   - How do you notify the next person fairly and with a deadline to claim?
   - How do you avoid a thundering herd when a slot opens?
10. How would you shard the reservations database as Airbnb grows across regions? [Likely]
    - Shard by listing, by guest or by host? Which queries become cross-shard?
11. How do you make the "Reserve" API idempotent when the mobile client retries on a flaky network? [Likely] [9]
12. Booking needs data from several services: listing, pricing, availability, payments and trust. How do you coordinate the transaction across them? [Likely] [9]
    - Would you use a saga, two-phase commit or an orchestrator? What are the compensating actions?

### B. Availability calendar

13. Design the availability calendar service for millions of listings, in which hosts block dates and set minimum and maximum nights and per-night prices. [Likely] [1][3]
    - What is the data model: one row per listing-night, interval ranges or bitmaps?
    - What are the read-to-write ratios for search, listing-page views and host edits?
14. Design Google Calendar (events, recurrence, invites, free/busy). [Reported] [3]
    - How do you store and expand recurring events efficiently?
    - How would you answer "is everyone free from 2 to 3pm" across many calendars?
15. Hosts sync calendars with other platforms through iCal. How do you import and export them without causing double bookings? [Likely]
    - What polling frequency would you use, and how do you handle conflicts between an external block and a pending Airbnb request?
16. How do you answer "which listings in Paris are available from June 3 to June 7 for 4 guests" at search time? [Likely] [1]
    - Do you pre-compute availability into the search index, or filter after retrieval?
    - How stale can the search index's availability be? What happens if a user clicks a listing that has just become unavailable?
17. Design the split-stay feature, which offers two listings that together cover a guest's dates when no single listing does. [Likely] [1]
    - How do you search pairs of listings without an O(n squared) blow-up?
    - What changes if you allow three-way splits?
18. How do you handle time zones and check-in/check-out boundaries for listings around the world? [Likely]
19. A host edits the calendar while a guest is booking the same dates. Which write wins, and how do you guarantee correctness? [Likely]
20. How would you propagate availability changes to search, pricing and ML ranking features within seconds? [Likely]
    - Would you use change data capture (CDC) to a log, dual writes or an outbox pattern?

### C. Search and discovery backend

21. Design the Airbnb home page: search, availability and ranking. [Reported] [3]
    - What are the latency budgets for retrieval, filtering and ranking?
    - How does the page behave when the ranking service is slow or down?
22. Design ranking for home search results when the user has typed no text query (only location, dates and guests). [Reported] [3]
23. Design a query system over user activity that filters by time window and geography. [Reported] [2][3]
    - What index structures would you use for the geo part (geohash, S2 cells, quadtrees) and for the time part?
    - How do you support queries such as "views in this bounding box in the last 7 days" at scale?
24. Design a map-based search: as the user pans and zooms, return the listings within the viewport. [Likely]
    - How do you avoid hammering the backend on every pan event?
    - How do you cluster pins when thousands of listings fall in the viewport?
25. How would you build the search index (inverted index plus geo index) and keep it fresh as listings, prices and availability change? [Likely]
    - Full reindex versus incremental updates: what triggers each?
26. Design autocomplete for destinations and points of interest. [Likely]
27. Design semantic search over user and listing embeddings. [Reported] [3]
    - How do you choose and operate an approximate nearest-neighbor index (HNSW, IVF-PQ) next to structured filters?
    - How do you re-embed and re-index without downtime?
28. Design the host listings page, which aggregates data from many services for hosts with hundreds of listings, and optimize it. [Reported] [3]
    - What would you pre-compute, cache or paginate?
    - How do you avoid N+1 calls across service boundaries?
29. Design pagination for search results that stays stable when ranking or availability changes between page loads. [Likely]
30. How would you cache search results, given that the same location and dates are queried by many users but prices and availability change? [Likely]
31. Design the wishlist service, including shared wishlists among group members. [Likely]

### D. Messaging and notifications

32. Design a group chat system (WhatsApp-like). [Reported] [5]
    - How do you guarantee per-conversation ordering and exactly-once display?
    - How do you deliver to offline users and sync across multiple devices?
33. Design group chat with a feature that returns a user's top-N most active chats. [Reported] [1]
    - How would you define and maintain "active" in real time?
34. Design a Slack-like chat system. [Reported] [3][4]
35. Design guest-host messaging on Airbnb: messages tied to a listing or reservation, with read receipts and translation. [Likely]
    - How do you store the conversation so it is linked to the reservation and visible to support agents?
    - Where would you hook in fraud or off-platform-payment detection without adding latency to sending?
36. Design a unified notification platform that handles both event-triggered and batch sends. [Reported] [1][3][4]
    - How do you prioritize transactional notifications (booking confirmed) over marketing?
    - How do you de-duplicate, rate-limit per user and respect quiet hours across time zones?
    - What happens when a downstream provider (APNs, FCM, an SMS gateway) is slow or failing?
37. How would you design per-user notification preferences and channel fallbacks (push, then email, then SMS)? [Likely]
38. Design a customer-support ticketing system that routes tickets to the right agent queue. [Reported] [3]
    - How do you combine rules-based routing with an ML classifier, and how do you fall back when the model is down?
39. Design a system that generates customer-support instructions from company policies, for example with an LLM. [Reported] [3]
    - How do you keep generated instructions in sync when a policy changes?
    - How do you evaluate correctness before showing them to agents?
40. Design a system that sends reminder messages (check-in instructions, review requests) at specific times relative to each reservation. [Likely]

### E. Payments, payouts and financial correctness

41. How would you design the payment flow so a guest is never charged twice, even with retries and partial failures? [Likely] [9]
    - What is an idempotency key? Who generates it, and how long do you keep it?
    - How do you separate the local DB transaction from the remote payment-provider call?
42. Design host payouts: scheduling, splitting among co-hosts, multiple currencies and payout methods. [Likely]
43. Design refunds across multiple payment methods and dates, honoring a priority order. [Likely] [10]
44. Design a ledger (double-entry) for Airbnb's payments platform. [Likely]
    - How do you reconcile your ledger against the payment processor's settlement files?
45. Design a banking-like service with deposit, withdraw, transfer and balance. Follow-up: how would you scale it into a distributed service? [Likely] [1]
46. Design a currency-conversion and price-display service that shows consistent prices from search through checkout. [Likely]
    - How do you lock an exchange rate for a quote?
47. Design a retryer or retry library for calls to unreliable downstream services. [Likely] [1]
    - How do you decide which errors are retryable?
    - How do you combine exponential backoff, jitter, deadlines and circuit breakers?
48. How would you design fraud checks into the payment path without hurting checkout latency? [Likely]

### F. Data, logging and analytics infrastructure

49. Design a highly interactive data analysis system. [Reported] [6]
    - What latency problem appears if you start with batch Spark processing?
    - How would you use Flink with Kafka for streaming aggregation?
    - How does recovery work when a Flink job fails (checkpoints, offsets, exactly-once)?
50. Design an ad-click (or listing-click) event aggregation system. [Reported] [3]
    - How do you handle late and duplicate events?
    - How do you reconcile real-time counts with batch-corrected counts?
51. Model cumulative booking metrics, including the first and last booking times per user or listing. [Reported] [3]
    - Would you compute this incrementally or recompute it? How do you backfill when the logic changes?
52. Design the logging pipeline that captures search impressions, clicks and bookings for ranking training data. [Likely]
    - How do you join impressions to bookings that happen days later?
    - How do you guarantee the logged features match what the model saw at serving time?
53. Design a metrics and dashboard system for marketplace health (bookings, nights, supply) with daily and real-time views. [Likely]
54. How would you design data-quality checks and alerting for core tables that feed models and experiments? [Likely]

### G. Feature store and ML platform infrastructure

55. Design a feature store that serves offline training and low-latency online ranking from one feature definition. [Likely] [8]
    - How do you guarantee point-in-time correctness in training backfills?
    - How do you measure and alert on online/offline consistency?
56. How would you support windowed aggregations (for example a listing's bookings in the last 7 days) in both streaming and batch? [Likely] [8]
    - How do you handle very skewed keys, such as a popular listing or a bot user, in the aggregation?
57. Design the online serving layer of a feature store for a ranking request that fetches features for 1,000 candidate listings in under 50 ms. [Likely] [8]
    - What key-value store would you choose? How do you batch, cache and tolerate tail latency?
58. How do you version features, and what happens to running models when a feature definition changes? [Likely]
59. Design a model-serving platform that supports canary releases, shadow traffic and rollback. [Likely]
60. Design an embedding-serving service that keeps listing embeddings fresh as listings and behavior change. [Likely]
61. Design a job scheduler for about ten million users. [Reported] [1][3][4]
    - How do you guarantee each job runs at least once, and when do you need exactly-once?
    - How do you scale the dispatcher and avoid one hot time bucket (for example midnight UTC)?
62. Design a distributed scheduler for ML training and backfill jobs with dependencies (a DAG). [Likely]

### H. Experimentation platform (backend view)

63. Design an A/B testing platform. [Reported] [3]
    - How do you assign users to variants deterministically and consistently across web, iOS and Android?
    - How do you handle logged-out users who later log in?
    - How do you log exposures so analysis is not diluted?
64. Design the assignment service's API and latency budget, given that every page load may evaluate dozens of experiments. [Likely]
    - Do you evaluate on the server or client SDK? How do you push config changes?
65. How would you support mutually exclusive experiment layers and holdouts? [Likely]
66. Design the metrics pipeline that computes experiment scorecards daily for hundreds of experiments and thousands of metrics. [Likely]
67. How would you design a kill switch or automated guardrail that stops an experiment when a key metric degrades? [Likely]
68. Design a network-aware experiment for a "Wi-Fi badge" on listings where treatment effects can spill over between listings. [Reported] [3]
    - How would the platform support cluster or market-level randomization?

### I. Generic distributed-systems building blocks (asked at Airbnb)

69. Design a distributed key-value store. Follow-up: walk through CAP trade-offs. [Reported] [1]
    - How do you replicate, and what consistency level do reads get?
    - How do you handle node failure and rebalancing?
70. Design an in-memory SQL engine. Agree on scope first. [Reported] [1]
    - Which subset of SQL (select, where, join, group by) do you commit to, and why?
71. Design an online chess game. [Reported] [3][4]
    - How do you sync game state in real time and handle disconnects and clocks?
72. Design a rate limiter for Airbnb's public APIs and internal services. [Likely]
73. Design a URL shortener for share links to listings and wishlists. [Likely]
74. Design a distributed cache in front of the listing service, and explain invalidation when a host edits a listing. [Likely]
75. Design account merge and user de-duplication when people create multiple accounts. [Likely] [1]
    - How do you merge reservations, reviews and payment methods safely?

### J. Other Airbnb product surfaces

76. Design the reviews system: double-blind reviews revealed only when both parties submit or the window closes. [Likely]
    - How do you implement the reveal-at-deadline logic reliably?
77. Design the listing creation and photo upload pipeline, including image processing and moderation. [Likely]
78. Design Experiences booking with time slots, group capacity and a waitlist. [Likely] [3]
79. Design a multi-traveler itinerary service that combines stays and experiences, for example filling a layover with experiences. [Likely] [10]
80. Design a relisting detection system that catches a banned host relisting the same property under a new account. [Reported] [2][3]
    - How do you use geohashing and photo or text similarity at scale?
81. Design a listing lifetime-value estimation service and its batch and online parts. [Reported] [2][3]
82. Design the trust and safety screening pipeline that scores every booking before confirmation. [Likely]
    - How do you keep it off the critical path when possible, and fail open or closed when it is down?
83. Design the host dashboard's earnings and occupancy insights. [Likely]
84. Design Airbnb's internationalization service for translated strings and listing translations. [Likely]

### K. Cross-cutting senior probes (apply to any prompt)

85. What are your back-of-the-envelope numbers for this design (QPS, storage, peak-to-average ratio)? [Likely] [1]
86. You have 45 minutes. Which two components would you deep-dive, and why those? [Likely] [1]
87. Where is the single point of failure in what you just drew? [Likely]
88. How do you deploy a schema change to the reservations table with zero downtime? [Likely]
89. How would you run this system across multiple regions? What happens during a region failover in the middle of a booking? [Likely]
90. What metrics, alerts and SLOs would you define for this service? [Likely]
91. Which part of this design would you A/B test, and what could go wrong with the test because of the architecture (caching, sticky assignment, logging gaps)? [Likely]
92. How would you migrate from the existing monolith to this service-oriented design incrementally? [Likely]
93. If traffic grew 10x tomorrow, what breaks first? [Likely]
94. What would you simplify if you had to ship a first version in two weeks? [Likely] [1]
95. How do you secure personal data (addresses, IDs, payment details) in this design, and who can access it? [Likely]

## References

Sources 1 to 8 were fetched in this session. Sources 9 and 10 were recalled or listed by another source and were not fetched.

1. "Airbnb Software Engineer Interview Process" (aggregated from 41 candidate reports, 2025 Q4 to 2026 Q2), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
2. "Airbnb Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
3. "Airbnb: every question" (system design titles with report dates, 2025 to 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
4. "Airbnb" company overview (rounds and prompts by round type), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/README.md
5. "Airbnb" company prep page (Jan 2026 senior SWE booking/hot-partition report, G8 loop description), jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
6. "Airbnb interview questions" (Problem 10: data analysis system with high interactivity), warrenzhu25/system-design, GitHub. https://github.com/warrenzhu25/system-design/blob/master/airbnb_interview_questions.md
7. "Interview formats of top companies", Tech Interview Handbook (yangshun), GitHub. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
8. "Chronon: a data platform for AI/ML" (Airbnb open-source feature platform), airbnb/chronon, GitHub. https://github.com/airbnb/chronon
9. "Avoiding Double Payments in a Distributed Payments System", The Airbnb Tech Blog (Medium), 2019. Recalled from memory; this session could not reach medium.com, so the URL was not checked. https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-2981f6b070bb
10. Airbnb 1point3acres-sourced coding prompts, including "Refund Allocation by Payment Method and Date Priority" and "Fill Layover Hours by Booking Experiences" (folder listing only), FlowSight/coding, GitHub. https://github.com/FlowSight/coding/tree/main/company%20wise/airbnb/1p3a
