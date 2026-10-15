# 06. General Backend System Design: Booking, Availability, Search, Messaging, Payments and Feature Store

> **Research note.** This chapter was re-checked on 2026-10-14. Only GitHub-hosted sources could be reached; Glassdoor, Blind, LeetCode, Medium, Hello Interview, IGotAnOffer, PracHub, TrueInterview and the Airbnb Tech Blog did not resolve, and the web-search budget was used up. The chapter rests on aggregated 2025 to 2026 candidate reports on GitHub [1][2][3][4], a community prep guide [5], a crowd-sourced question list [6], the Tech Interview Handbook [7], Airbnb's open-source Chronon repository [8], and English and Chinese 1point3acres interview reports mirrored on GitHub [11]-[45]. A question is tagged [Reported] only when a cited source reports it, or a very close variant, for Airbnb. All other questions are [Likely]: inferences from Airbnb's products and the senior bar those sources describe. Most mirrored 1point3acres posts carry no date. Where a date is given, it is 2025 or 2026 except [21] (2023).

## Format notes

- **Rounds and length.** Mid-level loops usually have one design round and senior loops two, each about an hour of discussion with no code. Senior loops can also move system design into the phone stage. A design round is sometimes added after the onsite when the signal is borderline [1]. Several candidates describe that extra round, for example a classic booking prompt focused on preventing double booking [25] [1]. The Tech Interview Handbook lists one "System Design/architecture" round in the general onsite [7]. One 2026 report for the 1-to-3-year level (G8) describes a system design screen plus a second system design round at the onsite [5].
- **Interviewer and track.** MLE candidates usually get one or two ML system design rounds instead (see chapter 03) [2][4]. Generic backend prompts still appear at senior level [1][3], and staff loops have included both a distributed job scheduler and Airbnb booking [29].
- **Senior bar.** Senior loops "expect depth one candidate called staff level" [1]. Several rejected senior candidates were told their design round fell short of the senior bar [22][32][21].
- **Common failures.** Running out of time while trying to cover everything (an in-memory SQL database prompt) [1]. "Chased perfection on the chat schema instead of aligning on trade-offs" [1]. Needing too many hints on API design [38]. Talking about scaling when the interviewer wanted to go deep on APIs and database design [16].
- **What gets probed.** Schemas and indexes that make the stated queries cheap (top-N chats, time and location filters); double booking under contention, idempotent refunds and delivery guarantees [1]. Interviewers often treat the supporting services (pricing, search, notifications) as given and focus on the data schema and one or two core flows [19].
- **What this means for an MLE.** Expect a classic marketplace prompt (booking, waitlist, chat, notifications, ticketing, job scheduler). Expect the interviewer to probe whatever your background makes natural: feature stores, logging pipelines and experiment assignment.

## Questions

### A. Booking and reservation system

1. Design Airbnb's booking system: guests search for available listings and book them, and the system must never double-book a listing. [Reported] [1][3][5][25][27][29]
   - What functional requirements would you commit to in the first few minutes, and what would you explicitly leave out?
   - Is a booking request a strongly consistent write? Which parts of the flow can be eventually consistent?
   - How do you guarantee two guests cannot book overlapping dates on the same listing at the same moment?
   - Would you use pessimistic row locks, optimistic concurrency with version numbers, or a uniqueness constraint on (listing, night)? Why?
   - How do you paginate and index availability search results? [1]
   - Follow-up: a big local event creates a hot partition. Which keys get hot, and how do you shed or queue load without letting anyone book a listing that is already taken? [5]
2. Design a hotel booking system, with rooms that have inventory counts rather than unique homes. [Reported] [3]
   - How does inventory-count booking differ from Airbnb's one-unit-per-listing model?
   - How do you handle overbooking policies, if any are allowed?
3. Walk through the state machine of a reservation: request, pending host approval, accepted, paid, checked in, completed, cancelled or altered. [Likely]
   - Where do you persist state transitions, and how do you make them auditable?
   - How do you expire a booking request the host never answers within 24 hours?
4. How would you implement a temporary hold on dates while the guest is on the checkout page? [Likely]
   - How long should the hold last, and what releases it?
   - What happens if the payment succeeds after the hold has expired?
5. Design Instant Book and Request to Book. How do the two flows differ in consistency and latency requirements? [Likely]
6. How do you handle reservation alterations, such as changing the dates or guest count on an existing booking? [Likely]
   - How do you avoid releasing the old dates before the new dates are secured?
7. Design the cancellation flow, including refunds that depend on the cancellation policy and the time left before check-in. [Likely]
8. Design a waitlist system for a hot listing, with APIs to join, check position and leave, and notifications when spots open under high contention. [Reported] [1][3][20]
   - How do you notify the next person fairly, with a deadline to claim?
   - How do you avoid a thundering herd when a slot opens?
9. Design a listing booking system in which users can join a waitlist when a listing is booked, and the first 20 waitlisted users are notified when it becomes available. Pricing, search and notification services already exist. [Reported] [19]
   - What is your data schema for bookings and waitlist entries?
   - How do you guarantee that exactly the first 20 users are notified once, even if the release event is delivered twice?
10. How would you shard the reservations database as Airbnb grows across regions? [Likely]
    - Would you shard by listing, by guest or by host? Which queries become cross-shard?
11. How do you make the "Reserve" API idempotent when the mobile client retries on a flaky network? [Likely] [9]
12. Booking needs data from several services: listing, pricing, availability, payments and trust. How do you coordinate the transaction across them? [Likely] [9]
    - Would you use a saga, two-phase commit or an orchestrator? What are the compensating actions?
13. Design a flight booking system that supports connecting flights while prices change. [Reported] [13]
    - How do you search for itineraries with connections (graph search, precomputed routes)?
    - Under a partition, and in normal operation, do you favor availability or consistency for prices and for seats (PA versus PC)? Agree on this before you design.

### B. Availability calendar

14. Design the availability calendar service for millions of listings, in which hosts block dates and set minimum and maximum nights and per-night prices. [Likely] [1][3]
    - What is the data model: one row per listing-night, interval ranges or bitmaps?
    - What are the read-to-write ratios for search, listing-page views and host edits?
15. Design Google Calendar (events, recurrence, invites, free/busy). [Reported] [3][39]
    - How do you store and expand recurring events efficiently?
    - What schema supports fast queries such as "is everyone free from 2 to 3pm" across many calendars?
16. Hosts sync calendars with other platforms through iCal. How do you import and export them without causing double bookings? [Likely]
    - What polling frequency would you use, and how do you resolve conflicts between an external block and a pending Airbnb request?
17. How do you answer "which listings in Paris are available from June 3 to June 7 for 4 guests" at search time? [Likely] [1]
    - Do you pre-compute availability into the search index, or filter after retrieval?
    - How stale can the search index's availability be? What happens if a user clicks a listing that has just become unavailable?
18. Design the split-stay feature, which offers two listings that together cover a guest's dates when no single listing does. [Likely] [1][25]
    - How do you search pairs of listings without an O(n squared) blow-up?
    - What changes if you allow three-way splits?
19. How do you handle time zones and check-in and check-out boundaries for listings around the world? [Likely]
20. A host edits the calendar while a guest is booking the same dates. Which write wins, and how do you guarantee correctness? [Likely]
21. How would you propagate availability changes to search, pricing and ML ranking features within seconds? [Likely]
    - Would you use change data capture (CDC) to a log, dual writes or an outbox pattern?
22. Walk through the data flow after a user modifies their profile information: which services, caches and indexes must update, and in what order? [Reported] [12]

### C. Search and discovery backend

23. Design the Airbnb home page: search, availability and ranking. [Reported] [1][3]
    - What are the latency budgets for retrieval, filtering and ranking?
    - How does the page behave when the ranking service is slow or down?
24. Design ranking for home search results when the user has typed no text query (only location, dates and guests). [Reported] [3]
25. Design a query system over user activity that filters by time window and geography. [Reported] [1][3]
    - What index structures would you use for the geo part (geohash, S2 cells, quadtrees) and for the time part?
    - How do you support queries such as "views in this bounding box in the last 7 days" at scale?
26. Design a map-based search: as the user pans and zooms, return the listings within the viewport. [Likely]
    - How do you avoid hammering the backend on every pan event?
    - How do you cluster pins when thousands of listings fall in the viewport?
27. Design a full-text search system: index building, query processing and integration with the underlying storage. [Reported] [13]
    - How do you optimize query latency and the efficiency of index updates?
    - When would you do a full reindex rather than incremental updates for listings, prices and availability?
28. Design typeahead (autocomplete) search that returns matching results in real time as the user types a prefix, for example for destinations. [Reported] [34]
    - What data structures and storage would you use, and how do you rank and cache prefix results?
29. Design semantic search over user and listing embeddings. [Reported] [3]
    - How do you choose and operate an approximate nearest-neighbor index (HNSW, IVF-PQ) next to structured filters?
    - How do you re-embed and re-index without downtime?
30. A host with more than 100 listings finds the host listings page very slow. Redesign the aggregation from the bottom up. [Reported] [1][3][25]
    - What would you pre-compute, cache or paginate?
    - How do you avoid N+1 calls across service boundaries?
31. Design cursor-based pagination for search results that stays stable when ranking or availability changes between page loads. [Likely] [1][41]
32. How would you cache search results, given that many users query the same location and dates but prices and availability keep changing? [Likely]
33. Design the wishlist service, including shared wishlists among group members. [Likely]

### D. Messaging and notifications

34. Design a group chat system (WhatsApp- or Slack-like): create groups, add members, send messages and view history. [Reported] [3][4][5][36][37][45]
    - The interviewer focuses on the database tables. What schema makes the required queries cheap?
    - How do you guarantee per-conversation ordering and exactly-once display?
    - How do you support presence and read receipts? [1]
35. Design group chat with an inbox that shows a user's top 20 chats, ordered by the most recent message or by activity over a configurable window. [Reported] [1][32][33][34][35]
    - How do you maintain this inbox in real time without scanning every group's messages?
    - How do you notify online and offline users of each new message (long polling, WebSockets, an asynchronous queue)? [35]
    - How do you route messages across many WebSocket servers, and how do you recover when one fails? [32]
36. Design a group-only instant messaging system with groups of 10,000 to 100,000 members. [Reported] [17]
    - Fan-out on write or fan-out on read? How do you avoid write amplification, and how do you shard?
    - Which delivery semantics do you choose, and how do you de-duplicate with message IDs?
    - How do you enforce ordering within a group (partition by group ID, a sequence generator)?
37. Design guest-host messaging on Airbnb: messages tied to a listing or reservation, with read receipts and translation. [Likely]
    - How do you store the conversation so it is linked to the reservation and visible to support agents?
    - Where would you hook in fraud or off-platform-payment detection without adding latency to sending?
38. Design a unified notification platform for guests, hosts and employees, covering event-triggered sends (an email when a guest books, a door lock code by SMS on check-in morning) and batch campaigns (promotions to prospective hosts). [Reported] [1][3][4][26][40]
    - How do you prioritize transactional notifications over marketing?
    - How do you de-duplicate, rate-limit per user and respect quiet hours across time zones?
    - What happens when a downstream provider (APNs, FCM, an SMS gateway) is slow or failing?
39. How would you design per-user notification preferences and channel fallbacks (push, then email, then SMS)? [Likely]
40. Design a customer-support ticketing system that routes tickets to the right agent queue. [Reported] [1][3][26][27]
    - How do you implement agent claim rules by language and location, and measure first-response time? [1]
    - How do you combine rules-based routing with an ML classifier, and fall back when the model is down?
41. Design the support ticket lifecycle: users file tickets, tickets move through created, pending assignment, assigned and resolved, and users and agents chat inside the ticket. [Reported] [25]
    - The prompt is broad. Which 70 percent would you design first if time ran short?
42. Design a ticket system with a watchlist or dashboard of open tickets and search by status, priority and creation date. [Reported] [21]
43. Design a system that generates customer-support instructions from company policies, for example with an LLM. [Reported] [3]
    - How do you keep generated instructions in sync when a policy changes?
    - How do you evaluate correctness before showing them to agents?
44. Design a system that sends reminder messages (check-in instructions, review requests) at specific times relative to each reservation. [Likely] [26]

### E. Payments, payouts and financial correctness

45. Design a payment system for hundreds of millions of daily active users, covering authentication, transaction processing and concurrency. [Reported] [15]
    - What is an idempotency key? Who generates it, and how long do you keep it? [9]
    - How do you separate the local DB transaction from the remote payment-provider call?
46. A payment service runs on two databases and can no longer handle the traffic. How do you add sharding? [Reported] [14]
    - How do you choose the shard key and migrate live data without double-charging or losing writes?
    - How do you handle cross-shard transfers and reporting queries after the split?
47. Design a digital wallet service (Google Wallet-style): stored balances, top-ups, transfers and transaction history. [Reported] [16][18]
    - Expect the interviewer to go deep on the APIs and the database schema rather than on scaling. What tables and constraints keep balances correct?
48. Design host payouts: scheduling, splits among co-hosts, multiple currencies and payout methods. [Likely] [1]
49. Design refunds across multiple payment methods and dates, honoring a priority order. [Likely] [10]
50. How do you make refunds idempotent when the refund request, the provider callback and an agent's manual action can all arrive more than once? [Likely] [1]
51. Design a double-entry ledger for Airbnb's payments platform. [Likely]
    - How do you reconcile your ledger against the payment processor's settlement files?
52. Design a banking-like service with deposit, withdraw, transfer and balance-history queries by time. How would you turn it into a distributed service that handles concurrent transactions? [Likely] [1][16]
53. Design a currency-conversion and price-display service that shows consistent prices from search through checkout. [Likely]
    - How do you lock an exchange rate for a quote?
54. Design a retryer or retry library for calls to unreliable downstream services. [Likely] [1][19]
    - How do you decide which errors are retryable?
    - How do you combine exponential backoff, jitter, deadlines and circuit breakers?
55. How would you design fraud checks into the payment path without hurting checkout latency? [Likely]

### F. Data, logging and analytics infrastructure

56. Design a highly interactive data analysis system. [Reported] [6]
    - What latency problem appears if you start with batch Spark processing?
    - How would you use Flink with Kafka for streaming aggregation?
    - How does recovery work when a Flink job fails (checkpoints, offsets, exactly-once)?
57. Design an ad-click (or listing-click) event aggregation system. [Reported] [3]
    - How do you handle late and duplicate events?
    - How do you reconcile real-time counts with batch-corrected counts?
58. Design a user event aggregation system, including its API, database, monitoring and SLOs. [Reported] [11]
    - Which SLOs would you commit to (freshness, completeness, query latency), and how would you alert on each?
59. Model cumulative booking metrics, including the first and last booking times per user or listing. [Reported] [3]
    - Would you compute this incrementally or recompute it? How do you backfill when the logic changes?
60. Design the logging pipeline that captures search impressions, clicks and bookings for ranking training data. [Likely]
    - How do you join impressions to bookings that happen days later?
    - How do you guarantee the logged features match what the model saw at serving time?
61. Design a metrics and dashboard system for marketplace health (bookings, nights, supply) with daily and real-time views. [Likely]
62. How would you design data-quality checks and alerting for core tables that feed models and experiments? [Likely]
63. An existing production system has a high error rate. Design changes to make it more reliable and faster. [Reported] [16]
    - How do you find where the errors come from before you redesign anything?

### G. Feature store and ML platform infrastructure

64. Design a feature store that serves offline training and low-latency online ranking from one feature definition. [Likely] [8]
    - How do you guarantee point-in-time correctness in training backfills?
    - How do you measure and alert on online/offline consistency?
65. Design the pipeline that measures online/offline feature consistency by logging online feature fetches, replaying their keys and timestamps as a backfill, and comparing the results. [Likely] [8]
    - At what sampling rate would you log, and how do you attribute a mismatch to a specific feature or pipeline?
66. How would you support windowed aggregations (for example a listing's bookings in the last 7 days) in both streaming and batch? [Likely] [8]
    - How do you handle very skewed keys, such as a popular listing or a bot user, in the aggregation?
67. Design the online serving layer of a feature store for a ranking request that fetches features for 1,000 candidate listings in under 50 ms. [Likely] [8]
    - Which key-value store would you choose? How do you batch, cache and tolerate tail latency?
68. A team adds a new feature to a ranking model already in production. Design the backfill, validation and rollout path. [Likely] [8]
69. How do you version features, and what happens to running models when a feature definition changes? [Likely]
70. Design a model-serving platform that supports canary releases, shadow traffic and rollback. [Likely]
71. Design an embedding-serving service that keeps listing embeddings fresh as listings and behavior change. [Likely]
72. Design the serving backend for host price suggestions: which parts are batch-computed, which are computed per request, and how do you keep them consistent with the calendar? [Likely] [42][43]
73. Design the serving path for listing recommendations ("similar listings"): what is precomputed, what is retrieved online, and how is it kept fresh? [Likely] [43]
74. Design a job scheduler for about ten million users. [Reported] [1][3][4][28][30][31]
    - How do you guarantee each job runs at least once, and when do you need exactly-once?
    - How do you scale the dispatcher and avoid one hot time bucket (for example midnight UTC)?
    - How would you shard the job database? How do queue features such as visibility timeouts and delayed delivery (for example in SQS) work underneath? [28]
75. Design a distributed job scheduler with task dependencies, priorities, high availability, retries and monitoring, for example for ML training and backfill DAGs. [Reported] [28][29]
76. A cache holds derived values that depend on other cache entries (a DAG). Design the invalidation and update path so that dependents are always correct when an input changes. [Likely] [44]

### H. Experimentation platform (backend view)

77. Design an A/B testing platform. [Reported] [3]
    - How do you assign users to variants deterministically and consistently across web, iOS and Android?
    - How do you handle logged-out users who later log in?
    - How do you log exposures so analysis is not diluted?
78. Design the assignment service's API and latency budget, given that every page load may evaluate dozens of experiments. [Likely]
    - Do you evaluate on the server or in a client SDK? How do you push config changes?
79. How would you support mutually exclusive experiment layers and holdouts? [Likely]
80. Design the metrics pipeline that computes experiment scorecards daily for hundreds of experiments and thousands of metrics. [Likely]
81. How would you design a kill switch or automated guardrail that stops an experiment when a key metric degrades? [Likely]
82. Design a network-aware experiment for a "Wi-Fi badge" on listings, where treatment effects can spill over between listings. [Reported] [3]
    - How would the platform support cluster or market-level randomization?

### I. Generic distributed-systems building blocks (asked at Airbnb)

83. Design a distributed key-value store. [Reported] [1][23][24]
    - Walk through the CAP trade-offs, consistent hashing and vector clocks. [1]
    - How do you support get, put, delete and a paginated prefix scan (get(key*)) under eventual consistency and high availability? [23]
    - How do you handle node failure, rebalancing and fault recovery? [24]
84. Design a highly reliable, highly available distributed key-value system that supports bulk upload, multiple versions per key and TTLs. [Reported] [22]
    - How does a bulk upload become visible atomically, and how do you garbage-collect expired versions?
85. Design an in-memory SQL engine. Agree on scope first. [Reported] [1]
    - Which subset of SQL (select, where, join, group by) do you commit to, and why?
86. Design an online two-player chess game. [Reported] [3][4][38]
    - What does the API look like? How do you sync game state in real time and handle disconnects and clocks?
87. Design a rate limiter for Airbnb's public APIs and internal services. [Likely]
88. A partner API complains that your service calls it too often. Redesign the integration to batch the calls without breaking freshness guarantees. [Likely] [19][41]
89. Design a distributed cache in front of the listing service, and explain invalidation when a host edits a listing. [Likely]
90. Design account merge and user de-duplication for people who create multiple accounts. [Likely] [1]
    - How do you merge reservations, reviews and payment methods safely?

### J. Other Airbnb product surfaces

91. Design the reviews system: double-blind reviews revealed only when both parties submit or the window closes. [Likely]
    - How do you implement the reveal-at-deadline logic reliably?
92. Design the listing creation and photo upload pipeline, including image processing and moderation. [Likely]
93. Design Experiences booking with time slots, group capacity and a waitlist. [Likely] [3]
94. Design a multi-traveler itinerary service that combines stays and experiences, for example filling a layover with experiences. [Likely] [10]
95. Design a relisting detection system that catches a banned host relisting the same property under a new account. [Reported] [2][3]
    - How do you use geohashing and photo or text similarity at scale?
96. Design a listing lifetime-value estimation service and its batch and online parts. [Reported] [2][3][42]
97. Design the trust and safety screening pipeline that scores every booking before confirmation. [Likely]
    - How do you keep it off the critical path where possible, and do you fail open or closed when it is down?

### K. Cross-cutting senior probes (apply to any prompt)

98. What are your back-of-the-envelope numbers for this design (QPS, storage, peak-to-average ratio)? [Likely] [1][17]
99. You have 45 minutes. Which two components would you deep-dive, and why those? [Likely] [1]
100. Write out the API and the core tables before you draw any boxes. Which queries does each index serve? [Likely] [1][16][19]
101. How do you deploy a schema change to the reservations table with zero downtime? [Likely]
102. How would you run this system across multiple regions? What happens during a region failover in the middle of a booking? [Likely] [17]
103. What metrics, alerts and SLOs would you define for this service? [Likely] [11]
104. Which part of this design would you A/B test, and what could go wrong with the test because of the architecture (caching, sticky assignment, logging gaps)? [Likely]
105. How would you migrate from the existing monolith to this service-oriented design incrementally? [Likely]
106. If traffic grew 10x tomorrow, what breaks first? [Likely]
107. What would you simplify if you had to ship a first version in two weeks? [Likely] [1]
108. How do you secure personal data (addresses, IDs, payment details) in this design, and who can access it? [Likely]

## References

All sources were fetched on 2026-10-14 except [9], which could not be reached (medium.com did not resolve) and is cited from memory. Sources [11]-[45] are individual Airbnb interview reports from 1point3acres, translated and mirrored in the FlowSight/coding GitHub repository. Each one was read in full.

1. "Airbnb Software Engineer Interview Process" (based on 41 candidate reports, mostly 2025 Q4 to 2026 Q2), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
2. "Airbnb Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
3. "Airbnb" question list (system design titles with report dates, Nov 2025 to Sep 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
4. "Airbnb" company overview (prompts by round type), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/README.md
5. "Airbnb" company prep page (Jan 2026 senior SWE booking and hot-partition report, Jan 2026 group chat report, G8 loop), jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
6. "Airbnb interview questions" (Problem 10: data analysis system design), warrenzhu25/system-design, GitHub. https://github.com/warrenzhu25/system-design/blob/master/airbnb_interview_questions.md
7. "Interview formats of top companies", Tech Interview Handbook (yangshun), GitHub. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
8. "Chronon: a data platform for AI/ML" (Airbnb open-source feature platform: point-in-time backfills, windowed aggregations, online/offline consistency measurement), airbnb/chronon, GitHub. https://github.com/airbnb/chronon
9. "Avoiding Double Payments in a Distributed Payments System", The Airbnb Tech Blog (Medium), 2019. Not verified in this session (domain unreachable). https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-2981f6b070bb
10. Airbnb 1point3acres coding prompts, including "Refund Allocation by Payment Method and Date Priority" and "Fill Layover Hours by Booking Experiences" (folder listing), FlowSight/coding, GitHub. https://github.com/FlowSight/coding/tree/main/company%20wise/airbnb/1p3a
11. Airbnb onsite report: "Design a user event aggregation system, including API, database, monitoring, and SLO", FlowSight/coding (1p3a #101). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/101.md
12. Airbnb phone screen report: 5-minute design question on the data flow after a user modifies their information, FlowSight/coding (1p3a #30). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/30.md
13. Airbnb infrastructure-team onsite report: full-text search system design, and flight booking design with an L7 interviewer (PA versus PC trade-off), FlowSight/coding (1p3a #70). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/70.md
14. Airbnb onsite report: "payment service with two databases ... How do we add shards?", FlowSight/coding (1p3a #71). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/71.md
15. Airbnb onsite report: payment system design for 300M DAU, FlowSight/coding (1p3a #17). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/17.md
16. Airbnb onsite report, Oct to Dec 2025: system design rounds on Google Wallet and on improving a system with a high error rate; the interviewer went deep on APIs and database design, FlowSight/coding (1p3a #16). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/16.md
17. Airbnb report with a detailed group-only IM design prompt (large groups, fan-out, delivery semantics, ordering, disaster recovery), FlowSight/coding (1p3a #6). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/6.md
18. Airbnb onsite report: "SD: Wallet service", FlowSight/coding (1p3a #67). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/67.md
19. Airbnb onsite report: listing booking with a waitlist notifying the first 20 users, with pricing, search and notification services given; retryer coding; code review on pagination and batching calls to a partner API, FlowSight/coding (1p3a #69). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/69.md
20. Airbnb onsite report: "Design a hot listing waitlist API", FlowSight/coding (1p3a #58). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/58.md
21. Airbnb (China office) senior report, 2023: ticket system with a watchlist or dashboard and filter search; failed on design against the senior bar, FlowSight/coding (1p3a #76). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/76.md
22. Airbnb Infrastructure senior report: distributed KV system with bulk upload, multiple versions and TTL; rejected because the design was not at senior level, FlowSight/coding (1p3a #62). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/62.md.md
23. Airbnb senior onsite report: distributed key-value store with paginated get(key*), eventual consistency and high availability, FlowSight/coding (1p3a #29). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/29.md
24. Airbnb staff-level report: key-value store design (sharding, fault recovery) in the first round, FlowSight/coding (1p3a #94). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/94.md
25. Airbnb report, Oct to Nov 2025 (4.5 YOE): ticketing system with routing, user-agent chat and statuses; an added design round on booking and double-booking prevention; slow page for hosts with more than 100 listings; split-stay screen, FlowSight/coding (1p3a #11). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/11.md
26. Airbnb senior report: customer ticket management system design, plus the Airbnb notification-system prompt (booking email, batch host promotions, lock-code SMS on check-in day), FlowSight/coding (1p3a #13). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/13.md
27. Airbnb report: two system design rounds, on a booking system and a customer ticket handling system, FlowSight/coding (1p3a #18). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/18.md
28. Airbnb onsite report, Jul to Sep 2025: job scheduler (DB sharding, SQS visibility timeout and delay scheduling) with dependencies, priorities and HA, FlowSight/coding (1p3a #24). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/24.md
29. Airbnb Staff SWE (L6) report: "Design Distributed Job Scheduler" and "Design Airbnb's booking system", FlowSight/coding (1p3a #78). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/78.md
30. Airbnb onsite report: system design rounds on group chat and a job scheduler, FlowSight/coding (1p3a #80). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/80.md
31. Airbnb report: "System design, job scheduler", FlowSight/coding (1p3a #85). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/85.md
32. Airbnb senior report: chat system with heavy schema focus and top-N group chat optimization, WebSocket routing and disaster recovery; failed against the senior bar, FlowSight/coding (1p3a #10). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/10.md
33. Airbnb report: chat design focused on the schema for "only displaying the top 20 recent chats", FlowSight/coding (1p3a #72). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/72.md
34. Airbnb onsite report: typeahead search design, and group chat with an inbox ordered by the most recent message, FlowSight/coding (1p3a #93). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/93.md
35. Airbnb onsite report, 2025: group chat schema for the top 20 chats, and notifications to online and offline users (long polling, WebSockets, async queues), FlowSight/coding (1p3a #104). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/104.md
36. Airbnb onsite report (August): group chat design focused on database tables and the query functions, FlowSight/coding (1p3a #28). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/28.md
37. Airbnb senior report, booking and orchestration team: group chat system with a focus on DB design, FlowSight/coding (1p3a #73). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/73.md
38. Airbnb onsite report: two-player chess game design; feedback was that the API design needed too many hints, FlowSight/coding (1p3a #39). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/39.md
39. Airbnb system design question: Google Calendar with recurring events and a schema for fast queries, FlowSight/coding (1p3a #102). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/102.md
40. Airbnb onsite report: architecture round on notification systems, FlowSight/coding (1p3a #99). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/99.md.md
41. Airbnb code review round report: PRs on adding pagination, input validation and batched calls to an external API, FlowSight/coding (1p3a #88). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/88.md
42. Airbnb MLE report: ML design rounds on LTV and a host pricing system, FlowSight/coding (1p3a #40). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
43. Airbnb MLE report: two ML design rounds, on listing recommendation and price prediction, FlowSight/coding (1p3a #79). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
44. Airbnb infrastructure report: coding on updating a cache with dependent entries (a DAG), FlowSight/coding (1p3a #35). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/35.md
45. Airbnb onsite report: "classic design group chat", approached like a WhatsApp design, FlowSight/coding (1p3a #3). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/3.md
