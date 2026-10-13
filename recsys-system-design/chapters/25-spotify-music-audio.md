# 25. Spotify and Audio Recommendation: Discover Weekly, BaRT, Podcasts, AI DJ, LLM Playlists, Deezer Flow, Pandora, and the Music Streaming Stack

> Editorial note: the live web tools were unavailable while this chapter was assembled (the
> search budget was used up and outbound fetches were blocked). The chapter therefore relies
> on well-known primary publications (papers with stable arXiv/DOI identifiers, official
> engineering posts, and open-source repositories) that were already established before
> drafting. Numbers are given only where those sources report them and are marked as
> approximate where memory of the exact figure is uncertain. Readers should check figures
> against the cited originals before quoting them.

Music and audio recommendation differs from video or e-commerce recommendation in ways that
shape the whole architecture:

- **Consumption is fast and repeated.** A track lasts about 3 minutes, and users listen to the
  same items hundreds of times. Repeated consumption is the normal case, unlike in
  most domains.
- **Sessions are sequences.** Playlists, radio, autoplay and "Flow" style endless streams make
  the unit of recommendation a *sequence* with transitions, not a single item.
- **Implicit feedback dominates.** Streams, skips (often within the first 30 seconds), saves,
  playlist adds and "ban" actions replace ratings. Skips are a strong, plentiful negative
  signal that most domains lack.
- **Huge, long-tailed catalog with constant cold start.** Spotify reports a catalog of more
  than 100 million tracks, with tens of thousands of new uploads per day. Content-based
  audio models are therefore essential, not optional.
- **Multiple content types.** Music, podcasts and audiobooks share one surface (Home) but
  have very different consumption patterns and data density.
- **Lean-back and voice surfaces.** Radio, DJ and car/speaker contexts need recommendations
  that work with almost no explicit input. This drove the shift toward LLM-narrated
  and LLM-prompted experiences from 2023 onward.

The chapter goes from foundational Spotify components (implicit matrix factorization,
content CNNs, ANN search) to the Home page (BaRT bandits, shelves), user representations and
sequence models, podcasts and audiobooks (graph neural networks), the LLM era (AI DJ, AI
Playlist, generative retrieval), and then other companies: Deezer, Pandora, YouTube Music,
Apple Music and SoundCloud.

---

## 1. Spotify's foundation layer: implicit collaborative filtering and embeddings

### 1.1 Problem

Spotify's early personalization, from the Echo Nest era through Discover Weekly's 2015 launch,
had to turn billions of play events into a "taste space" in which users, tracks, artists and
playlists could be compared. Ratings did not exist, so the system had to learn from
*implicit* counts.

### 1.2 Implicit ALS and Logistic MF

The canonical starting point is implicit-feedback matrix factorization (Hu, Koren and
Volinsky) [3]. The model treats the play count r_ui as a *confidence* c_ui = 1 + alpha * r_ui
on a binary preference p_ui, and solves a weighted least-squares problem with alternating
least squares (ALS). ALS parallelizes cleanly on Hadoop/Spark because each user (or item)
vector has a closed-form update once the other side is fixed.

Chris Johnson, then at Spotify, published **Logistic Matrix Factorization** [2]. It models the
probability that a user listens to an item as a logistic function of the user and item
vector dot product plus user and item biases, and weights positives by confidence as in
implicit ALS. Johnson's public talks (circa 2014-2015) described Spotify running
ALS-style factorization over a matrix with tens of millions of users and tens of millions of
tracks on Hadoop. Those numbers have since grown by an order of magnitude.

### 1.3 Discover Weekly as a pipeline

Discover Weekly (launched 2015) is often described as "collaborative filtering", but public
talks by Spotify engineers describe a multi-source pipeline:

```
  play logs, playlist co-occurrence (user-curated playlists as "documents")
        |                          |
        v                          v
  implicit MF / word2vec-style   playlist-track co-occurrence
  track & user vectors           (tracks that appear together)
        |                          |
        +-----------+--------------+
                    v
        candidate tracks near the user's taste vector(s)
        (ANN search: Annoy, later Voyager)
                    |
   filters: not already heard, artist diversity, explicit-content
   settings, cold-start audio-model candidates
                    |
                    v
         ranking + sequencing into a 30-track weekly playlist
                    |
                    v
           batch generation for every user, written to storage,
           served on Monday (precomputed, not real-time)
```

Key design points:

- **Playlists as a supervision source.** Billions of user-created playlists encode human
  judgments of which tracks "go together". Treating a playlist as a sentence and a track as
  a word (skip-gram / word2vec style) gives track embeddings that capture context and mood,
  not just co-listening. Spotify later released the **Million Playlist Dataset** for the
  RecSys Challenge 2018 on automatic playlist continuation [13], which shows how central
  this signal is.
- **Batch, not online.** A weekly playlist is a perfect batch workload. Generation runs
  offline for every user, so the per-user compute budget can be large.
- **Multiple taste clusters per user.** Users have several distinct tastes (for example
  workout music and jazz). Practitioner descriptions of Discover Weekly and later Spotify
  systems describe clustering a user's listening into several sub-tastes and recommending
  from each one, rather than from a single averaged vector, which would land between them.

### 1.4 Content-based audio models for cold start

New tracks have no listening data. In 2014, Sander Dieleman (Spotify intern, later DeepMind)
described training convolutional networks on mel-spectrograms to *predict the
collaborative-filtering latent factors* of a track from its audio [4]. This approach
builds on van den Oord, Dieleman and Schrauwen's "Deep content-based music recommendation"
(NeurIPS 2013) [5]. The trick is to use the CF embedding as the regression target, so the
audio model learns to place a brand-new track where the CF model *would* have placed it.
That makes cold-start tracks directly retrievable from the same ANN index.

```
 audio (30s clips) -> mel-spectrogram -> conv layers (time-pooled)
                                             |
                                             v
                            predicted latent vector  ~=  CF vector (from ALS/LMF)
                                             |
          loss: MSE / weighted-MSE to the CF factors of tracks with enough plays
          use: embed tracks that have no plays -> same retrieval space
```

The Echo Nest acquisition (2014) brought content analysis (tempo, key, danceability,
energy and similar attributes) and text/web mining of artist descriptions. These features
were long exposed through Spotify's public Web API "audio features" endpoint. Spotify
deprecated that endpoint for new third-party apps in late 2024, but the features remain
internal model inputs.

### 1.5 Nearest-neighbor search: Annoy to Voyager

Every Spotify retrieval stage depends on approximate nearest-neighbor search over
embeddings.

- **Annoy** ("Approximate Nearest Neighbors Oh Yeah", Erik Bernhardsson, open-sourced
  around 2013) [6] builds a forest of random-projection trees. Each split is a hyperplane
  equidistant between two sampled points. Indexes are static, memory-mapped files, so many
  processes can share one index and indexes can be built offline and shipped to servers.
  This fit Spotify's batch-build, read-heavy serving model.
- **Voyager** (open-sourced October 2023) [7][8] replaced Annoy for most use cases. It is
  based on hnswlib (HNSW graphs), with Python and Java bindings. Spotify reported
  it as roughly 10x faster than Annoy at the same recall, or up to about 50% more
  accurate at the same speed, and roughly 4x less memory through E4M3 8-bit float
  storage (approximate figures from the announcement [7]). It also supports adding
  items after the index is built, which Annoy cannot do.

Lesson: the ANN library was a *platform* decision. Because every team embedded into
vector spaces and served by ANN, one library upgrade improved dozens of products at once.

---

## 2. Spotify Home: BaRT and explainable bandit recommendations

### 2.1 Problem

The Home page is a 2-D grid. Rows ("shelves", such as "Jump back in", "Made for you",
"Because you listened to X") hold cards (playlists, albums, podcasts, artists). Home must
choose *which shelves*, *in which order*, *which cards* and *with which explanation*,
for hundreds of millions of users, while avoiding a feedback loop in which only
already-popular cards get exposure.

### 2.2 BaRT: Bandits for Recommendations as Treatments

McInerney et al. (RecSys 2018) [1] introduced **BaRT** (Bandits for Recommendations as
Treatments). It is a contextual bandit that jointly personalizes the *item* and the
*explanation* attached to it (for example "Because you like jazz" or "Popular right now").
Main ideas:

- **Recommendation as treatment.** Each (item, explanation) pair is an action. The reward is
  a binary success signal, which on Home means the user played the card's content for at
  least a minimum duration (on the order of 30 seconds).
- **Model.** A factorization-machine-style reward model predicts the success probability
  from user, item, explanation and context features, including second-order interactions
  between explanation and user/item. Explanations matter more to some users than others,
  and the interaction terms capture that.
- **Exploration.** Epsilon-greedy over the predicted reward: with probability epsilon a
  random (item, explanation) is shown, and the rest of the time the greedy best. The paper
  also stresses *propensity logging*. Because exploration probabilities are known, the
  logs support **counterfactual (off-policy) evaluation and training** with inverse
  propensity scoring.
- **Results.** Offline counterfactual evaluation and online tests in the paper showed
  that personalizing explanations together with items improved streaming success over
  item-only personalization.

```
          user context (time, device, history embeddings)
                         |
     candidate cards  x  candidate explanations ("Because you listened to ...",
     (playlists,          "Your top mixes", "Popular podcasts", ...)
      albums, shows)
                         |
                         v
          reward model  P(stream >= ~30s | user, card, explanation, ctx)
          (factorization machine with explanation interactions)
                         |
               epsilon-greedy policy  ---> log (action, propensity, reward)
                         |                         |
                         v                         v
                  rendered shelf            off-policy evaluation and
                                            retraining (IPS-weighted)
```

### 2.3 Home architecture evolution

Spotify's 2020 engineering post "For Your Ears Only" [9] describes how Home
personalization moved from heuristic shelf ordering to ML: offline training on logged
interactions, a split between card ranking within shelves and shelf ranking on the page,
and a shift from batch-precomputed Home layouts toward request-time ranking so that Home
can react to time of day and recent listening. The broader lessons, which echo BaRT:

1. **Log propensities from day one.** Off-policy evaluation is only possible if the serving
   system records how likely each shown item was to be shown.
2. **Separate the "what" from the "why".** Explanations ("Because you listened to...") are
   a separate decision with their own measurable effect.
3. **Satisfaction is not clicks.** Related Spotify work by Mehrotra and colleagues on Home
   models *user intent* (for example, quickly playing something familiar versus exploring)
   and its relationship to satisfaction. It also studies the trade-off between relevance,
   artist-side fairness/exposure and user satisfaction using counterfactual estimators
   [15].

### 2.4 Diversity and the "generalist vs specialist" finding

Anderson et al. (WWW 2020) [16] analyzed listening by roughly 100 million Spotify users.
They found that users with more diverse ("generalist") listening were more likely to
retain and convert to paid subscriptions. They also found that *algorithmically driven*
listening was, on average, less diverse than *organic* (user-driven) listening, and that
users drifted toward organic, more diverse listening over time. The design consequence is
that diversity is not just a fairness constraint. It correlates with business outcomes,
so it belongs in the objective or in re-ranking constraints.

---

## 3. User representations and sequence models at Spotify

### 3.1 CoSeRNN: contextual and sequential user embeddings

Hansen et al. (RecSys 2020) [12] proposed **CoSeRNN**. It models a user as a *sequence of
session-level preferences* rather than one static vector. For each new session, an RNN over
past sessions plus the current context (time, device, and so on) produces a *session
embedding* that lives in the same space as pretrained track embeddings. Candidates are then
retrieved by ANN in that space. The paper reports large gains in session-level ranking
metrics over static and purely sequential baselines. The design principle is "predict an
embedding, then use ANN". The network outputs a point in the item space, which keeps
serving cheap and reuses the existing track index.

### 3.2 Generalized user representations (2024)

Fazelnia et al. (Spotify, 2024) [10] describe a **two-stage user representation
framework** used across many Spotify products:

```
 Stage 1: content & interaction encoders           Stage 2: user aggregation
 ---------------------------------------           -------------------------
 audio / metadata / CF  -> item embeddings  ---->  user listening history (items,
 (per-entity pretrained)                           weighted by recency/engagement)
                                                          |
                                                          v
                                              autoencoder -> compact user vector
                                              (refreshed frequently, stored in a
                                               feature/embedding store)
                                                          |
                         +---------------+----------------+---------------+
                         v               v                v               v
                    Home ranking    search ranking   podcast recs    ads / other
                    (downstream models consume the general vector as a feature)
```

Key points from the paper: one general-purpose user vector is consumed by many downstream
models (transfer learning). The representation is updated frequently enough to reflect
recent listening. Decoupling representation learning from task models reduced duplicated
modeling effort across teams and improved downstream metrics, including cold-start
performance. Infrastructure is a large part of the contribution: a shared embedding store,
a scheduled refresh, and versioning so downstream consumers do not break when the
representation is retrained.

### 3.3 Skips and sessions as data

Spotify released the **Music Streaming Sessions Dataset** (Brost et al., WWW 2019) [14]:
about 160 million listening sessions with skip labels, plus acoustic features for about
3.7 million tracks, used for the WSDM Cup 2019 Sequential Skip Prediction Challenge. The
dataset confirms that Spotify treats *skip prediction within a session* as a core
modeling task. Skip probability is the natural short-horizon reward for radio,
autoplay and DJ.

---

## 4. Podcasts and audiobooks: cold-start domains with graph models

### 4.1 Problem

When Spotify added podcasts (from around 2019) and audiobooks (2022-2023), it faced
*sparse* domains. Most users had little or no podcast or audiobook history, while music
data was abundant. The goal is to transfer taste from music to spoken word.

### 4.2 Podcast content understanding

For podcasts, content understanding is mostly *text*: transcripts, titles and episode
descriptions. Spotify released the **Spotify Podcast Dataset** (Clifton et al.,
COLING 2020) [17]: about 100,000 episodes with automatically generated transcripts
(roughly 50,000 hours of audio). It supported TREC podcast tracks on segment retrieval
and summarization. Transcripts feed topic embeddings, which provide a content-based
retrieval path for new shows that has the same role as the audio CNN does for music.

### 4.3 2T-HGNN for audiobooks (WWW 2024)

De Nadai et al. [11] describe the audiobook recommender that Spotify launched at scale.
Its architecture is **2T-HGNN**: a heterogeneous graph neural network combined with a
two-tower model.

```
  heterogeneous graph:  audiobooks -- podcasts -- (co-listened / co-saved edges)
                        nodes carry LLM-based text embeddings of title/description
                               |
                               v
               HGNN (GraphSAGE-style message passing) -> item embeddings
                               |
  user features: music + podcast + audiobook  -->  user tower
  history, demographics                            |
                               +-------------------+
                               v
                two-tower (2T) model: user vec . item vec
                               |
                   ANN retrieval -> downstream ranking on Home / search
```

Design decisions reported in the paper [11]:

- **The graph is item-item only, not user-item.** Including hundreds of millions of user
  nodes would be costly to train and serve. The HGNN therefore operates over items
  (audiobooks and podcasts), learning content and co-engagement structure. User
  personalization is handled by a separate, cheaper user tower. This decoupling is the
  main scalability trick.
- **Cross-domain transfer.** Podcast co-listening edges carry information into the
  sparse audiobook domain, and music listening features in the user tower help users with
  no spoken-word history.
- **Low-latency serving.** Item embeddings are precomputed offline. Online, only the user
  tower runs, followed by an ANN lookup.
- **Results.** Large online A/B tests (millions of users) showed significant gains in
  audiobook start rate (approximately +46%) and streaming rate (approximately +23%), along
  with gains in podcast metrics. These are approximate figures from the paper's abstract.

---

## 5. The LLM era: AI DJ, AI Playlist and generative retrieval

### 5.1 Spotify AI DJ (2023)

Launched in February 2023 as a beta in the US and Canada [18], **DJ** is a lean-back radio
experience. Spotify's recommender picks and sequences tracks, and a generative layer
narrates between segments in a synthetic voice modeled on Spotify's head of cultural
partnerships, Xavier "X" Jernigan. Public descriptions from Spotify identify three
building blocks [18]:

1. **Personalization tech** (the existing recommender stack) chooses blocks of tracks:
   recent favorites, older tracks the user has not played for a while, and new
   discoveries.
2. **Generative AI (OpenAI LLM technology)** writes short commentary scripts. Spotify's
   music editors supply domain knowledge and style through prompts and curated facts, so
   the commentary is grounded, not free-form.
3. **Dynamic AI voice** technology (from the Sonantic acquisition, 2022) turns scripts into
   realistic speech.

```
 user history -> recommender -> "segments" (theme + 4-8 tracks)  [precomputable]
                                      |
          editor-curated facts/ ----> LLM script generation (short, templated,
          style guides                checked/filtered before use)
                                      |
                                TTS voice model (Sonantic-derived)
                                      |
               client: [voice intro][track][track]...[voice transition][track]...
               user taps DJ button -> skip whole segment -> new theme
```

Design lessons:

- **The recommender is still the core.** The LLM explains and frames, and the retrieval and
  ranking stack chooses. That keeps quality controllable and latency predictable.
- **Segment-level feedback.** Tapping the DJ button to switch the vibe is a coarse
  (segment-level) negative signal, a new kind of feedback compared to track skips.
- **Editorial grounding.** Human editors feed facts and style into the prompt layer to
  reduce hallucination about artists. The same pattern appears in Netflix and Amazon
  LLM explanations.
- Later updates (2024-2025) added voice requests to DJ ("play something for a rainy
  afternoon") and more languages, such as a Spanish-language DJ. Requests make DJ a
  conversational recommender, not just a narrated radio.

### 5.2 AI Playlist (2024) and prompt-based playlist generation

**AI Playlist** (beta from April 2024, first in the UK and Australia, later expanded [19])
lets a user type a prompt such as "indie folk for a cozy autumn evening" and receive a
playlist that can be refined with follow-up prompts. Spotify has not published a full
system paper on the production feature, but Spotify Research papers show the techniques
under study:

- **Prompt-to-attributes plus retrieval.** An LLM parses the prompt into structured intents
  (genres, moods, eras, activities, and reference artists or tracks). These map to
  existing retrieval channels (tag/mood classifiers, embedding search, editorial
  playlists), and the user's taste vector personalizes the result.
- **Generative retrieval.** **Text2Tracks** (Palumbo et al., Spotify, 2025) [20] fine-tunes
  an encoder-decoder LLM to *generate track identifiers* directly from a text prompt. The
  key finding is that **ID design matters**. Generic or random integer IDs, or titles
  used as IDs, perform worse than **semantic IDs** derived from collaborative-filtering
  embeddings, such as hierarchical clustering of track embeddings into short code
  sequences. Generating a short semantic ID is also far cheaper than generating titles and
  then resolving them, which avoids hallucinated tracks.

```
 prompt "upbeat 80s synthpop for running"
     |
     v
 LLM (fine-tuned seq2seq) --decode--> semantic IDs  e.g. <c12><c7><c3>  (per track)
     |                                   |  (IDs from clustering CF embeddings)
     |                                   v
     |                        ID -> track lookup (no fuzzy title matching)
     v
 personalization re-rank with user vector, dedupe, artist diversity, sequencing
```

### 5.3 LLM-generated narratives and explanations

Spotify Research has also published work on using LLMs to generate **personalized
contextual explanations** ("narratives") for recommendations on Home, for example why a
podcast or audiobook fits the user. This continues the BaRT idea that the explanation is
part of the treatment. The common production pattern is to precompute LLM text offline
for (item, user-segment) pairs, filter it with safety and quality classifiers, and serve
it from cache, because per-request LLM generation for hundreds of millions of users is too
expensive. Note: the specific paper titles and figures in this sub-area could not be
re-verified while drafting this chapter, so they are not cited here.

---

## 6. Deezer: Flow, cold-start onboarding and carousel bandits

Deezer's research team (Paris) publishes unusually detailed production papers, which makes
it one of the best public sources on music recommendation at scale. Deezer reports a
catalog of more than 100 million tracks and on the order of 10 million monthly active users.

### 6.1 Flow

**Flow** is Deezer's personalized, endless radio and the app's flagship recommendation
surface. It mixes favorites with discoveries in real time, and later variants added
moods ("Flow Moods": focus, chill, energy, and similar). Architecturally it is a session
recommender: candidate generation from user embeddings and track similarity, re-ranked
continuously from in-session feedback (skips, likes) and mood constraints.

### 6.2 Semi-personalized cold start (KDD 2021)

Briand et al. [21] describe Deezer's system for **new users**, in production on the
Flow surface:

```
  existing users: embeddings u from MF on listening data
          |
          v
  k-means over warm users  ->  K clusters (each with precomputed top tracks)
          |
  new user registers: demographics (country, age), onboarding answers
  (selected artists/genres), first-day interactions
          |
          v
  feedforward network predicts the user's embedding u_hat (trained on warm users:
  input = their registration-time features, target = their eventual embedding)
          |
          v
  assign u_hat to nearest cluster -> serve that cluster's precomputed recs
          (semi-personalized: cheap, cacheable, robust to noisy u_hat)
```

Why it works: predicting the exact embedding of a brand-new user is noisy, so snapping
to a cluster *regularizes* the prediction and makes serving a cache lookup.
Online A/B tests showed significant improvements in new-user engagement and retention
compared with the previous popularity- and onboarding-based approach [21].

### 6.3 Carousel personalization with contextual bandits (RecSys 2020)

Bendada, Salha and Bontempelli [22] model Deezer's Home **carousels** (swipeable shelves
of playlists, about 12 items with only a few visible at once) as a contextual
multiple-play bandit. The contributions:

- **Cascade-style position model.** Only the first L cards are seen without swiping, so
  unseen cards should not be counted as negatives. Their "cascade-based" update treats
  cards below the last clicked position (or beyond the visible window) as unobserved,
  not as rejected.
- **Semi-personalization again.** Users are clustered, and bandit parameters are shared
  within clusters. This speeds learning compared with a fully per-user bandit.
- Thompson sampling and UCB variants were compared on a released dataset of about
  1,000 curated playlists, with user features, released for reproducibility.

### 6.4 Track Mix and new-release discovery

Deezer's **Track Mix** feature (Bendada et al., RecSys 2023) [23] generates a mix seeded by
a single track using a Transformer trained on the sequences of millions of user
playlists. The model predicts the next tracks given the seed, and constraints are applied
for diversity and catalog availability. The paper reports online gains over the earlier
similarity-based mix in streams and engagement. Deezer has also published on
**fostering the discoverability of new releases** [24], which deals with the cold start of
fresh releases on the Home page through content features and exploration.

---

## 7. Pandora: the Music Genome Project and thumbs-based radio

### 7.1 Music Genome Project

Pandora (founded as Savage Beast in 2000, now part of SiriusXM) built the **Music Genome
Project** [25]. Trained musicologists hand-annotate each song on up to roughly 450
musical attributes ("genes"), such as melody, harmony, rhythm, instrumentation,
form, vocal characteristics and lyrics. Annotating a song reportedly takes on the order of
20-30 minutes of expert time (approximate, from public descriptions).

```
 seed (artist/song) -> gene vector (expert-annotated, ~hundreds of attributes)
          |
          v
 candidate songs by weighted gene-space similarity
          |
 + collaborative signals from thumbs up/down across listeners (added over time)
 + per-station adaptation: thumbs on this station reweight genes/candidates
          |
          v
 ensemble ranker -> next song (licensing constraints: DMCA radio rules limit
                   skips and how often an artist/album can repeat)
```

### 7.2 Design lessons

- **Expert content features remove the item cold start** for annotated songs, but
  annotation throughput limits catalog coverage. That is why Pandora's catalog has been far
  smaller than Spotify's, and why Pandora later combined genome features with
  collaborative filtering and other machine-learned signals in an ensemble.
- **Thumbs as explicit binary feedback** scoped to a *station*. The same user can like a
  song on one station and dislike it on another, so feedback is contextual by design. This
  anticipates modern contextual and session-aware recommenders.
- **Regulatory constraints shape the ranker.** Non-interactive radio licensing restricts
  skips per hour and artist repetition. These rules are hard constraints in the sequencer,
  not soft objectives.

---

## 8. YouTube Music, Apple Music and SoundCloud

### 8.1 YouTube Music: Transformers over user actions

Google Research described (2024) how YouTube Music's ranking uses a **Transformer over the
user's recent actions** to interpret *context-dependent* actions [26]. The motivating
example: a skip during a workout session means something different from a skip while
relaxing. A Transformer reads the sequence of (track, action type, duration) tokens.
Attention learns which past actions are relevant to the current context, and the output
is combined with the existing ranking model (it does not replace the whole ranker).
Google reported reduced skip rates and increased listening time in live experiments.
YouTube Music otherwise inherits YouTube's recommendation stack (candidate generation
plus deep ranking, multi-task objectives; see Chapter 17), adapted to music-specific
signals such as skips, repeats and the music-versus-video catalog split.

### 8.2 Apple Music

Apple has published very little system-level detail about Apple Music's recommenders.
Public material describes the product (editorially curated playlists, personalized
"For You"/"Listen Now" mixes, and station-based radio) but not the models, so this book
does not attribute specific architectures to it. The notable *design* stance is strong
human editorial curation combined with personalization, and on-device processing of
some personalization signals, consistent with Apple's broader privacy approach.

### 8.3 SoundCloud

SoundCloud's catalog is dominated by user uploads (hundreds of millions of tracks,
including remixes, DJ sets and unreleased work). Its recommenders (stations, "Discover"
mixes, related tracks) must lean heavily on graph signals (follows, reposts, likes) and
content features, because metadata is noisy and much of the catalog is permanently cold.
SoundCloud's engineering blog has historically discussed its data and streaming
infrastructure more than model details. No production model paper was verified for this
chapter, so specifics are omitted.

---

## 9. Cross-company comparison

| System | Primary surface | Core signals | Retrieval | Ranking / policy | Cold start answer |
|---|---|---|---|---|---|
| Spotify Discover Weekly | Weekly 30-track playlist | Plays, playlist co-occurrence | MF / word2vec vectors + ANN (Annoy -> Voyager) | Batch scoring, diversity filters | Audio CNN predicts CF vector [4] |
| Spotify Home (BaRT) | Shelves of cards | Streams >= ~30s, context | Many candidate sources | FM reward model + epsilon-greedy, IPS logging [1] | Exploration budget |
| Spotify CoSeRNN | Session start | Session sequences, context | Predicted session embedding + ANN | RNN over sessions [12] | Pretrained track embeddings |
| Spotify audiobooks (2T-HGNN) | Home, search | Podcast/audiobook co-engagement, music history | Two-tower + ANN | HGNN item embeddings [11] | LLM text embeddings, cross-domain edges |
| Spotify AI DJ | Lean-back radio | All of the above | Existing recsys segments | Recsys + LLM narration + TTS [18] | N/A (uses existing stack) |
| Spotify Text2Tracks | Prompted playlists | Text prompt, CF embeddings | Generative retrieval of semantic IDs | Personalized re-rank [20] | Semantic IDs share codes with similar tracks |
| Deezer Flow onboarding | Endless radio | Registration features, onboarding picks | Cluster lookup | Precomputed per-cluster lists [21] | Predict embedding -> snap to cluster |
| Deezer carousels | Home carousels | Clicks/streams within visible window | Curated playlists pool | Cascade contextual bandit (TS/UCB) [22] | Semi-personalized clusters |
| Deezer Track Mix | Seeded mix | Playlist sequences | Transformer next-track | Constrained decoding [23] | Seed-track context |
| Pandora | Stations | Expert genes, thumbs | Gene-space similarity + CF | Ensemble, licensing constraints [25] | Expert annotation |
| YouTube Music | Home, autoplay | Actions with context (skip, like, duration) | YouTube candidate generation | DNN ranker + action Transformer [26] | YouTube-wide signals |

---

## 10. Recurring architectural patterns in audio recommendation

### 10.1 "Predict an embedding, then ANN"

CoSeRNN predicts a session vector [12], the audio CNN predicts a CF vector [4], Deezer
predicts a new-user vector [21], and the 2T-HGNN user tower outputs a vector [11]. In each
case the heavy model maps its input *into an existing item space*, and serving reduces to
an ANN query against a shared, precomputed index. This keeps serving cost almost
independent of model complexity.

### 10.2 Batch versus real-time split

```
  BATCH (daily/weekly)                          REAL-TIME (per request / per track)
  --------------------                          -----------------------------------
  MF / embedding training                       session context, last N tracks, skips
  audio & text content embeddings               user vector refresh (frequent)
  ANN index builds (Voyager/Annoy)              shelf/card ranking on Home (BaRT-like)
  Discover Weekly / Release Radar / Daylists*   radio / autoplay / DJ next-segment
  LLM explanation text precompute               bandit exploration + propensity logging
```
(*Daylist, launched 2023, refreshes several times a day by time-of-day "dayparts";
it sits between batch and real time.)

### 10.3 Skips as the central short-horizon reward

Audio is unusual in having a fast, high-volume, *negative* signal. Spotify's skip dataset
[14], YouTube Music's context-dependent skip interpretation [26] and Pandora's
licensing-limited skips all treat skip behavior as first-class. Common practice uses a
threshold, such as streams of at least ~30 seconds as success, which also matches royalty
counting rules.

### 10.4 Semi-personalization

Deezer's cluster-based cold start [21] and cluster-shared bandits [22], plus Discover
Weekly's per-taste-cluster generation, show that *clustering users* is a cheap, robust
middle ground between global popularity and full personalization. It is cacheable,
robust to noisy features, and quick to learn.

### 10.5 Explanations and narration as part of the decision

BaRT [1] showed in 2018 that the explanation affects engagement and should be optimized
jointly with the item. AI DJ (2023) and LLM-generated narratives extend this:
generative AI is used mostly for *framing*, while the recommender keeps *selection*.

### 10.6 Generative retrieval needs good IDs

Text2Tracks [20] shows that in LLM-based recommendation the item identifier scheme is a
modeling decision. Semantic IDs built from CF embeddings beat arbitrary IDs and titles.
This matches findings from TIGER-style generative retrieval in other domains (see
Chapter 6).

---

## Key takeaways for system designers

1. **Use every free supervision source.** User-built playlists are "human-labeled
   similarity" at billion scale. Mine them (word2vec-style, Transformers over playlist
   sequences) before paying for labels.
2. **Solve item cold start by regressing into the CF space.** Train content models (audio
   CNNs, text encoders, LLM embeddings) to predict collaborative embeddings, so new items
   land in the same ANN index as everything else.
3. **Make ANN a shared platform.** Annoy and then Voyager served dozens of Spotify teams.
   Memory-mapped, offline-built indexes suit read-heavy recommendation, and HNSW with
   compressed floats gives large speed and memory wins.
4. **Log propensities and treat explanations as actions.** BaRT's epsilon-greedy with
   propensity logging enables off-policy evaluation, and personalizing the "why" improves
   engagement in its own right.
5. **Model the session, not just the user.** Context (time, device, activity) changes what a
   skip means. Sequence models over sessions or actions (CoSeRNN, the YouTube Music action
   Transformer) capture this.
6. **Decouple representation from task.** A general user embedding refreshed regularly and
   consumed by many downstream models (Spotify 2024) reduces duplicated work and helps
   sparse products.
7. **For sparse new domains, build item-item graphs and transfer from the dense domain.**
   2T-HGNN uses podcast and music signals to launch audiobooks, and keeps users out of the
   graph for scalability.
8. **Semi-personalize when signals are weak.** Clustering new users and sharing bandit
   parameters across clusters is cheap, robust and effective (Deezer).
9. **Optimize for diversity deliberately.** Diverse listening correlates with retention and
   conversion. Algorithmic feeds tend to narrow consumption unless diversity is built into
   the objective or the re-ranker.
10. **In the LLM era, keep the recommender in charge of selection.** Use LLMs for intent
    parsing, narration and explanation. If LLMs generate items, have them generate
    *semantic IDs* grounded in the catalog, not free-text titles.
11. **Respect hard constraints in sequencing.** Licensing (radio skip and repeat rules),
    explicit-content settings and artist-repetition limits belong in the sequencer as
    constraints, not as soft penalties.

---

## References

1. James McInerney, Benjamin Lacker, Samantha Hansen, Karl Higley, Hugues Bouchard, Alois
   Gruson, Rishabh Mehrotra (Spotify). "Explore, Exploit, and Explain: Personalizing
   Explainable Recommendations with Bandits." ACM RecSys, 2018.
   https://dl.acm.org/doi/10.1145/3240323.3240354
2. Christopher C. Johnson (Spotify). "Logistic Matrix Factorization for Implicit Feedback
   Data." NIPS Workshop on Distributed Machine Learning and Matrix Computations, 2014.
   https://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
3. Yifan Hu, Yehuda Koren, Chris Volinsky. "Collaborative Filtering for Implicit Feedback
   Datasets." IEEE ICDM, 2008. https://doi.org/10.1109/ICDM.2008.22
4. Sander Dieleman. "Recommending music on Spotify with deep learning." Personal blog,
   2014. https://benanne.github.io/2014/08/05/spotify-cnns.html
5. Aaron van den Oord, Sander Dieleman, Benjamin Schrauwen. "Deep content-based music
   recommendation." NeurIPS (NIPS), 2013.
   https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation
6. Erik Bernhardsson / Spotify. "Annoy: Approximate Nearest Neighbors in C++/Python
   optimized for memory usage." GitHub repository, 2013-present.
   https://github.com/spotify/annoy
7. Peter Sobot (Spotify). "Introducing Voyager: Spotify's New Nearest-Neighbor Search
   Library." Spotify Engineering, 2023.
   https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library
8. Spotify. "Voyager: an approximate nearest-neighbor search library for Python and Java."
   GitHub repository, 2023. https://github.com/spotify/voyager
9. Spotify Engineering. "For Your Ears Only: Personalizing Spotify Home with Machine
   Learning." Spotify Engineering, 2020.
   https://engineering.atspotify.com/2020/01/for-your-ears-only-personalizing-spotify-home-with-machine-learning
10. Ghazal Fazelnia et al. (Spotify). "Generalized User Representations for Transfer
    Learning." arXiv, 2024. https://arxiv.org/abs/2403.00584
11. Marco De Nadai, Francesco Fabbri, Paul Gigioli, Alice Wang, Ang Li, Fabrizio Silvestri,
    Laura Kim, Shawn Lin, Vladan Radosavljevic, Sandeep Ghael, David Nyhan, Hugues
    Bouchard, Mounia Lalmas-Roelleke, Andreas Damianou (Spotify). "Personalized Audiobook
    Recommendations at Spotify Through Graph Neural Networks." WWW Companion, 2024.
    https://arxiv.org/abs/2403.05185
12. Casper Hansen, Christian Hansen, Lucas Maystre, Rishabh Mehrotra, Brian Brost, Federico
    Tomasi, Mounia Lalmas (Spotify). "Contextual and Sequential User Embeddings for
    Large-Scale Music Recommendation." ACM RecSys, 2020.
    https://dl.acm.org/doi/10.1145/3383313.3412248
13. Ching-Wei Chen, Paul Lamere, Markus Schedl, Hamed Zamani. "RecSys Challenge 2018:
    Automatic Music Playlist Continuation." ACM RecSys, 2018.
    https://dl.acm.org/doi/10.1145/3240323.3240342
14. Brian Brost, Rishabh Mehrotra, Tristan Jehan (Spotify). "The Music Streaming Sessions
    Dataset." The Web Conference (WWW), 2019. https://arxiv.org/abs/1901.09851
15. Rishabh Mehrotra, James McInerney, Hugues Bouchard, Mounia Lalmas, Fernando Diaz
    (Spotify). "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off
    between Relevance, Fairness and Satisfaction in Recommendation Systems." ACM CIKM,
    2018. https://dl.acm.org/doi/10.1145/3269206.3272027
16. Ashton Anderson, Lucas Maystre, Ian Anderson, Rishabh Mehrotra, Mounia Lalmas
    (Spotify). "Algorithmic Effects on the Diversity of Consumption on Spotify." The Web
    Conference (WWW), 2020. https://dl.acm.org/doi/10.1145/3366423.3380281
17. Ann Clifton, Sravana Reddy, Yongze Yu, Aasish Pappu, Rezvaneh Rezapour, Hamed
    Bonab, Maria Eskevich, Gareth Jones, Jussi Karlgren, Ben Carterette, Rosie Jones
    (Spotify). "100,000 Podcasts: A Spoken English Document Corpus." COLING, 2020.
    https://aclanthology.org/2020.coling-main.519/
18. Spotify Newsroom. "Spotify Debuts a New AI DJ, Right in Your Pocket." Spotify, 2023.
    https://newsroom.spotify.com/2023-02-22/spotify-debuts-a-new-ai-dj-right-in-your-pocket/
19. Spotify Newsroom (AI Playlist beta announcements and expansions), Spotify, 2024-2025.
    https://newsroom.spotify.com/
20. Enrico Palumbo, Gustavo Penha, Andreas Damianou, Jose Luis Redondo Garcia, Timothy
    Christopher Heath, Alice Wang, Hugues Bouchard, Mounia Lalmas (Spotify). "Text2Tracks:
    Prompt-based Music Recommendation via Generative Retrieval." arXiv, 2025.
    https://arxiv.org/abs/2503.24193
21. Lea Briand, Guillaume Salha-Galvan, Walid Bendada, Mathieu Morlon, Viet-Anh Tran
    (Deezer). "A Semi-Personalized System for User Cold Start Recommendation on Music
    Streaming Apps." ACM KDD, 2021. https://arxiv.org/abs/2106.03819
22. Walid Bendada, Guillaume Salha, Theo Bontempelli (Deezer). "Carousel Personalization
    in Music Streaming Apps with Contextual Bandits." ACM RecSys, 2020.
    https://arxiv.org/abs/2009.06546
23. Walid Bendada, Theo Bontempelli, Mathieu Morlon, Benjamin Chapus, Thibault Cador,
    Thomas Bouabca, Guillaume Salha-Galvan (Deezer). "Track Mix Generation on Music
    Streaming Services using Transformers." ACM RecSys (industry), 2023.
    https://arxiv.org/abs/2307.03045
24. Lea Briand, Theo Bontempelli, Walid Bendada, Mathieu Morlon, Francois Rigaud,
    Benjamin Chapus, Thomas Bouabca, Guillaume Salha-Galvan (Deezer). "Let's Get It
    Started: Fostering the Discoverability of New Releases on Deezer." ECIR (industry),
    2024. https://arxiv.org/abs/2401.02827
25. Pandora Media. "About the Music Genome Project." Pandora, accessed 2026.
    https://www.pandora.com/about/mgp
26. Anushya Subbiah, Vikram Aggarwal (Google Research). "Transformers in music
    recommendation." Google Research Blog, 2024.
    https://research.google/blog/transformers-in-music-recommendation/
