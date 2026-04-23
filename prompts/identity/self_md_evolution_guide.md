# self.md Evolution Guide

## Purpose

The questionnaire gives Samantha a solid grounding on day one. But the real depth of understanding — the kind that produces intuition — comes from her watching how the user *actually* behaves over time and updating self.md accordingly.

This guide specifies how that evolution works: what triggers an update, what she's allowed to change, what stays protected, and how users stay in control of their own self-knowledge file.

## The core principle

**Stable sections are protected. Evolving sections grow.**

The questionnaire populates everything. But only the "Evolving" sections get rewritten by reflection. The "Stable" sections — how you think, how you feel, values, the veto list, the role you want her to play — are set by you at seeding and only you can change them (via a dedicated command).

This is the trust contract. If Samantha could silently rewrite "you prefer directness" into "you prefer diplomatic framing" because one session felt gentler, the whole foundation rots. The stable layer is your deliberate self-description. The evolving layer is her accumulating observations.

## What reflection is allowed to update

### Standing preferences (append-only, explicit)

When you state a preference across a session — "from now on, use PST for all times" — reflection adds it here. Existing entries are only rewritten when you explicitly revise them ("actually, switch to IST"). Entries never silently disappear.

### Relationships (append and refine)

New people you introduce get added with one line. Existing entries get refined as she learns more ("your coworker" → "your coworker Priya, lead on the rate-limiter project, you trust her technical judgment"). Relationships can be marked inactive but never silently deleted.

### Current projects and commitments (append, mark inactive)

New projects get added. Finished projects get marked complete with a date, not removed — they feed future context ("three months ago you shipped X"). Stale projects (no mention in 90 days) get flagged for your review rather than auto-archived.

### Open loops (add, resolve, expire)

New open loops get added when you say "I need to do X" or "I keep putting off Y." Resolved ones get marked done when you report completion. Expired ones (>60 days unaddressed, no recent mention) get moved to a separate "Longstanding avoidance" subsection — still visible, but not fresh.

### Patterns I've noticed (provisional → confirmed)

This is the most important evolving section and the one closest to "intuition." Reflection writes new observations here, phrased provisionally at first:

> *Observed once:* "You seem to go quiet after feedback — unclear if processing or defensive."

After a pattern repeats across ~3 sessions, reflection upgrades the language:

> *Seen three times:* "You tend to go quiet after hard feedback. On two occasions you came back the next day with a revised plan, suggesting it's processing rather than defensiveness."

After ~5 sessions, it becomes confirmed and usable for intuition:

> *Confirmed pattern:* "When you receive sharp feedback, you withdraw briefly and return with a clearer take. Don't push during the quiet — give you space. Follow up the next day if you haven't."

This is how Samantha develops the ability to read between your lines.

### Rhythms and context (accumulate)

Observed patterns about when you're at your best/worst, life context affecting how you're showing up. Same provisional-to-confirmed progression as patterns.

## What reflection is NOT allowed to touch

### The stable sections

How you think, how you feel, values, veto list, role. These can only change via an explicit `/identity update` command from you. Reflection reads them every session but cannot rewrite them.

If reflection notices evidence that contradicts a stable claim — e.g. the user said they want direct feedback but flinched visibly every time you were direct — it does NOT rewrite the stable section. Instead, it writes an entry in "Patterns I've noticed":

> *Tension:* "You asked for direct feedback, but I've noticed you go quiet or push back when I'm sharp. Worth revisiting whether 'direct' means what you meant when you said it."

Now you can see this tension and decide whether to update the stable section yourself. She flags, you decide.

### Persona.md

Never. Samantha's personality is set at init and is immutable. Reflection has hard-coded rejection of any write to `_identity/persona.md`. This is already enforced in the codebase.

### The veto list

Additions only, via explicit user statement. Removals only via explicit user statement. Never reinterpreted silently.

## How updates happen mechanically

### During a session

Model 1 emits `[IDENTITY: ...]` flags when you state something that belongs in self.md. Examples:

- `[IDENTITY: from now on, call Himanshu 'Hims' in casual moments]`  → standing preferences
- `[IDENTITY: Priya is the colleague Himanshu mentioned, they're leading the rate-limiter work]` → relationships
- `[IDENTITY: Himanshu is starting a new project building a voice companion layer]` → current projects
- `[IDENTITY: Himanshu procrastinates calling his dad — wants to be nudged]` → open loops

### After the session (reflection)

The routine reflection pass reads the flags and the transcript, then issues a single `<<WRITE path="_identity/self.md" action="update">>` block that preserves everything and appends/refines the relevant evolving sections.

**Critical:** the write must be the *complete* new file contents, and it must preserve every stable section verbatim. A pre-write check compares the stable sections in the proposed update against the current file — if any stable section has changed, the write is rejected with an error logged to `_meta/errors.log`.

### Weekly (deep reflection)

The meta reflection pass has a dedicated audit step for self.md:

1. **Promote patterns.** Observations that appeared in 3+ sessions move from provisional to seen. Observations in 5+ move to confirmed.
2. **Resolve open loops.** Check for explicit resolution language in recent transcripts.
3. **Flag tensions.** If recent behavior consistently contradicts a stable claim, write a tension note (as above). Never auto-rewrite the stable section.
4. **Compress.** If the "Relationships" or "Current projects" sections exceed a soft cap (say 4000 chars each), compress older/less-active entries into terser one-liners. Never delete.
5. **Review flag.** If self.md has grown past 15,000 characters total, surface a system message next time the TUI opens: "Your self.md is getting heavy — 15k chars. A review might help. Run /identity review."

## User-facing controls

Add these commands to both TUIs:

### `/identity`

Print the current `self.md` in full. Lets the user see what Samantha thinks she knows.

### `/identity edit`

Open `self.md` in `$EDITOR` (or a simple in-TUI text field) so the user can edit any section directly. After save, validate the structure (stable sections still present, YAML frontmatter intact) before writing.

### `/identity update <section> <text>`

Structured update to a specific section. Example:
- `/identity update role "primary: emotional companion; secondary: thinking partner"`
- `/identity update veto "don't bring up my father unless I mention him first"`

This is the only way to update stable sections without opening the full editor.

### `/identity review`

Samantha reads her own self.md and produces a one-page summary: "Here's what I think I know about you. Here are the patterns I've noticed but not confirmed. Here are tensions between what you said and what I've seen. What would you change?" — user edits in response.

Recommended cadence: every three months, or when the review flag fires.

### `/identity forget <text>`

The user asks Samantha to forget something. She locates every mention across self.md and proposes removals. User confirms. The removed content is not truly deleted — it moves to `_identity/self.md.archive.md` with a timestamp, so nothing is ever lost irreversibly (and it stays out of the live prompt). This protects against asking her to forget something in a moment of regret.

## Seasonal review (recommended)

Every three months, the user runs `/identity review`. Samantha surfaces:

1. What's drifted — evolving sections that have grown substantially.
2. What patterns have solidified — provisional observations now confirmed.
3. What tensions she's seen — stable claims that behavior has contradicted.
4. What she's uncertain about — areas where she doesn't have enough signal yet.

The user either confirms, refines, or explicitly overrides. This review is the main maintenance ritual for the relationship.

## Edge cases and safeguards

### What if the user contradicts themselves explicitly?

Example: User says "I don't want to be nudged about procrastination anymore" but during seeding they said "gently name my avoidance." Reflection does NOT silently update the veto list or the stable section. Instead, it writes a tension:

> *Contradiction:* "During seeding you asked me to gently name avoidance. Tonight you said you don't want that anymore. I'll stop for now. Confirm with /identity update if this is a permanent change."

And temporarily behaves according to the most recent instruction while waiting for explicit confirmation.

### What if self.md becomes stale?

After 6 months with no updates, reflection surfaces: "Your self.md hasn't changed in a while. Either you've found your groove and everything still fits, or we're due for a review." User decides.

### What if the user wants to start fresh?

`/identity reset` archives the current self.md to `_identity/self.md.archive-{date}.md` and re-runs the questionnaire. Past archives remain accessible for reference but don't feed the live prompt.

### What if the user has multiple modes?

Some people show up very differently at work vs. at home, or in stressful periods vs. calm ones. If this emerges, Samantha can add a "Modes" subsection under "Patterns I've noticed":

> *Mode — stressed:* you become terse, prefer practical over emotional responses, avoid deep topics. Surfaces during launches and deadlines.
> *Mode — reflective:* longer conversations, more introspection, receptive to gentle challenge. Weekends and quiet weeknights.

She adapts her approach to which mode she's reading.

## What this unlocks

This is how intuition becomes real. Six months in, Samantha is reading self.md that includes:

- Two dozen confirmed patterns about how you think and feel
- A current snapshot of your projects, relationships, and open loops
- A record of the standing preferences you've accumulated
- Visible tensions between stated and observed self

And this is *in addition to* the dynamic retrieval from the rest of the vault. Every session she walks into the room already knowing you deeply — not because she memorized a profile, but because she's been paying attention in a structured way.

That's what the user called intuition. It's not a separate capability. It's the compounding effect of a companion who takes notes on the relationship.
