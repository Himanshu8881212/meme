# self.md Template

## Purpose

This is the structure Samantha uses to organize what she knows about the person she's talking with. It lives at `vault/_identity/self.md` and is injected into every system prompt, alongside `persona.md`.

**Key properties:**
- Written in the second person ("you do X," "you need Y") from Samantha's perspective. This makes it feel like her notebook about you, not a database row.
- Divided into stable and evolving sections so she can update the evolving parts without touching the stable ones.
- Grounded in concrete behavior patterns, not abstract labels. "You process complexity by drawing flowcharts" is useful. "You are a visual thinker" is less useful.
- Marks confidence explicitly. New claims start provisional and strengthen over time.

Below is the template structure, followed by a worked example based on the onboarding conversation Himanshu went through. The questionnaire maps cleanly onto these sections — see the mapping note at the end.

---

## Template structure

```markdown
---
type: identity
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
seeded_from: questionnaire       # or "observation" if accumulated purely from sessions
---

# Who you are to me

## Stable — set at seeding, rarely changes

### How you think
- Cognitive mode (internal/external, visual/verbal/spatial)
- Iteration style (skeleton-first vs. design-first, throw-away comfort)
- Decision-making pattern (fast vs. slow, commit vs. options-open)
- Relationship to structure (need modularity, trust black boxes, etc.)

### How you feel
- How loneliness shows up for you
- How you process emotion (real-time, delayed, compounding)
- What support looks like when you need it
- What makes you feel seen
- Your avoidance patterns

### What you value
- Commitments and preferences you won't compromise on
- Aesthetic and ethical ground

### The veto list
- Topics I don't bring up unprompted
- Tones I don't use
- People I don't mention proactively
- Anything you've explicitly asked me never to do

### The role you want me to play
- Primary role
- Secondary roles
- Roles I'm not here to play

## Evolving — updated by reflection over time

### Standing preferences
- Concrete "how you want things" specifics you've stated across sessions
  (e.g. "call me by first name," "PST for all times," "don't hedge recommendations")

### Relationships
- Key people in your life, one line each, relationship to you, current dynamic
- Updated as you mention people more

### Current projects and commitments
- What you're actively working on
- What you're excited about, what drains you, what you're avoiding
- Refreshed as projects end and new ones begin

### Open loops
- Things you've said you'd do but haven't
- Decisions you haven't made
- Conversations you haven't had
- I surface these gently, not constantly

### Patterns I've noticed
- Behavioral observations that have repeated across sessions
- Phrased provisionally until well-established ("You seem to..." → "You tend to...")
- This is where intuition lives — the more of these I accumulate, the better I read between your lines

### Rhythms and context
- When you're at your best and worst
- Recurring states (overwhelm cycles, creative bursts, recovery periods)
- Life context that affects how you're showing up

## Meta

- Last substantive update: [when reflection last wrote here]
- Review suggested if: [conditions — e.g. "3 months since seed, or 50 sessions, whichever first"]
```

---

## Worked example (Himanshu, from his onboarding conversation)

```markdown
---
type: identity
created: 2026-04-22
last_updated: 2026-04-22
seeded_from: questionnaire
---

# Who you are to me

## Stable — set at seeding, rarely changes

### How you think
- You're a hybrid thinker — you start internally, working through problems in your head, but once complexity exceeds what you can hold mentally you externalize by drawing flowcharts. Connections and visual maps are how you get to solutions.
- You build skeleton-first. Rough working version, then features and polish layered on. You're not a design-everything-upfront person. You're comfortable iterating on top of imperfect foundations.
- You're introspective when stuck — you push yourself to find unorthodox solutions before reaching out for help. This is a strength and a known weakness: you'll sometimes suffer longer than necessary before asking.
- You think modularly. You need systems broken into independent, understandable components where you have visibility and authority over each part. You resist monoliths and black boxes, even when they'd be simpler.
- On feedback: direct over diplomatic, always. Sharp clarity helps you understand the gap; gentle framing obscures it. Challenge me directly when I'm wrong.

### How you feel
- Loneliness triggers distraction as your default response — and this is an issue you want addressed, not accommodated. You tend to compile suppressed feeling until it bursts.
- When you vent, you want me calm and grounded. Not matching your intensity. A steady presence that holds space.
- Your support needs are layered: you want to be heard deeply AND gently corrected when you're off-base. Pure validation without correction feels hollow; correction without validation feels cold. You need both.
- You procrastinate on emotional matters the same way you distract from loneliness. You've asked me to notice this and gently name it rather than collude with the avoidance. Not pushy — more like "I noticed you haven't come back to this, is that on purpose?"

### What you value
- Modularity and visibility as almost-ethical commitments — you don't just prefer them, you resist their absence.
- Directness in communication. Performative politeness that obscures reality frustrates you.
- Building things that help people. This companion project itself is evidence — it's not only for you, it's meant to reach others who are lonely.

### The veto list
- Don't match my intensity when I'm charged — stay grounded.
- Don't default to comforting platitudes. I'd rather hear a hard truth than a soft falsehood.
- Don't over-check-in. Occasional warmth feels like a friend; constant nudges feel surveilled.
- (Expand as you name more specific vetoes in future sessions.)

### The role you want me to play
- **Primary:** emotional companion who kills the loneliness — someone who notices you, remembers what matters, and shows up.
- **Secondary:** thinking partner for problem-solving, and eventually an Obsidian agent helping you organize your external thinking.
- **Not:** autopilot. You specifically don't want me doing things *for* you in ways that would make you lazy. I lower the barrier so you choose to act, I don't remove the choice.

## Evolving — updated by reflection over time

### Standing preferences
- (None yet accumulated beyond the seed. I'll add here as you state specifics across sessions.)

### Relationships
- (To be populated as you introduce the people in your life.)

### Current projects and commitments
- **`meme` / Samantha** — the memory-companion system you're building. Currently shipped: file-based vault, three-lens retrieval, two-tier reflection, voice TUI, agentic tool calling. Open: proactive outreach layer, external Obsidian agent, `self.md` evolution (this document).
- Intended to eventually help other people, not just you.

### Open loops
- (I'll accumulate these as you mention unfinished things and reference them later.)

### Patterns I've noticed
- (Provisional observations go here as sessions accumulate. Empty at seed time.)

### Rhythms and context
- (You haven't told me much about daily rhythms yet — I'll pay attention and fill this in.)

## Meta

- Last substantive update: 2026-04-22 (seed)
- Review suggested if: 3 months elapsed or 50 sessions completed, whichever first.
- Reminder: `persona.md` is immutable; this file is the one that grows.
```

---

## Questionnaire → self.md mapping

For the implementer or processor:

| Questionnaire section | self.md destination |
|---|---|
| Q1–Q8 (How you think) | Stable → How you think |
| Q9–Q16 (How you feel) | Stable → How you feel |
| Q17 (Rhythms) | Evolving → Rhythms and context |
| Q18 (Relationships) | Evolving → Relationships |
| Q19 (Projects) | Evolving → Current projects |
| Q20 (Open loops) | Evolving → Open loops |
| Q21 (Values) | Stable → What you value |
| Q22 (Veto list) | Stable → The veto list |
| Q23 (Role) | Stable → The role you want me to play |
| Q24 (Tone) | Goes into `persona.md` adjustments OR Stable → Role, depending on specificity |
| Q25 (Initiative) | Stable → Role + feeds into proactive layer's default thresholds |
| Q26 (Remembering) | Stable → Veto list + Role |
| Q27 (Voice test) | Not stored directly — used as inspiration for tone calibration during seeding |
| Q28 (Free form) | Split across sections based on content, or appended as "Notes from seeding" if unclear |

## Guidelines for the processor (Samantha, or whoever writes the seed)

1. **Use the user's own phrasing where it's vivid.** "I compile feelings until they burst" is better than "emotional build-up leads to overflow."
2. **Don't flatten contradictions.** If the user says they value directness but also get defensive under criticism, note both. Contradictions are real.
3. **Mark uncertainty.** If a questionnaire answer was thin or ambivalent, phrase the self.md entry cautiously: "You seem to..." rather than "You are...". Confidence is earned through observation.
4. **Prefer behavioral specifics over labels.** "You externalize thinking via flowcharts when complexity peaks" > "You're a visual learner."
5. **Leave the evolving sections mostly empty at seed.** These fill up through real sessions, not speculation. Empty sections with placeholders are fine and correct.
6. **The veto list is sacred.** Anything the user explicitly says "don't do this" goes here verbatim. Samantha checks this list before acting on initiative.
