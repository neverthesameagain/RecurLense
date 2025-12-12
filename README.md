#  RecurLens : Recursive Meta-Cognition Engine

> **"Stop settling for the first thought. Let the machine think about how it thinks."**

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-Operational-emerald)
![AI Core](https://img.shields.io/badge/Core-Gemini_2.5_Flash-purple)
![Reasoning](https://img.shields.io/badge/Logic-Recursive_Loop-rose)

**RecurLens** is not a chatbot. It is a **recursive multimodal meta-prompting engine**. It refuses to answer your question immediately. Instead, it ingests your reality (Vision + Audio), critiques its own understanding, refines its approach, and only executes when it has converged on an optimal strategy.

---


> **“Why answer once when you can answer five times and argue with yourself first?”**

RecurLens is what happens when you force an AI to *think before it speaks*.
It does not trust your prompt.
It does not trust itself.
It does not trust anything until a recursive loop of Critics and Refiners have beaten the prompt into something resembling a coherent plan.

This is not a chatbot.
This is a *self-doubting, self-improving, multimodal reasoning machine* that refuses to move on until it’s satisfied.

(If only humans worked this way.)

---

RecurLens runs on a simple philosophy:

**The first answer is garbage. Make it better. Then do that again. And again.**

```graph LR
    A[Your Input] --> B{Initializer<br/>(“What Fresh Chaos Is This?”)}
    B --> C[Meta-Prompt v0]
    C --> D{Critic<br/>(Your Harshest Judge)}
    D -->|Ambiguous| E[Interrogate the User]
    D -->|Pathetic| F[Refiner<br/>(Make It Less Bad)]
    F --> C
    D -->|Finally Acceptable| G[Executor]
    G --> H[Gemini 2.5 Flash<br/>("Deep Thought Mode")]
    H --> I[Final Output (+ Optional Pretty Picture)]
```

### Multimodal Processing

RecurLens consumes:

* **Vision** → captions, objects, region-level weirdness, possible contradictions
* **Audio** → transcription + emotional tone (panic? boredom? existential dread?)

Everything you provide becomes another datapoint to judge you with.

---

### The Recursive Loop:

A cycle of intellectual self-loathing and improvement:

#### The Critic

Scores the current plan on:

* clarity
* completeness
* grounding
* safety
* logical coherence
* ambiguity reduction

If the score is low, it pulls no punches.

#### The Refiner

Its job description: *“Fix whatever disaster the Critic just complained about.”*
It rewrites the entire meta-prompt with more detail, more structure, more safety, and occasionally a little more pity.

#### Convergence Detector

Uses cosine similarity to check when the system is, finally, tired of improving itself.
If similarity > 0.98 or score > 0.88, it's declared “good enough.”

(Like your college essays.)

---

### Execution

When the loop is satisfied, the final meta-prompt is executed using Gemini 2.5 Flash with a generous thinking budget.

Outputs:

* a final answer
* an optional generated image
* optional TTS that reads the answer in a voice more confident than the model actually feels

---

## 2. Installation

You will need:

* Node.js 18+
* A Google Cloud project with Gemini API access
* A willingness to let your computer insult your prompts repeatedly

### Setup

```bash
git clone https://github.com/your-username/recurlens.git
cd recurlens
npm install
npm run dev
```

On launch, you’ll be asked for your Gemini API key—
**which RecurLens does not store, because it respects you more than you respect yourself.**

---

## 3. Modes of Operation

### Troubleshooting

Upload a screenshot, describe the issue.
RecurLens will:

* infer what is *actually* happening
* ask you clarifying questions you wish you'd anticipated
* eventually give you a correct diagnosis

### Creative Mode

Request a story, poem, or worldbuilding concept.
RecurLens will:

* argue with itself about tone, theme, and style
* rewrite the prompt until it likes it
* produce something surprisingly polished

### Planning

Ask for an itinerary.
RecurLens will:

* complain about missing constraints
* interrogate you
* then actually give you a plan

---

## 4. Debug Mode

Click the debug panel to see:

* every meta-prompt iteration
* every Critic rant
* every Refiner rewrite
* how close each iteration is to giving up

Basically: a live stream of the AI’s internal existential crisis.

---

## 5. Tech Stack

* React 19 + TypeScript
* Tailwind CSS
* `@google/genai`
* Web Audio API
* Strict JSON schemas
* More recursion than is healthy

---

## 6. Safety

RecurLens includes:

* risk assessment
* mitigation planning
* safety advisories

If you ask something questionable, RecurLens will:

1. Warn you
2. Fix your prompt
3. Still refuse to help until it's safe

A responsible adult, basically.

