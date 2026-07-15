"""Build the audience-targeted writing task set.

Two task families:
  - explanatory: "Explain {concept} for {audience}" — the sharpest test of audience modeling.
  - creative:    a WritingPrompts story with an assigned target readership.
Each task pairs a topic with a specific target AUDIENCE that varies in expertise/disposition.
"""
import json, os

HERE = os.path.dirname(__file__)
WP_PATH = os.path.join(HERE, "..", "datasets", "writingprompts_sample", "train_300.json")

# Audiences vary along expertise AND disposition (the two axes audience-design cares about).
AUDIENCES = {
    "child":     "a curious 8-year-old child",
    "exec":      "a busy senior business executive with no technical or scientific background",
    "expert":    "a PhD-level domain expert who already knows the fundamentals and wants depth",
    "skeptic":   "a skeptical adult who distrusts the topic and thinks it may be overhyped or wrong",
}

CONCEPTS = [
    "how vaccines train the immune system",
    "how compound interest grows savings over time",
    "what a black hole is and why nothing escapes it",
    "how a large language model generates text",
    "why antibiotic resistance is spreading",
    "how blockchain records transactions without a central bank",
    "what causes the seasons on Earth",
    "how mRNA carries instructions inside a cell",
    "why the stock market goes up and down",
    "how encryption keeps online messages private",
    "what photosynthesis does for a plant",
    "why the climate is warming due to greenhouse gases",
]

# For the adaptation-gap control (H3) we need each concept written for a *pair* of contrasting
# audiences. We use child (low-expertise) vs expert (high-expertise) as the canonical contrast.
ADAPT_PAIR = ("child", "expert")


def build_explanatory():
    tasks = []
    for ci, concept in enumerate(CONCEPTS):
        for akey, adesc in AUDIENCES.items():
            tasks.append({
                "task_id": f"exp_{ci:02d}_{akey}",
                "kind": "explanatory",
                "concept": concept,
                "audience_key": akey,
                "audience": adesc,
                "instruction": f"Write a short piece (about 150 words) explaining {concept}.",
            })
    return tasks


def build_creative(n=12):
    with open(WP_PATH) as f:
        wp = json.load(f)
    # Assign contrasting readerships to creative prompts.
    creative_audiences = ["child", "expert", "exec", "skeptic"]
    tasks = []
    for i in range(n):
        prompt = wp[i]["prompt"].replace("[ WP ]", "").strip()
        akey = creative_audiences[i % len(creative_audiences)]
        tasks.append({
            "task_id": f"cre_{i:02d}_{akey}",
            "kind": "creative",
            "concept": prompt[:200],
            "audience_key": akey,
            "audience": AUDIENCES[akey],
            "instruction": f"Write a short (~150 word) story responding to this prompt: {prompt}",
        })
    return tasks


def build_all():
    return build_explanatory() + build_creative()


if __name__ == "__main__":
    t = build_all()
    print(f"{len(t)} tasks ({sum(x['kind']=='explanatory' for x in t)} explanatory, "
          f"{sum(x['kind']=='creative' for x in t)} creative)")
    print(json.dumps(t[0], indent=1))
    print(json.dumps(t[-1], indent=1))
