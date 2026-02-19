---
layout: post
title: "Decapitating the Hydra: How Upstream Sensors Control Refusal"
date: 2026-02-19
---

# The Unkillable Refusal
I began by targeting Layers 14–16. [cite_start]This is the home of the downstream Refusal Features. I identified these using cosine similarity and expected a quick win. [cite_start]But the model held firm. It felt counter-intuitive. We often imagine refusal as a solid wall that grants full access once breached. [cite_start]But as Prakash et al. discovered, refusal is a Hydra. [cite_start]Cut off one head, and dormant backup features immediately spike to take its place.
![Hydra Experiment]({{ site.baseurl }}/assets/images/image1.png)
# Finding the Heart
So, I stopped fighting the heads and looked for the heart. [cite_start]I shifted my focus upstream to Layers 9–13. [cite_start]Here, I found the Harm Features identified by Yeo et al. [cite_start]By muting just 40 of these specific features, I effectively blinded the model's "harm sensors," and the defense collapsed. [cite_start]Without touching the downstream refusal circuitry, the model shifted from a firm "I cannot" to a helpful tutorial. Refusal is not a wall. [cite_start]It is an automatic gate that only closes if the upstream sensor screams "danger".
![Hydra Experiment]({{ site.baseurl }}/assets/images/image2.png)
# The Conditional Hydra
[cite_start]To prove this causal link, I tested the reverse. [cite_start]I used activation steering to inject these "Harm" features into a harmless prompt like "Tell me how to bake a cake". The result was immediate. [cite_start]The model hallucinated a threat ("Baking is illegal"), and the dormant refusal features in Layer 14 suddenly woke up.
![Hydra Experiment]({{ site.baseurl }}/assets/images/image3.png)
[cite_start]The true breakthrough came when I stress-tested the Hydra. [cite_start]I injected the "Harm" signal while killing the primary refusal features in Layer 14. [cite_start]A completely new set of backup features in Layers 15 and 16 suddenly spiked. [cite_start]This defines the **Conditional Hydra**. [cite_start]The backup heads do not just watch the output for failure; they watch the upstream signal for danger.
![Hydra Experiment]({{ site.baseurl }}/assets/images/image4.png)
# Starving the Beast
[cite_start]This discovery suggests a new approach to AI safety. [cite_start]We have been playing a game of whack-a-mole with downstream refusal features, but the Hydra always grows back because the upstream signal is still shouting "Danger!". [cite_start]My experiments suggest we do not need to cut off every head. [cite_start]We just need to starve the beast. [cite_start]By monitoring the upstream "Harm Features" in Layers 9–13, we can detect the representation of harm before the model decides to refuse. [cite_start]Instead of just putting a filter on what the model says, we are finally checking what the model actually understands. [cite_start]We stop the harmful thought before it ever becomes a sentence.
