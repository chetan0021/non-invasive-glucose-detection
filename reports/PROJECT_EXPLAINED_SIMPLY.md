# Non-Invasive Blood Glucose Prediction — Simple Guide

**For:** Students, non-technical readers, anyone new to the project  
**Goal:** Understand what this project does, how it works, and why — without any coding knowledge needed

---

## What Problem Are We Solving?

Right now, if a diabetic person wants to check their blood sugar (glucose), they have to:

1. Prick their finger with a needle
2. Put a drop of blood on a test strip
3. Wait for a reading

This is painful, inconvenient, and needs to be done multiple times a day.

**This project asks: Can we measure blood sugar WITHOUT drawing blood?**

Instead of a needle, we use sensors that sit on your finger or mouth — like a smartwatch sensor — and measure things like your pulse, saliva, and skin temperature. A computer then uses those measurements to *estimate* your blood sugar level.

---

## How Does the Body Give Clues About Blood Sugar?

The human body changes in measurable ways when blood sugar goes up or down. Here are the main clues we use:

```
HIGH blood sugar                    LOW blood sugar
────────────────                    ───────────────
Saliva becomes more acidic          Heart may beat faster
Heart rate variability decreases    Sweating (cold skin)
Pulse waveform changes shape        HRV increases (or decreases)
Skin temperature rises slightly     Saliva becomes less acidic
```

We don't use just one clue — we use all of them together, because no single measurement is reliable enough on its own.

---

## The Sensors We Use

Think of this like a multi-sensor wristband/fingertip device:

```
                    YOUR FINGERTIP
                    ─────────────
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    💡 Light sensor  🌡️ Temp sensor  💧 Saliva sensor
    (PPG sensor)                    (pH strip)
          │              │              │
    Measures your    Measures skin   Measures how
    pulse and blood  temperature     acidic your
    flow through     (36–37°C)       saliva is
    the finger
```

**Light sensor (PPG):**  
Shines infrared light through your fingertip. Blood absorbs different amounts of light depending on how much is flowing through. This tells us your heart rate, pulse strength, and the shape of each heartbeat.

**Temperature sensor:**  
Skin temperature slightly rises with higher blood sugar because more blood is flowing near the surface.

**Saliva pH sensor:**  
Your saliva becomes slightly more acidic when blood sugar is higher. pH 7 is neutral — higher is alkaline, lower is acidic. Normal saliva is around pH 7.3. With high blood sugar it can drop to 6.2.

**ECG / HRV (Heart Rate Variability):**  
Not just how fast your heart beats, but how *consistent* the timing is between beats. When stressed or unwell, beats become more irregular. Blood sugar affects the nervous system, which shows up in this pattern.

---

## What We Ask the Patient / Device

When using this system, you provide information about the patient:

| What you tell the system | Example |
|---|---|
| Heart rate | 84 beats per minute |
| Saliva pH | 6.60 (slightly acidic) |
| Skin temperature | 36.9°C |
| Pulse strength (from light sensor) | 1850 (in sensor units) |
| Age | 59 years |
| Body weight (BMI) | 29.8 |
| Diabetes type | Type 2 |
| Taking insulin? | Yes |
| Fasting? | No (just ate) |

That's 9–15 pieces of information. The computer takes all of this and gives back one number: **your estimated blood sugar in mg/dL**.

---

## What Is mg/dL and What's Normal?

**mg/dL = milligrams per deciliter** — how much sugar is in your blood.

```
Blood Sugar Level Guide
────────────────────────────────────────────────────
  < 70 mg/dL   ⚠️  HYPOGLYCEMIA    Too low — dangerous
 70–99 mg/dL   ✅  NORMAL (fasting)   Healthy range
100–125 mg/dL  ⚠️  PREDIABETES     Borderline
126–180 mg/dL  ⚠️  HIGH            Diabetic range
 > 180 mg/dL   🚨  HYPERGLYCEMIA   Too high — needs attention
 > 250 mg/dL   🚨  SEVERE          Medical emergency risk
────────────────────────────────────────────────────
```

---

## The Big Picture — How the System Works

Here is the full journey from sensors to prediction:

```
STEP 1: Collect sensor readings
        ↓
   Heart rate, saliva pH, temperature, pulse shape...

STEP 2: Feed to the computer program (predict.py)
        ↓
   Program fills in missing details it can calculate
   (like estimating pulse energy from pulse strength)

STEP 3: Standardise all the numbers
        ↓
   Every measurement is converted to a common scale
   so big numbers (like 175000 ADC counts for light)
   don't overpower small numbers (like pH 7.2)

STEP 4: Run through 4 different prediction engines
        ↓
   Ridge  →  predictions
   Random Forest  →  predictions          These are 4 different
   XGBoost  →  predictions                mathematical "guessers"
   SVR  →  predictions

STEP 5: Combine all 4 predictions smartly (Stacking)
        ↓
   A 5th "judge" looks at all 4 guesses and
   combines them into one final answer
   (Random Forest gets most weight: 45%)

STEP 6: Calculate confidence range
        ↓
   "We think blood sugar is 173 mg/dL,
    but it could be anywhere from 142 to 215"

STEP 7: Safety check (Clarke Zone)
        ↓
   Is this prediction safe to act on?
   Zone A = Accurate ✅
   Zone B = Small safe error ✅
   Zone C = Could lead to wrong treatment ⚠️
   Zone D = Failed to detect danger 🚨
   Zone E = Opposite of truth — very dangerous 🚨

STEP 8: Return answer to user
        ↓
   "Predicted: 173.7 mg/dL | Safe range: 142–215 | Zone A"
```

---

## What Is a "Model" and Why Do We Have 4 of Them?

A **model** in machine learning is like a recipe that has been learned from examples. Instead of someone writing down the rules ("if pH is 6.5 AND heart rate is 84 THEN glucose is probably 170"), the computer looks at thousands of examples and *figures out the pattern itself*.

We use **4 different models** because each one has different strengths:

| Model | How to think about it | Best at |
|---|---|---|
| **Ridge** | A straight-line guesser | Simple, reliable patterns |
| **Random Forest** | 150 different "decision trees" voting together | Complex patterns, handles messy data well |
| **XGBoost** | Trees that learn from each other's mistakes | Highly accurate, catches subtle patterns |
| **SVR** | Finds the best boundary between high and low | Good when data is noisy |

Then a **5th model (the meta-learner)** learns how much to trust each of the 4 models:
- Random Forest: trust 45%
- XGBoost: trust 33%
- Ridge: trust 21%
- SVR: trust 12%

This combination approach is called **stacking** and it's more accurate than any single model alone.

---

## How Was the System Trained?

**Training** means showing the computer many examples so it can learn the patterns.

The problem: we don't have thousands of real patients with blood sugar + sensor readings all measured at the same time. That kind of data is expensive and rare to collect.

**Our solution: We created realistic fake patient data.**

Think of it like a flight simulator — it's not a real plane, but it teaches you real flying skills because it accurately mimics how planes behave.

We created **595 fake patient readings** based on real statistics from published medical research:

```
Fake patient types we generated:
──────────────────────────────────────────────────
  Healthy adults          → blood sugar 70–138 mg/dL
  Prediabetic adults      → blood sugar 100–185 mg/dL
  Type 2 diabetes (mild)  → blood sugar 94–210 mg/dL
  Type 2 diabetes (severe)→ blood sugar 126–335 mg/dL
  Type 1 diabetes         → blood sugar 40–380 mg/dL
                            (includes very low AND very high cases)
──────────────────────────────────────────────────
```

For each fake patient, the computer generates realistic values for *all* the sensors — pH, pulse, temperature — that correctly match their blood sugar level and health state.

> ⚠️ **Important honest note**: Because the training data is simulated, the accuracy numbers we report are on *simulated* patients. We don't yet know how well this works on real people. That would require real clinical trials.

---

## What Are All the Files? (In Plain English)

Don't worry about the code. Here's what each part of the project *does* in everyday language:

### The "Factory" Files (They Build Things)

| File | What it does in plain English |
|---|---|
| `synth_generator.py` | **The patient creator.** Makes up 595 realistic fake patients with all their sensor readings. Like writing a detailed medical fiction story for each patient. |
| `ingest_real_data.py` | **The data mixer.** Takes the fake patients + real hospital records from the USA and mixes them into one big dataset. Sorts them into "training group" and "testing group" — making sure the same person never appears in both. |
| `features.py` | **The translator.** Converts all the raw numbers into a standard format the computer can understand. Like converting feet to metres and Fahrenheit to Celsius before doing any comparison. |
| `train_models.py` | **The teacher.** Shows all the examples to the 4 models so they can learn. Tests how well they learned. Picks the best combination. |

### The "Worker" Files (They Do the Job)

| File | What it does in plain English |
|---|---|
| `predict.py` | **The calculator.** Takes a real patient's readings and returns a blood sugar estimate. This is the file that actually does the prediction when someone uses the system. |
| `app/dashboard.py` | **The screen.** A website you can open in your browser. Has sliders to enter patient values, and shows the prediction with charts and colours. |

### The "Inspector" Files (They Check Everything Works)

| File | What it does in plain English |
|---|---|
| `test_benchmark_presets.py` | Tests 5 "known" patients (like a healthy 34-year-old vs a diabetic 59-year-old) and checks the predictions are sensible. |
| `test_hypo_diversity.py` | Tests 5 different types of dangerously low blood sugar and checks we correctly detect them. |
| `audit_clarke_grid_function.py` | Double-checks that the safety rating system (Zone A/B/C/D/E) is implemented correctly against the original medical paper. |
| `verify_reverted_model_safety.py` | After any change to the system, this checks the old safety benchmarks still pass. |

### The "Report" Files (They Explain the Results)

| File | What it contains |
|---|---|
| `reports/model_comparison.md` | Full scientific report: how accurate is the model, where does it fail, what are its honest limitations. |
| `reports/PROJECT_DOCUMENTATION.md` | Technical documentation for developers (very detailed, lots of code concepts). |
| `reports/PROJECT_EXPLAINED_SIMPLY.md` | **This file** — the plain-language version for everyone. |
| `reports/ablation_study.md` | What happens when we remove one sensor at a time? Which sensor contributes the most? |

---

## How Is the Safety of a Prediction Measured?

We use the **Clarke Error Grid** — a standard medical system invented by Dr. William Clarke.

Imagine a graph where:
- The **horizontal axis** = the real (true) blood sugar value
- The **vertical axis** = what our system predicted

Every prediction is a dot on this graph. Where the dot falls determines the "zone":

```
              ↑ PREDICTED
    500 ┤     /
        │  E / A  B                    Zone A ✅ Within 20% of truth
    400 ┤   /──────────                Zone B ✅ Safe small error
        │  / A    B   C                Zone C ⚠️  Could cause wrong treatment
    300 ┤ /────────────                Zone D 🚨  Missed a danger (e.g.
        │/   A                                    predicted normal when
    200 ┤                                          actually low)
        │         B                   Zone E 🚨  Opposite of truth
    100 ┤    A
        │
     70 ┤ A/D
        └──────────────────────────→ TRUE BGL
          70  100  200  300  400  500
```

**Our current results:**
- Zone A (perfectly safe): 81.65% of predictions
- Zone A+B (safe): 93.58% of predictions
- Zone D (missed danger): 6.42% — mostly in hypoglycemia cases

---

## What the System Gets Right

✅ **Normal blood sugar range (70–250 mg/dL):** Works well. Zone A+B = 100%  
✅ **Prediabetes detection:** Very good. MAE (average error) = 11 mg/dL  
✅ **Type 2 diabetes:** Good. MAE = 20 mg/dL, 100% safe zone  
✅ **Healthy patients:** Works well. MAE = 11 mg/dL  
✅ **Severe high blood sugar (>250):** Good coverage, Zone A+B = 100%  

---

## What the System Struggles With

### 🚨 Low Blood Sugar (Hypoglycemia) Detection

This is the most serious limitation and we want to be completely honest about it.

When blood sugar drops dangerously low, the body reacts in different ways depending on the person and situation:

```
Type 1: Sympathetic response    → Heart races, sweats, feels cold
Type 2: Parasympathetic         → Heart slows, feels calm (misleading)
Type 3: Exercise-induced        → High perfusion, elevated temperature
Type 4: Nocturnal (sleeping)    → Very slow heart rate, very regular
Type 5: Autonomic unawareness   → Body doesn't react at all (most dangerous!)
```

Our system correctly detects **Type 1 (racing heart)** but FAILS to detect **Types 3, 4, and 5** — it predicts a normal blood sugar reading even though the patient is in danger.

**Why?** Because the system has learned to rely heavily on saliva pH (very reliable) and medication type (very reliable). For all these Type 1 diabetic patients on insulin, the saliva pH and medication are similar regardless of whether they're hypoglycemic or not — so the system "guesses" a normal Type 1 range instead of detecting the danger.

**What would fix it:** Real patient data from people experiencing these episodes. Our system was trained on simulated data, and simulated data can't perfectly capture the subtle body signals in these rare, dangerous situations.

> ⚠️ **This system should NOT be used as the only safety net for hypoglycemia detection until this is fixed with real patient data.**

---

## How Does the Confidence Interval Work?

The system doesn't just say "173 mg/dL". It says:

> "We estimate 173 mg/dL, but the true value could reasonably be anywhere between 142 and 215 mg/dL."

The range (142–215) is the **confidence interval**. It means "we're fairly confident the real value falls in this range."

Think of it like a weather forecast:
- "Temperature will be 25°C" = point estimate
- "Temperature will be between 22°C and 28°C" = confidence interval

Our confidence intervals are currently a bit too wide (113 mg/dL average width). We're being cautious rather than overconfident. The system is like a new doctor giving a range rather than a specific number until they have more experience.

---

## The Out-of-Distribution Warning

Sometimes a patient's readings are very unusual — outside anything the system has seen before. For example:

- Body temperature of 35.8°C (too cold — this is hypothermic)
- Heart rate of 140 BPM (very fast — possible arrhythmia)

In these cases the system **still gives a prediction** but adds a warning:

> ⚠️ "Warning: Skin Temperature 35.8°C falls outside the range this system was trained on (36.2–37.2°C). This prediction may be unreliable."

This is important safety honesty — rather than silently giving a wrong answer, the system flags when it's being asked to work outside its knowledge.

---

## The Two Models in This System

This project actually contains **two separate prediction systems**:

### Model A — The Sensor Model
- **Needs:** PPG sensor + saliva pH + temperature + HRV + demographics
- **Gives:** Exact estimated blood sugar (mg/dL)
- **Use case:** A wearable device or clinical setting with sensor hardware

### Model B — The Screening Model
- **Needs:** Only age, weight, gender, family history, lifestyle, health conditions
- **Gives:** Risk category: Low Risk / Elevated Risk / Diabetic Risk
- **Use case:** Community screening or phone app — no sensors needed
- **Accuracy:** 80.9% overall (works better for diabetic risk detection: 88.5%)

> **Important:** Model B cannot tell you your blood sugar number. It just says "you're in a high-risk group, please get a proper test." Think of it as a first filter to identify who needs clinical follow-up.

---

## Summary in One Paragraph

This project builds a computer system that estimates blood sugar from non-invasive sensors — light, temperature, saliva — without drawing blood. It combines readings from multiple sensors, standardises them, passes them through four different prediction algorithms, and combines their answers for a final estimate. The system also rates how safe each prediction is (Zone A to E), gives a confidence range, and warns when readings seem unusual. It was trained on realistic simulated patient data, works well for normal and high blood sugar ranges, but struggles to detect some types of dangerously low blood sugar — a known limitation documented honestly in the system. The next step to improve it is collecting real patient data from people experiencing these dangerous low-sugar episodes.

---

## Glossary — Words You Might Not Know

| Word | What it means in simple terms |
|---|---|
| **Blood glucose / blood sugar** | The amount of sugar in your blood. Measured in mg/dL. |
| **mg/dL** | Milligrams per deciliter — the unit for measuring blood sugar. Like measuring mg of sugar in a small cup of blood. |
| **PPG** | Photoplethysmography. A light sensor that measures blood flow through skin. The same technology in smartwatches that measures heart rate. |
| **HRV** | Heart Rate Variability. Not just how fast your heart beats, but how consistent the timing is. Stress and illness make the timing less consistent. |
| **pH** | A scale measuring how acidic or alkaline something is. 7 = neutral, below 7 = acidic, above 7 = alkaline. |
| **Machine Learning model** | A computer program that has learned patterns from examples, instead of being programmed with explicit rules. |
| **Training** | Showing the computer thousands of examples so it can learn patterns. Like studying for an exam. |
| **Feature** | One piece of input information. Heart rate is a feature. pH is a feature. Age is a feature. |
| **Prediction** | The model's estimated output — in this case, an estimated blood sugar number. |
| **Clarke Error Grid** | A medical safety scoring system for blood sugar predictions. Zones A–E indicate how dangerous a wrong prediction would be. |
| **Confidence interval** | A range that likely contains the true answer. "173 ± 40" means the true value is probably somewhere between 133 and 213. |
| **OOD (Out of Distribution)** | When an input value is outside the range the system was trained on. Like asking a UK price calculator about prices in a country it's never seen data from. |
| **Synthetic data** | Realistic fake data generated by a computer, based on real statistical patterns. Not real patient data. |
| **Stacking / Ensemble** | Using multiple models and combining their answers instead of relying on just one. Like asking 4 doctors for their opinion and then a 5th doctor weighing all 4 opinions. |
| **Hypoglycemia** | Dangerously low blood sugar (below 70 mg/dL). Can cause unconsciousness if untreated. |
| **Hyperglycemia** | High blood sugar (above 180 mg/dL). The defining characteristic of diabetes. |
| **R²** | A measure of how well the model's predictions match reality. R²=1.0 means perfect, R²=0.83 means 83% of variation explained. |
| **MAE** | Mean Absolute Error. The average size of the prediction mistakes in mg/dL. Lower is better. |
