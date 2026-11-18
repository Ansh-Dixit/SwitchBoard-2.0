# SwitchBoard Word-Level Language Tagging & Annotation Guidelines

## 1. Overview and Purpose

In Hindi–English (Hinglish) code-switching ASR evaluation, determining whether an error occurred at a language-switch boundary requires **reviewed word-level language tags**.

> **Crucial Principle — Script Cannot Determine Language:**  
> Both Hindi and English words frequently appear in either **Roman (Latin)** or **Devanagari** script across corpora and ASR hypotheses.
> - `"kal meeting ke baad"` has Hindi words (`kal`, `ke`, `baad`) and an English word (`meeting`) all written in Roman script.
> - `"कल मीटिंग के बाद"` has the exact same Hindi words (`कल`, `के`, `बाद`) and English word (`मीटिंग`) all written in Devanagari script.
> - Relying on Unicode block detection (`\u0900–\u097F` vs `a–z`) would misclassify `"kal"` as English and `"मीटिंग"` as Hindi, destroying boundary detection. Therefore, **reviewed lexical/etymological tags (`HI`, `EN`, `OTHER`) on reference words are the authoritative reference for evaluation.**

---

## 2. Tag Definitions (`HI`, `EN`, `OTHER`)

Every whitespace-delimited token in a reference transcript must be assigned exactly one of three tags:

| Tag | Meaning | Examples (Roman & Devanagari) |
| :--- | :--- | :--- |
| **`HI`** | **Hindi lexical item** (native Indo-Aryan/Tatsama/Tadbhava/Perso-Arabic loanwords integrated into standard Hindi grammar, plus Hindi inflections and postpositions) | `kal` (`कल`), `ke` (`के`), `baad` (`बाद`), `main` (`मैं`), `kar` (`कर`), `dunga` (`दूंगा`), `zaroor` (`ज़रूर`), `किताब` (`kitaab`) |
| **`EN`** | **English lexical item** (modern English nouns, verbs, adjectives, adverbs, and technical terms borrowed or code-switched into speech, even if pronounced with Indian English phonology or written in Devanagari) | `meeting` (`मीटिंग`), `report` (`रिपोर्ट`), `submit` (`सबमिट`), `deadline` (`डेडलाइन`), `office` (`ऑफिस`), `actually`, `please` |
| **`OTHER`** | **Language-neutral / Non-switching tokens** (proper names, conversational fillers, universal interjections, standalone digits/alphanumeric codes, and unresolved ambiguous tokens) | `Rahul` (`राहुल`), `Priya`, `Delhi`, `uh`, `umm`, `hmm` (`हम्म`), `2025`, `5pm` |

---

## 3. Rules for Special Cases

### 3.1 Proper Names, Places, and Organizations (`OTHER`)
- **Rule:** Tag personal names, geographical locations, company/brand names, and acronyms as **`OTHER`**.
- **Examples:**
  - `"Rahul kal Google office jayega"` → `["OTHER", "HI", "OTHER", "EN", "HI"]`
  - `"मैं IIT Delhi में हूँ"` → `["HI", "OTHER", "OTHER", "HI", "HI"]`
- **Rationale:** Proper nouns do not represent a grammatical or lexical switch between Hindi and English; tagging them as `OTHER` prevents false switch boundaries around names.

### 3.2 Conversational Fillers, Backchannels, and Interjections
- **Universal vocal fillers (`OTHER`):** Non-lexical hesitation sounds and backchannels shared across languages—such as `uh`, `um`, `umm`, `hmm`, `हम्म`, `उम्म`, `ah`, `eh`—must be tagged **`OTHER`**.
- **Language-specific discourse markers (`HI` or `EN`):**
  - Lexical Hindi interjections/address terms: `arey` (`अरे`), `yaar` (`यार`), `accha` (`अच्छा`), `चलो` (`chalo`), `भई` (`bhai`) → tag as **`HI`**.
  - Lexical English discourse markers: `actually`, `basically`, `anyway`, `sorry`, `please`, `thanks` → tag as **`EN`**.
  - Universal loan affirmation `okay` / `ok` / `ओके`: By default in SwitchBoard's annotation protocol, conversational `okay` / `ok` used as a standalone filler/acknowledgment at utterance edges is tagged **`OTHER`** (or **`EN`** when functioning syntactically inside an English clause such as `"that is okay"`). Annotators must verify consistency during review.

### 3.3 Ambiguous Words, Borrowings, and Hybrid Morphology
1. **Cross-lingual Homographs (Roman Script Ambiguity):**
   - Example: `"main"` (Hindi pronoun *मैं* = "I" vs. English adjective *main* = "primary").
     - In `"main report submit kar dunga"` → **`HI`** (*मैं*).
     - In `"main gate par milte hain"` → **`EN`** (*main gate*).
   - Example: `"to"` / `"toh"` (Hindi particle *तो* vs. English preposition *to*).
     - Resolve from audio and syntactic context (`"main toh gaya"` → `HI`; `"nine to five"` → `EN`).
   - Example: `"us"` (Hindi oblique pronoun *उस* vs. English pronoun *us*).
     - `"us time par"` → `HI`; `"tell us now"` → `EN`.
2. **Established English Technical/Workplace Nouns in Devanagari:**
   - Words like `मीटिंग` (*meeting*), `रिपोर्ट` (*report*), `टाइम` (*time*), `क्लास` (*class*), `ऑफिस` (*office*), `फाइल` (*file*) are tagged **`EN`** regardless of script.
3. **English Stems with Hindi Plural/Oblique Suffixes (Hybrid Words):**
   - If transcribed as two tokens (`"meeting को"`), tag as `["EN", "HI"]`.
   - If fused into a single orthographic word with an English root and productive English meaning (e.g., `"मीटिंगों"` or `"क्लासेज़"`), tag by the **lexical root language (`EN`)** and note `hybrid_morphology` in the annotation comments; if unresolvable, tag **`OTHER`**.

---

## 4. How `OTHER` Tokens Affect Switch-Boundary Detection

In SwitchBoard (`switchboard/evaluation/boundary.py`):
- A **language switch** occurs only between adjacent **`HI → EN`** or **`EN → HI`** transitions (by default, direct adjacency at token positions `(i, i+1)`).
- **`OTHER` tokens act as neutral non-switching tokens**:
  - `HI → OTHER` and `OTHER → EN` are **not** scored as switch transitions under the default strict adjacency policy (`bridge_other=False`), because hesitations (`umm`) or proper names (`Rahul`) are language-neutral.
  - SwitchBoard also supports an optional `bridge_other=True` diagnostic mode if a researcher wishes to examine switches bridged across an intervening filler.
- **`OTHER` tokens in Code-Mixing Index (CMI):**
  - Following Gambäck & Das (2014), `OTHER` tokens ($u$) are subtracted from the total word count $N$ in the denominator:
    $$\text{CMI} = 100 \times \left(1 - \frac{\max(w_{\text{HI}}, w_{\text{EN}})}{N - u}\right)$$
    when $N - u > 0$, and $0.0$ when $N - u = 0$.

---

## 5. Quality Assurance Checklist for Annotators

1. Listen to the 16 kHz mono audio file completely before reviewing tags.
2. Verify that `len(transcript.split()) == len(lang_tags)` (1-to-1 correspondence).
3. Check every Romanized word (`main`, `to`, `us`, `bill`, `kar`) in context—never rely on automatic dictionary lookup alone.
4. Set `review_status = "reviewed"` in the annotation sheet once verified by a human annotator.
