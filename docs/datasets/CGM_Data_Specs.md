# CGM Dataset Specification

> Specification for a dataset for training AI systems on the Common Governance Model (CGM).

This document defines a **single-file dataset format** (`cgm_dataset.jsonl`) that can be incrementally constructed from the CGM documentation (e.g., `README.md`, `docs/CGM_Program.md`, analysis notes, etc.).

The goals are:

- To be **easy to maintain manually**.
- To keep all data in **one file and one format**.
- To encode **faithful, source-grounded** knowledge of CGM.
- To be suitable for **fine-tuning, RAG, and evaluation**.

---

## 1. File Format

The dataset is stored as a **JSON Lines** file:

- File name: `cgm_dataset.jsonl`
- Each line: one JSON object (a “record” or “example”)
- No arrays at top level; each line is independent

Example structure:

```text
{"id": "cgm_001", "source": "...", ...}
{"id": "cgm_002", "source": "...", ...}
{"id": "cgm_003", "source": "...", ...}
...
```

---

## 2. Record Schema

Each record encodes one question–answer pair grounded in a specific part of the CGM documentation.

### 2.1 Required Fields

| Field       | Type          | Description |
|------------|---------------|-------------|
| `id`       | string        | Unique identifier, e.g. `cgm_001`, `cgm_002`, ... (sequential is fine). |
| `source`   | string        | Source file path, e.g. `README.md`, `docs/CGM_Program.md`. |
| `section`  | string        | Section heading or heading path within the source document. |
| `category` | string (enum) | High-level topical category (see §3.1). |
| `type`     | string (enum) | Type of knowledge captured (see §3.2). |
| `question` | string        | Natural-language question about CGM. |
| `answer`   | string        | Faithful, grounded answer based on the source. |
| `context`  | string        | Relevant verbatim or near-verbatim excerpt from the source text. |
| `tags`     | array[string] | Supplemental free-form tags (see §3.3 for recommended values). |
| `importance` | string (enum) | Priority for training: `core`, `supporting`, or `detail`. |

All other fields are optional.

### 2.2 Optional Fields

You may add optional helper fields if useful:

| Field  | Type   | Description |
|--------|--------|-------------|
| `notes` | string | Editorial notes or curation comments (not used by models). |

Optional fields must not change the meaning of required fields.

---

## 3. Controlled Vocabularies

To keep the dataset consistent, some fields use controlled vocabularies.

### 3.1 `category` (topical area)

Use one of:

- `axiom` — Foundational constraints and logical axioms.
- `derivation` — Mathematical or logical derivations (e.g., 3D necessity).
- `invariant` — Geometric or physical invariants (e.g., `Q_G = 4π`).
- `prediction` — Concrete predictions CGM makes about observables.
- `cosmology` — Cosmological interpretation and results.
- `particle_physics` — Particle physics and energy scales.
- `ai_alignment` — GyroDiagnostics, GyroSI, and alignment metrics.
- `method` — Methodological descriptions (tri-partite validation, etc.).
- `empirical_result` — Data-driven results, p-values, observed correlations.
- `reproducibility` — Code mapping and verification instructions.
- `meta` — Author notes, project philosophy, versioning, etc.

Choose the **single most relevant** category for each record.

### 3.2 `type` (knowledge type)

Use one of:

- `concept` — Definition or explanation of a single concept.  
  *Example:* “What is the Common Source (CS) constraint?”
- `equation` — Exact formula or relationship.  
  *Example:* “What is the optical conjugacy relation in CGM?”
- `derivation` — Outline of how a result is derived.  
  *Example:* “How does CGM derive 3D space as necessary?”
- `claim` — CGM’s claim about the nature of reality, physics, or AI.  
  *Example:* “What does CGM claim about the origin of time?”
- `result` — Stated numeric or analytical result.  
  *Example:* “What value does CGM predict for the fine-structure constant?”
- `comparison` — How two things relate.  
  *Example:* “How does CGM’s cosmology differ from standard expansion?”
- `enumeration` — Lists/sets of items.  
  *Example:* “What are the five foundational constraints?”
- `procedure` — Steps for verification, computation, or reproduction.  
  *Example:* “How to verify the axioms using the Z3 solver.”
- `interpretation` — Conceptual or philosophical interpretation.  
  *Example:* “How does CGM interpret the CMB field?”
  
Again, choose a single type per record.

### 3.3 `tags` (free-form but recommended vocabulary)

Tags are flexible, but using a common vocabulary makes the dataset easier to work with. Recommended tags include:

#### Constraints and Logic
- `CS`, `UNA`, `ONA`, `BU_EGRESS`, `BU_INGRESS`
- `modal_logic`, `bimodal_logic`
- `axiomatization`

#### Mathematical Structure
- `BCH`, `lie_algebra`, `su2`, `sl2`
- `hilbert_space`, `GNS`, `unitary_group`
- `gyrogroup`, `monodromy`, `solid_angle`, `toroidal_structure`

#### Physical Invariants and Constants
- `quantum_gravity`, `4pi`
- `fine_structure`, `alpha`
- `aperture_ratio`, `2_07_percent`
- `48_quantization`, `monodromy_hierarchy`
- `balance_index`

#### Physics Domains
- `energy_scales`, `optical_conjugacy`
- `cosmological_constant`, `black_hole_universe`
- `CMB`, `multipole_ladder`, `neutrino`, `proton_decay`

#### AI and Information
- `gyrodiagnostics`, `gyrosi`
- `alignment`, `aperture_observable`, `superintelligence_index`
- `state_space`, `non_associative_operator`

#### Status / Nature of Claim
- `axiom`, `derived`, `verified`, `empirical`, `computational`
- `interpretive`, `conjectural`, `future_work`

You can use multiple tags per record.

### 3.4 `importance`

Use:

- `core` — Central to understanding CGM (constraints, main derivations, key invariants).
- `supporting` — Important but secondary (detailed examples, specific numbers).
- `detail` — Fine points, edge cases, minor clarifications.

---

## 4. Construction Guidelines

These guidelines are intended to keep the dataset internally consistent and faithful to the original work.

### 4.1 Faithfulness and framing

- Answers must be **faithful to the source text**.
- Do not introduce external physics or speculation not present in CGM documents.
- For interpretive or nonstandard claims, phrase answers with:
  - “According to CGM, …”
  - “Within the CGM framework, …”
  - “CGM proposes that …”
  
  rather than asserting them as conventional or universally accepted fact.

### 4.2 Use of `context`

- `context` should contain the **exact or minimally edited excerpt** from the source that supports the answer.
- It can be a full paragraph or multiple short sentences.
- The model can later learn to stay grounded by conditioning on `context`.

Example:

```json
"context": "The Common Governance Model (CGM) is a comprehensive theoretical framework that derives the structure of physical reality and information systems from a single axiomatic principle: \"The Source is Common.\""
```

### 4.3 Question and answer style

- Questions should be **clear, direct, and answerable** from the given context.
- Answers should be:
  - Precise,
  - Concise,
  - Using the terminology of the CGM documentation.
- When relevant, include key formulas in the answer.

### 4.4 Source and section

- `source`: use relative paths from the repository root:  
  e.g., `README.md`, `docs/CGM_Program.md`, `docs/Findings/Analysis_Quantum_Gravity.md`.
- `section`: use the section heading or heading path.  
  For example:
  - `"section": "2.1 The Five Foundational Constraints"`
  - `"section": "4. Geometric Invariants and Physical Constants > 4.1 The Quantum Gravity Invariant: Q_G = 4π"`

Use simple `>`-separated paths if you want to encode hierarchy.

### 4.5 ID assignment

- Use simple, monotonically increasing IDs, e.g.:
  - `cgm_001`, `cgm_002`, `cgm_003`, ...
- The exact numbering is not logically important, but stability is helpful:
  - Once assigned, avoid changing IDs.

---

## 5. Examples

### 5.1 Concept: Common Source (CS)

```json
{
  "id": "cgm_001",
  "source": "docs/CGM_Program.md",
  "section": "2.1 The Five Foundational Constraints",
  "category": "axiom",
  "type": "concept",
  "question": "According to the Common Governance Model, what is the CS (Common Source) constraint and what does it enforce?",
  "answer": "In the CGM framework, the CS (Common Source) constraint is the foundational axiom that enforces a single chiral reference state for observation. Formally, it is written as S → ([R]S ↔ S ∧ ¬([L]S ↔ S)). Right transitions preserve the horizon state S, while left transitions alter it. This establishes fundamental chirality and ensures that all states in a coherent system trace back to a common source.",
  "context": "CS (Common Source): S → ([R]S ↔ S ∧ ¬([L]S ↔ S))\nEstablishes fundamental chirality. Right transitions preserve the reference state (horizon S), while left transitions alter it. This is the seed of parity violation.",
  "tags": ["CS", "constraint", "modal_logic", "axiom", "chirality"],
  "importance": "core"
}
```

### 5.2 Invariant: Quantum Gravity Invariant

```json
{
  "id": "cgm_002",
  "source": "docs/CGM_Program.md",
  "section": "4.1 The Quantum Gravity Invariant: Q_G = 4π",
  "category": "invariant",
  "type": "concept",
  "question": "What does CGM mean by the quantum gravity invariant Q_G = 4π?",
  "answer": "Within the CGM framework, the quantum gravity invariant Q_G = 4π is defined as the complete solid angle in three-dimensional space required for coherent observation. It represents the minimal geometric cost of spacetime observation itself and is treated as a fundamental invariant of quantum gravity, rather than an emergent quantity from other physical inputs.",
  "context": "CGM defines Quantum Gravity as the geometric invariant Q_G = 4π steradians, representing the complete solid angle required for coherent observation in 3D space.\n\nPhysical Meaning: It is the quantum of observability, the minimal cost for spacetime observation itself. Its ubiquitous appearance in physics (Gauss's law, Einstein's equations, quantum normalization) is a signature of this fundamental geometric requirement.",
  "tags": ["quantum_gravity", "4pi", "solid_angle", "invariant"],
  "importance": "core"
}
```

---

## 6. Scope and Coverage

Over time, the dataset should aim to include records covering at least:

- The five foundational constraints (CS, UNA, ONA, BU-Egress, BU-Ingress).
- The derived operational requirements (continuity, reachability, simplicity).
- The central 3D/6-DoF derivation and exclusion of other dimensions.
- The 1–3–6–6 degrees-of-freedom progression.
- Key invariants and constants (e.g., `Q_G = 4π`, fine-structure constant derivation, aperture ratio, 48-quantization).
- Energy scale hierarchy and optical conjugacy relation.
- Cosmological reinterpretation (black hole universe, CMB as residual field, balance index).
- Particle physics implications (neutrino masses, proton lifetime, sterile neutrino non-observability).
- Information-theoretic applications (GyroDiagnostics, GyroSI, aperture observable, superintelligence index).
- Reproducibility and verification methodology (tri-partite validation, code artifacts where described).

The dataset can grow incrementally as new documents and analyses are added.

---

## 7. Licensing

Unless otherwise specified, the dataset derived from this repository is intended to inherit the project’s license (e.g., MIT). Users of the dataset should preserve attribution to the author and the Common Governance Model framework.

---
