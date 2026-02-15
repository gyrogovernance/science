# CGM Science Dataset (Main)

**File:** `cgm_dataset_main.jsonl`  
**Version:** 1.0.0  
**Date:** 2026-02-15  
**Total Records:** 1,024

## Overview

This dataset contains a structured knowledge base derived from the **Common Governance Model (CGM)** framework. It transforms the project's technical documentation, formal proofs, geometric analyses, and cosmological findings into machine-readable instruction data.

It is designed for:
1.  **Fine-Tuning (SFT):** Training LLMs to understand CGM-specific terminology, logic, and physics derivations.
2.  **RAG Systems:** Providing a "Gold Standard" retrieval corpus with verifiable ground-truth contexts.
3.  **Evaluation:** Benchmarking AI reasoning capabilities on novel axiomatic frameworks.

## Dataset Structure

The dataset is provided in **JSON Lines (JSONL)** format. Each line is a valid JSON object representing a single Question-Answer pair grounded in specific source text.

### Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Unique identifier (e.g., `cgm_001`). Stable across versions. |
| `source` | String | Relative path to the origin Markdown file (e.g., `docs/CGM_Paper.md`). |
| `section` | String | The specific heading or hierarchy where the information resides. |
| `category` | String | Domain tag (e.g., `axiom`, `cosmology`, `ai_alignment`). |
| `type` | String | Knowledge structure (e.g., `concept`, `equation`, `derivation`). |
| `question` | String | A natural language query about the content. |
| `answer` | String | A faithful, accurate response derived *strictly* from the source text. |
| `context` | String | The verbatim excerpt from the document used to ground the answer. |
| `tags` | List | Searchable keywords (e.g., `["4pi", "aperture", "Q_G"]`). |
| `importance`| String | Training weight: `core`, `supporting`, or `detail`. |

### Example Record

```json
{
  "id": "cgm_062",
  "source": "docs/CGM_Paper.md",
  "section": "Gyrogroup Theory: The Physical Realization > Geometric Invariants",
  "category": "invariant",
  "type": "concept",
  "question": "What is Q_G in the paper's geometric invariants section?",
  "answer": "Q_G is presented as the quantum gravity horizon invariant, equal to 4π, and interpreted as an operational horizon quantity that later manifests as total solid angle in the L²(S²) realization.",
  "context": "##### The Quantum Gravity Horizon: Q_G = 4π",
  "tags": ["Q_G", "quantum_gravity", "horizon", "4pi"],
  "importance": "core"
}
```

## Content Coverage

The dataset covers the full breadth of the CGM research program:

*   **Foundational Logic:** The 5 constraints (CS, UNA, ONA, BU-Egress, BU-Ingress) and modal logic proofs.
*   **Geometric Derivations:** Proof of 3D space, 6 Degrees of Freedom (DOF), and Gyrogroup algebra.
*   **Physical Constants:** Geometric derivation of the Fine-Structure Constant ($\alpha$), Proton Lifetime, and Neutrino Masses.
*   **Cosmology:** The Black Hole Universe hypothesis, Optical Illusion of expansion, and UV-IR Optical Conjugacy.
*   **AI Alignment:** GyroDiagnostics, Tetrahedral Measurement topology, and the 2.07% Aperture requirement for evolutionary capacity.
*   **Dynamics:** Gyrational motion and Universal Correction Operators for galactic rotation curves.

## Source Documents

The data was extracted from the following high-fidelity analysis documents:

*   `docs/CGM_Paper.md` (Core Framework)
*   `docs/CGM_Program.md` (Research Roadmap)
*   `Analysis_3D_6DOF_Proof.md` (Dimensional Necessity)
*   `Analysis_Hilbert_Space_Representation.md` (Quantum Mechanics Bridge)
*   `Analysis_CGM_Units.md` (Geometric Units & Constants)
*   `Analysis_Monodromy.md` (Geometric Memory)
*   `Analysis_Fine_Structure.md` ($\alpha$ Derivation)
*   `Analysis_Energy_Scales.md` (Energy Hierarchy)
*   `Analysis_Quantum_Gravity.md` ($Q_G = 4\pi$)
*   `Analysis_48_States.md` (Factor 48 & Angular Harmonics)
*   `Analysis_BH_Aperture.md` (Black Hole Thermodynamics)
*   `Analysis_BH_Universe.md` (Cosmological Model)
*   `Analysis_Measurement.md` (AI & Collective Intelligence)
*   `Analysis_Capacity_Concepts.md` (Capacity Synthesis)
*   `Analysis_Motion.md` (Gyrational Dynamics)
*   `Analysis_Universal_Corrections.md` (Galactic Dynamics)

## Usage

### Loading in Python

```python
import json
import pandas as pd

# Option 1: Load as list of dicts
data = []
with open('cgm_dataset_main.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Option 2: Load into Pandas DataFrame
df = pd.DataFrame(data)

# Filter for specific topics
cosmology_df = df[df['category'] == 'cosmology']
print(f"Loaded {len(df)} records.")
```

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='cgm_dataset_main.jsonl')
# Returns a DatasetDict with a 'train' split
print(dataset['train'][0])
```

## License & Citation

This dataset derives from the Common Governance Model framework.

**Citation:**
> Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**License:** MIT
