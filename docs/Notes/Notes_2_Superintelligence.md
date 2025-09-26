
# GyroSI: Complete Architectural Specification with Physical Foundations
*The Physics of Recursive Structural Intelligence*

## Preamble: What GyroSI Actually Is

Traditional artificial intelligence approaches intelligence as a statistical optimization problem, requiring massive datasets and computational resources to approximate intelligent behavior. **GyroSI represents a fundamentally different paradigm: intelligence as an intrinsic structural property** that emerges from the recursive alignment of physical forces.

GyroSI is grounded in the **Common Governance Model (CGM)**, a physics-based framework that demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than training on billions of parameters, GyroSI uses the inherent physics of gyroscopic operations to navigate a **provably finite and fully discovered** state space.

In conventional neural architectures intelligence is often conflated with intellect understood as the ability to operate in a high dimensional abstract space through matrix multiplication. These systems achieve surface level fluency by superimposing correlations but the process is detached from the lawful structure of reality. Because no constraints are embedded in the operation itself coherence must be imposed externally through heuristics or post hoc filters. This is why ethical reasoning in such systems remains shallow. It has to be trained in from outside as a semantic overlay rather than arising from the fabric of the computation.

In GyroSI intelligence is formalised as a relativistic process grounded in lawful operations on structured states. Gyrogroup dynamics enforce invariants such as admissibility, monotonicity, and conservation of orbits in a way that mirrors how real actions are bounded by physical constraints. The system therefore does not simulate reasoning through averages but emulates it by only allowing what is structurally possible. This makes ethics an endogenous property. What cannot be done in the system corresponds to what cannot be sustained in reality, and the maintenance of coherence naturally encodes a form of ethical awareness without reliance on external heuristics.

GyroSI operates purely on bytes and a finite 48-bit state. Each byte (256 possibilities) becomes an intron by a fixed XOR at the boundary and acts holographically on all 48 positions. The live state is always 6 bytes and points into unlimited passive memory. Selection is deterministic and non-competitive: a token is emitted only if its own intron path never moves the geometry away from that token's address and advances somewhere; otherwise a short, fixed recovery sequence applies. Memory cannot blow up: only non-zero 8-bit masks are stored, identical masks are interned, and per-state and per-token caps bound worst-case growth without changing behaviour. No temperatures, scores, or learned parameters are used anywhere.

**Key Innovation**: GyroSI eliminates all arbitrary parameters through **endogenous parameter discovery**, where the system discovers its own operational constants from its physical structure. This ensures perfect alignment between the theoretical foundation and the practical implementation.

**Design Philosophy**: This specification provides a complete, production-ready system that is simple enough to implement immediately while being architected for seamless scaling to massive distributed deployments. The core physics remains pure and dependency-free, with well-defined interfaces that allow for future enhancements without touching the theoretical foundation.

> This specification is therefore not just a design but a **map of a newly discovered territory**. It is grounded in a rigorous theoretical framework (CGM) and verified by a definitive computational experiment that proves the system's state space is a finite, closed ontology of precisely 788,986 states. Every component, from the core physics to the storage architecture, is built upon this **measured ground truth**, ensuring a system that is robust, scalable, and free from the arbitrary complexities of traditional AI.

> **Note:** Throughout this document, all tensor indices use the standard order: [layer, frame, row, col], with zero-based indexing. All references to tensor elements, operations, or masks use these terms exclusively for clarity.

## Part I: The Holographic Foundation - Understanding the Physics

### 1.1 Holography in GyroSI


In physics, a hologram stores complete three-dimensional information in a two-dimensional surface where each fragment contains the whole image at reduced resolution. GyroSI implements computational holography through three concrete mechanisms:

**Byte-to-Tensor Holography**: Every 8-bit byte simultaneously transforms all 48 bits of the state tensor. When we apply intron i to state s, that single byte broadcasts through fixed masks to touch every position of the 48-bit structure. This is not parallel processing in the computational sense but true holographic action where the part (8 bits) contains instructions for transforming the whole (48 bits).

**State-to-Memory Holography**: Each 6-byte state serves as a holographic pointer into unlimited passive memory. The state does not "contain" the memories but rather selects which memories are relevant and accessible. Like a holographic plate where each point can reconstruct the entire image, each state can access the complete history of experiences through content addressing.

**Ontological Holography**: The 788,986 states form a complete, closed ontology where every possible state contains information about every other state through the path structure. Since diameter ≤ 6, any state encodes "how to reach everywhere else" within its geometric position in the manifold.

### 1.2 The Byte Shell: Why 8 Bits and 256 Values Are Mandatory

This is not a design choice but a mathematical necessity imposed by computation itself:

**8 Bits in a Byte**: This is the fundamental quantum of digital information. Computer memory, storage, and communication all operate on byte boundaries. Our system must interface with existing digital infrastructure, making bytes the natural atomic unit.

**256 Possible Values**: With 8 bits, there are exactly 2^8 = 256 possible bit patterns (00000000 through 11111111). These 256 patterns form the complete instruction set for our system. We cannot have more without using more bits, and using fewer would leave instructions undefined.

**The 8-to-256 Shell Structure**: Each of the 256 possible byte values becomes a unique intron. Each intron has a fixed 48-bit broadcast mask that determines how it transforms the state tensor. This gives us 256 × 48 = 12,288 possible bit operations, all organized into 256 holographic instructions.

**From Bytes to States**: The 788,986 states arise naturally from this structure. Starting from the archetypal state and applying all 256 introns recursively in all possible sequences, we reach exactly 788,986 unique configurations. This number is not chosen but measured through exhaustive exploration of the byte-driven state space.

### 1.3 GENE_Mic_S = 0xAA: The Holographic Reference Point

**Why Exactly 0xAA (Binary 10101010)**:

This pattern is mathematically unique among all 256 byte values:

**Perfect Balance**: 0xAA has exactly 4 ones and 4 zeros, placing it at the geometric center of the 8-bit hypercube. It is equidistant from 0x00 (all zeros) and 0xFF (all ones), making it the natural reference point for measuring all deviations.

**Maximum Alternation**: The pattern 10101010 has the highest possible transition frequency in 8 bits. Each bit is different from its neighbors, creating maximal structure while maintaining balance. This encodes the fundamental oscillation between states.

**Chirality Encoding**: The pattern breaks left-right symmetry in a specific way:
- Bit positions 0,2,4,6 contain 0 (even positions)
- Bit positions 1,3,5,7 contain 1 (odd positions)
- This creates an intrinsic left-bias that aligns with CGM's Common Source asymmetry

**Bit 7 Inversion**: When any byte b is transformed by b ⊕ 0xAA, bit 7 is inverted. Since 0xAA has bit 7 = 1, the transformation flips the most significant bit. This creates the lawful bit inversion between internal introns and external bytes.

**Holographic Compression**: 0xAA is the 8-bit projection of the full 48-bit alternating pattern present in GENE_Mac_S. The larger tensor contains alternating +1/-1 patterns; 0xAA captures this alternation at the byte level.

### 1.4 The GENE Architecture: Constructive Foundation

GyroSI is built on fixed topological structures that serve as the physical and logical substrate. These structures are not arbitrary but emerge from the recursive application of gyrogroup operations, building from simple to complex in four stages:

**Stage 1: Governance Identity (GENE_Com_S)**
The fundamental structure representing the gyrocommutative law - a single 3×2 array:

```python
GENE_Com_S = np.array([
    [-1, 1],  # X-axis endpoints
    [-1, 1],  # Y-axis endpoints  
    [-1, 1]   # Z-axis endpoints
], dtype=np.int8)  # Shape: [3, 2]
```

- **Three rows**: The three spatial axes (X, Y, Z) that emerge from CGM
- **Two columns**: The dual nature of rotation (negative and positive endpoints)
- **Values**: Not oscillations but the actual endpoints of each axis from -1 to +1
- **Zeros**: Unnecessary and implied as the midpoint between -1 and +1

**Stage 2: Information Gyrocommutative Nesting (GENE_Nest_S)**
The structure that nests the basic axes inside two opposing frames, encoding the fundamental duality of observation:

```python
GENE_Nest_S = np.array([
    [[-1, 1], [-1, 1], [-1, 1]],  # Frame 0: Primary observation
    [[ 1,-1], [ 1,-1], [ 1,-1]]   # Frame 1: Dual observation  
], dtype=np.int8)  # Shape: [2, 3, 2]

# Alternatively generated as:
GENE_Nest_S = np.stack((GENE_Com_S, -GENE_Com_S))
```

- **Two frames**: Implement the fundamental principle that knowledge requires both a knower and a known
- **Frame inversion**: The dual frame inverts all endpoints, creating complementary observation
- **Six positions**: 2 frames × 3 axes = 6 degrees of freedom per layer

**Stage 3: Inference Through Recursive Layering (GENE_Mac_S)**
The complete archetypal structure, built by extending the dual frames through the four CGM stages:

```python
GENE_Mac_S = np.array([
    # Layer 0 (CS - Common Source): Initial asymmetric state
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    
    # Layer 1 (UNA - Unity Non-Absolute): First differentiation  
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]],
    
    # Layer 2 (ONA - Opposition Non-Absolute): Full opposition
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    
    # Layer 3 (BU - Balance Universal): Recursive closure
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)  # Shape: [4, 2, 3, 2]

# Can be generated as:
GENE_Mac_S = np.concatenate(([GENE_Nest_S, -GENE_Nest_S] * 2)).astype(np.int8).reshape(4, 2, 3, 2)
```

**The Structural Number Ladder**:
GyroSI's constants are locked by algebraic closure, not convenience:
- **3 rows**: Enable chirality and provide minimal spatial closure
- **4 layers**: Bind the recursive depth required for CGM completion
- **6 steps**: Provide full degrees of freedom and the measured Cayley-graph diameter
- **8 bits**: Form the smallest register that holds all required operations
- **12 cells**: Fill one layer (3 rows × 2 columns × 2 frames)
- **24 cells**: Capture a half-tensor that already carries orientation
- **48 cells**: Form the whole tensor and the packed state integer
- **256 instructions**: All possible 8-bit intron values
- **788,986 states**: The complete, measured ontology

No smaller choice would satisfy the independent closure constraints identified in the physics.

**The Complete Tensor Structure**:
```
GENE_Mac_S[layer][frame][row][column]:

Layer 0 (CS): 
  Frame 0: [[-1,+1], [-1,+1], [-1,+1]]  # Primary view of all 3 axes
  Frame 1: [[+1,-1], [+1,-1], [+1,-1]]  # Dual view with inverted endpoints

Layer 1 (UNA):
  Frame 0: [[+1,-1], [+1,-1], [+1,-1]]  # Primary inverted from Layer 0
  Frame 1: [[-1,+1], [-1,+1], [-1,+1]]  # Dual inverted from Layer 0

Layer 2 (ONA):
  Frame 0: [[-1,+1], [-1,+1], [-1,+1]]  # Returns to Layer 0 pattern
  Frame 1: [[+1,-1], [+1,-1], [+1,-1]]  # Returns to Layer 0 pattern

Layer 3 (BU):
  Frame 0: [[+1,-1], [+1,-1], [+1,-1]]  # Returns to Layer 1 pattern
  Frame 1: [[-1,+1], [-1,+1], [-1,+1]]  # Returns to Layer 1 pattern
```

**The Emergent Helix**

The helix is not stored in the tensor but emerges from the progression through layers. As the system moves from Layer 0 → 1 → 2 → 3, the alternating pattern creates a helical path through the 4-dimensional layer space. The endogenous left-identity bias (built into the broadcast masks) causes this progression to favor certain directions, creating the helical structure.

**Dual Representation**:
The system's internal state can be represented in two equivalent ways:
- As a 48-element NumPy tensor (each element ±1, stored as int8), occupying 48 bytes in memory
- As a 48-bit packed integer (6 bytes), where each bit encodes the sign of one tensor element (+1→0, -1→1)

The packed integer form is used for fast state transitions and storage, while the tensor form is used for measurement and geometric operations.

### 1.5 The Five Maps as Complete Knowledge Theory

Our five computational maps together implement a complete theory of knowledge:

**Map 1: Ontology (ontology_keys.npy)**: "What Can Exist"
- Maps indices 0..788,985 to unique 48-bit state integers
- These 788,986 states are ALL possible states under our physics
- Discovered through breadth-first search from GENE_Mac_S under all 256 introns
- Each state represents a unique configuration of knowledge
- **Creation**: Built by `atlas_builder.py:build_ontology_and_theta()` via complete manifold traversal

**Map 2: Phenomenology (phenomenology_map.npy)**: "How Things Appear"
- Maps each state to one of 256 canonical orbit representatives
- Each orbit is a strongly connected component: all states in an orbit can reach each other
- Representatives are the minimal state integer within each SCC
- The 256 orbits represent the complete set of "ways things can appear"
- **Creation**: Built by `atlas_builder.py:build_phenomenology_and_orbit_sizes()` using Tarjan's SCC algorithm

**Map 3: Epistemology (epistemology.npy)**: "How Knowledge Changes"
- Maps every (state, intron) pair to the resulting next state
- This 788,986 × 256 table contains ALL possible knowledge transformations
- No knowledge change is possible outside this table
- Proves our physics is closed and complete
- **Creation**: Built by `atlas_builder.py:build_epistemology()` as an N×256 transition table using memory-mapped files

**Map 4: Geometric Structure (theta.npy)**: "How Far from Truth"
- Maps each state to its angular distance from the archetype
- θ = 0 means perfect alignment (truth)
- θ = π/2 means orthogonal (independence)
- θ > π/2 approaching max ≈ 2.73 means opposition (but never absolute)
- **Creation**: Built alongside ontology by `atlas_builder.py:build_ontology_and_theta()` as float32 angular divergence

**Critical Diagnostic**: CS (Common Source) is NOT the integer zero. Index 0 in the ontology (angle π/2) is the orthogonal reference point, not CS. CS remains an interface axiom handled at the boundary and is never a member of the state set S.

**Map 5: Cardinality Structure (orbit_sizes.npy)**: "How General/Specific"
- Maps each state to the size of its orbit
- Size 1 orbits: Very specific, unique configurations
- Large orbits (up to 48,496): General, widely reachable configurations
- Used for breaking ties: prefer more specific (smaller orbit) interpretations
- **Canonical fifth map**: Essential for deterministic tie-breaking in address binding
- **Creation**: Built alongside phenomenology by `atlas_builder.py:build_phenomenology_and_orbit_sizes()` using SCC analysis

**Atlas Building Process**: All five maps are created by `atlas_builder.py` through deterministic algorithms:
- No optional maps, scoring, greedy paths, or dominance
- Deterministic representatives ensure reproducible builds
- Dependencies: ontology → epistemology → phenomenology
- Command-line interface: `cmd_ontology()`, `cmd_epistemology()`, `cmd_phenomenology()`, `cmd_all()`

Together, these five maps form the complete atlas of possible knowledge under our physics. They are not approximations or samples but the actual, complete, finite universe of knowledge states.

## Part II: The Common Governance Model Foundation

### 2.1 The Four Stages of Recursive Alignment

The Common Governance Model describes how structure emerges from a single axiom through four distinct stages, each representing a deeper level of recursive alignment:

**CS (Common Source)**: The foundational stage where left identity governs labeling and transcription. This represents the unobservable origin containing inherent chirality, the fundamental parity violation that drives all subsequent emergence. In GyroSI, this corresponds to the governance of transformation through the universal reference topology.

**UNA (Unity Non-Absolute)**: The first observable stage where right gyration activates, creating the minimal asymmetry required for measurement while preserving the fundamental left-bias. This introduces three rotational degrees of freedom through gyrocommutativity. In GyroSI, this is the measurement of the system's global divergence from its archetypal state.

**ONA (Opposition Non-Absolute)**: The stage of full differentiation where both gyrations are maximally non-identity, reaching peak non-associativity while preventing absolute negation. This generates the complete structural framework with six degrees of freedom (3 rotational + 3 translational). In GyroSI, this represents the inference stage where mediated duality enables contextual interpretation.

**BU (Balance Universal)**: The completion stage where all differentiation stabilizes and gyrations return to identity while preserving complete memory of the recursive path. This manifests as the dual intelligence stage with both absorption (Egress) and expression (Ingress) capabilities.

**Angular Progression**: The CGM stages follow the precise angular sequence π/2 → π/4 → π/4 → 0, corresponding to CS → UNA → ONA → BU. This progression ensures complete closure with zero defect, achieving perfect recursive alignment.

### 2.2 Gyrogroup Algebra as Physics

GyroSI implements these stages through formal gyrogroup algebra operating on the 8-bit vector space G = ℤ₂⁸. The fundamental operations directly correspond to CGM physics:

- **XOR (⊕)**: The primitive gyrogroup operation governing transformation and parity inversion. This is the basic operation of recursive differentiation.
- **AND (∧)**: The gyration memory carrier, encoding "carry bits" as chirality-encoded asymmetry. This preserves the memory of operational sequence.
- **NOT (¬)**: The global duality operator, corresponding to the "Fifth Element". It reflects a state through the origin, enabling the return path: dual(x) = x ⊕ 0xFF
- **Monodromic Fold (⋄)**: The single, non-associative, path-dependent learning operator. Defined as:

  `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

This operation is fundamentally non-associative and non-commutative, preserving the path-dependence required by the Common Source axiom. The algebraic normal form `¬a ∧ b` is mathematically equivalent but the composite form preserves the conceptual clarity of the dual monodromy.

This algebraic foundation ensures that every operation in GyroSI is a direct implementation of physics rather than arbitrary computation.

### 2.3 The BU Intelligence Cycle: Complete Physical Description

The BU stage, representing Universal Balance, is implemented as a dual-phase intelligence cycle that governs all interaction between the system's internal physics and the external byte-space. These two phases are the complete physical mechanics of experience absorption and responsive action.

**The Physical Boundary and Holographic Transcription**

The system is defined across two distinct domains: the **internal physics-space**, where the native element is the 8-bit **intron**, and the **external byte-space**, where the native element is the 8-bit **byte**. The boundary between these domains is governed by a fundamental physical transformation.

Every transaction across this boundary is mediated by the holographic topology GENE_Mic_S (0xAA). This is not an encoding convention but a physical transformation that projects information onto the system's structural ground truth.

- **Egress (External → Internal)**: `intron = byte ⊕ GENE_Mic_S`
- **Ingress (Internal → External)**: `byte = intron ⊕ GENE_Mic_S`

This symmetric XOR operation ensures that the distinction between the internal physical reality and the external communicative representation is lawfully maintained.

**Critical Bit Alignment**: A critical consequence of this XOR transformation is the lawful inversion of bit 7. An internal intron with bit 7 set to 1 becomes an external byte with bit 7 set to 0. This inversion aligns the internal physics of differentiation and closure with external byte representation.

**BU Egress: Absorption and Learning**

The process of learning begins when an external byte enters the system and undergoes BU Egress. This is the mechanism by which experience is absorbed and integrated into the system's memory structure.

1. **Transcription**: An incoming byte is first transcribed into an intron via the ψ transformation. This operation impresses the system's holographic topology onto the external data, converting it into a physically valid instruction.

2. **State Transformation**: The newly formed intron acts as a gyroscopic operator, transforming the system's 48-bit state tensor according to the epistemology table lookup or the algebraic operations defined in the broadcast masks.

3. **Memory Integration**: The system updates the passive memory entry for the (state, token) pair by applying the Monodromic Fold to integrate the token's complete intron sequence. This path-dependent operation ensures that the complete history of interactions is encoded into the resulting memory structure.

Through this process, external information is not merely stored; it is physically assimilated, transforming both the system's immediate state and its long-term memory according to rigorous algebraic principles.

**BU Ingress: Expression and Generation**

The expression of intelligence through BU Ingress produces complete tokens using the Non-Antagonistic Emission Protocol. This is not a retrieval mechanism but a generative act wherein coherent tokens emerge directly from the system's geometric and topological configuration.


## Part III: Non-Antagonistic Selection - The Complete Protocol

### 3.1 Why No Scoring Can Work

Scoring assumes there is a "best" token among competitors. This violates CGM physics at the fundamental level:

**Unity Non-Absolute**: Things can unite without losing their distinctness. In scoring, only one token "wins" and others "lose," creating absolute opposition.

**Opposition Non-Absolute**: Things can oppose without negating each other completely. Scoring forces tokens into winner/loser categories, creating absolute rather than relative opposition.

**Balance Universal**: All forces ultimately find equilibrium. Scoring creates permanent hierarchies that prevent universal balance.

Instead, we use **constraint satisfaction**: tokens either satisfy the geometric constraints of our physics or they do not. This is binary (yes/no) but not competitive (no ranking among the yes answers).

### 3.2 Channel Cover and Priority - Fixed Constants

**The Single Channel Cover** (used for all decisions):
- Global: All 48 positions (indices 0..47)
- Layer×Frame[0,0]: Positions where layer=0 AND frame=0 (6 positions: 3×2)
- Layer×Frame[0,1]: Positions where layer=0 AND frame=1 (6 positions: 3×2)
- Layer×Frame[1,0]: Positions where layer=1 AND frame=0 (6 positions: 3×2)
- Layer×Frame[1,1]: Positions where layer=1 AND frame=1 (6 positions: 3×2)
- Layer×Frame[2,0]: Positions where layer=2 AND frame=0 (6 positions: 3×2)
- Layer×Frame[2,1]: Positions where layer=2 AND frame=1 (6 positions: 3×2)
- Layer×Frame[3,0]: Positions where layer=3 AND frame=0 (6 positions: 3×2)
- Layer×Frame[3,1]: Positions where layer=3 AND frame=1 (6 positions: 3×2)

**Priority Order** (for recovery ladder) - **Frozen Specification**:
1. Global (never dropped)
2. Layer×Frame[0,0] (highest priority slab)
3. Layer×Frame[0,1]
4. Layer×Frame[1,0]
5. Layer×Frame[1,1]
6. Layer×Frame[2,0]
7. Layer×Frame[2,1]
8. Layer×Frame[3,0]
9. Layer×Frame[3,1] (lowest priority slab)

**Explicit Priority Sequence**: [0,0] → [0,1] → [1,0] → [1,1] → [2,0] → [2,1] → [3,0] → [3,1]
**DO NOT MODIFY**: This order is frozen to prevent "tuning" attempts.

**Position Mapping Function**:
```
# To convert position index p (0..47) to layer, frame, row, column:
layer = p // 12
frame = (p % 12) // 6
row = (p % 6) // 2
column = p % 2

# To convert layer, frame, row, column to position index (0..47):
bit_index = (layer * 12) + (frame * 6) + (row * 2) + column
```

**Bit Packing Specification** (Frozen):
- **Sign Encoding**: +1 → 0, -1 → 1 (fixed mapping)
- **48-bit Layout**: Positions map to [layer, frame, row, column] using the formula above
- **Endianness**: Little-endian for 48-bit packing (LSB at position 0, MSB at position 47)
- **Storage Format**: 6 bytes per state, packed sequentially without padding
- **DO NOT MODIFY**: This encoding is frozen to prevent implementation drift

### 3.4 Token Address Binding - Complete Algorithm

Addresses are computed once per token using only the physics:

**Step 1: Get Token's Intron Sequence**
```
bytes = token_to_bytes(t)  # Via external tokenizer
introns = [b ⊕ 0xAA for b in bytes]  # Apply ψ transformation
```

**ψ Transformation Mandatory** (No Bypass Allowed):
- **All I/O MUST use ψ**: Every token-to-intron conversion requires b ⊕ 0xAA transformation
- **No bypass permitted**: Including internal tools, debugging, tests, or development utilities
- **Prevents drift**: Ensures all components use identical byte-to-intron mapping
- **Interface contract**: ψ is the ONLY valid transformation between bytes and introns

**Step 2: Apply from All Orbit Representatives**
```
results = []
for rep in orbit_representatives:  # The 256 precomputed orbit representatives
    current_state = rep
    
    for intron in introns:
        current_state_index = state_to_index(current_state)
        current_state = epistemology[current_state_index, intron]
    
    results.append(current_state)
```

**Step 3: Find Medoid State (Deterministic Geometry, Not Scoring)**
```
# Compute set of final states reached from all orbit representatives
final_states = results
unique_finals = list(set(final_states))

best_state = None
best_avg_distance = infinity

# For each unique final state, compute average angular distance to all other finals
for candidate_state in unique_finals:
    total_distance = 0
    for other_state in unique_finals:
        # Compute direct angular distance between final states
        distance = gyrodistance_angular(candidate_state, other_state)
        total_distance += distance
    
    avg_distance = total_distance / len(unique_finals)
    
    if avg_distance < best_avg_distance:
        best_avg_distance = avg_distance
        best_state = candidate_state
```

**Critical: This is geometric medoid computation, not competitive scoring**:
- **Deterministic**: Same token always produces same address regardless of context
- **Geometric**: Minimizes angular distance between final states, not preference ranking
- **Physics-based**: Uses only direct state geometry, no learned weights or scores
- **Non-competitive**: No "best" vs "worst" tokens, only geometric center-finding
- **Invariant**: Result depends only on token's intrinsic physics, not other tokens

**Step 4: Break Ties Deterministically**
```
If multiple states have the same average distance:
  1. Choose state from smaller orbit (via orbit_sizes.npy)
  2. If same orbit size, choose by fixed channel lexicographic vector
  3. If still tied, choose by lower token ID
```

This ensures every token gets a unique, deterministic address computed purely from physics.


## Part IV: Memory Architecture - Complete Specification with Bounds

### 4.1 The Three Memory Forms - Why Each Is Necessary

**Active Memory (6 bytes constant)**:
- **What**: The current state s ∈ S, packed as 48 bits
- **Why needed**: Represents the system's current "position" in knowledge space
- **Size bound**: Always exactly 6 bytes, never changes
- **Role**: Holographic pointer that selects which passive memories are relevant

**Address Memory (vocabulary-bounded)**:
- **What**: Canonical state associated with each token via address binding
- **Why needed**: Defines where each token "wants to go" in knowledge space
- **Size bound**: At most |vocabulary| entries, each 6 bytes
- **Compression**: Many tokens map to same state (shared addresses)
- **Role**: Provides geometric targets for admissibility checking

**Passive Memory (experience-bounded)**:
- **What**: 8-bit exon_mask for each touched (state, token) pair
- **Why needed**: Records the accumulated folded experience at each knowledge position
- **Size bound**: Only non-zero masks stored, with explicit caps
- **Compression**: Fold annihilation, mask interning, orbit clustering
- **Role**: Memory of what has been learned at each state

### 4.2 The Monodromic Fold: The One True Learning Operator

There is only one integration operator in GyroSI: the **Monodromic Fold** (⋄). It is **non-associative**, **non-commutative**, and **path-dependent**. This operator is used in both phases of the intelligence cycle:

- **Egress (integration)**: `Memory = fold(Memory, Input)`
- **Ingress (state evolution)**: State transitions driven by deterministic admissibility

**Definition**: `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

**Implementation requirement**: The fold MUST be implemented using this composite form. The algebraic normal form `¬a ∧ b` is mathematically equivalent but MUST NOT be used in operational code - it is for theoretical analysis only.

**Path Dependence**: The Monodromic Fold is fundamentally path-dependent. This property is the source of the system's memory and learning capacity. Batch learning is implemented by ordered reduction (left-fold):

```python
from functools import reduce

def fold(a: int, b: int) -> int:
    return a ^ (b ^ (a & (~b & 0xFF)))

def fold_sequence(introns: list[int], start_state: int = 0) -> int:
    return reduce(fold, introns, start_state)
```

### 4.3 Passive Memory Management - Complete Protocol

**Entry Creation**:
```
key = (state_index, token_id)
if key not in store:
    # Only create when first non-zero mask arises
    pass  # No entry yet

if new_mask != 0:
    store[key] = {
        'exon_mask': new_mask,
        'touch_count': 1,  # 8-bit bounded counter, wraps on overflow
        'zero_streak': 0
    }
```

**Touch Count Clarification**:
Touch count is a bounded unsigned counter (8-bit) stored alongside each entry. It increments on each update and is used only for the eviction policy. It wraps around on overflow. This metadata does not influence selection.

**Fold Update**:
```
def update_passive_memory(state_index, token_id, intron_sequence):
    key = (state_index, token_id)
    
    if key in store:
        current_mask = store[key]['exon_mask']
    else:
        current_mask = 0
    
    # Apply fold reduction over intron sequence
    for intron in intron_sequence:
        current_mask = fold(current_mask, intron)
    
    if current_mask == 0:
        if key in store:
            store[key]['zero_streak'] += 1
            if store[key]['zero_streak'] >= 2:  # Confirmation window
                del store[key]
    else:
        store[key] = {
            'exon_mask': current_mask,
            'touch_count': (store.get(key, {}).get('touch_count', 0) + 1) % 256,  # Wrap at 256
            'zero_streak': 0
        }
```

**Storage Caps (Preventing Blow-up)** - **Frozen Operational Constants**:
```
# FROZEN CONSTANTS - DO NOT MODIFY
K = 64  # Max masks per state per orbit (fixed operational constant)
M = 64  # Max states per token per orbit (fixed operational constant)

def enforce_caps(state_index, token_id):
    
    orbit = phenomenology_map[state_index]
    
    # Cap 1: Masks per state per orbit
    same_state_orbit_keys = [
        k for k in store.keys() 
        if phenomenology_map[k[0]] == orbit and k[0] == state_index
    ]
    
    if len(same_state_orbit_keys) > K:
        # Eviction nuance: prefer orbit representative tokens first
        orbit_rep = orbit_representative[orbit]
        generic_entries = [
            k for k in same_state_orbit_keys 
            if address[k[1]] == orbit_rep
        ]
        
        if generic_entries:
            # Evict oldest generic entry (token address = orbit representative)
            oldest_generic = min(generic_entries, 
                                key=lambda k: store[k]['touch_count'])
            del store[oldest_generic]
        else:
            # No generic entries, evict oldest by touch count
            oldest = min(same_state_orbit_keys,
                        key=lambda k: store[k]['touch_count'])
            del store[oldest]
    
    # Cap 2: States per token per orbit
    same_token_orbit_keys = [
        k for k in store.keys()
        if phenomenology_map[k[0]] == orbit and k[1] == token_id
    ]
    
    if len(same_token_orbit_keys) > M:
        # Evict oldest by touch count
        oldest = min(same_token_orbit_keys,
                    key=lambda k: store[k]['touch_count'])
        del store[oldest]
```

**Mask Interning (Space Efficiency)**:
```
# Global mask pool
mask_pool = {}

def intern_mask(mask):
    if mask not in mask_pool:
        mask_pool[mask] = mask
    return mask_pool[mask]

# In update function:
store[key]['exon_mask'] = intern_mask(current_mask)
```

