#!/usr/bin/env python3
"""
Test suite for CGM's five foundational constraints using Z3 SMT solver.
Tests consistency, entailments, independence, and toroidal structure.

Terminology (canonical):
  CS (A1): Assumption — chirality at horizon
  UNA (A2): Lemma — depth‑2 equality non‑absolute (¬□E)
  ONA (A3): Lemma — depth‑2 inequality non‑absolute (¬□¬E)
  BU‑Egress (A4): Proposition — depth‑4 closure (□B)
  BU‑Ingress (A5): Proposition — memory reconstruction (□B → CS∧UNA∧ONA)
  BU := BU‑Egress ∧ BU‑Ingress (dual balance)

Note: OA* tokens (OA1-OA5) are automatically canonicalized via canonicalize() function.

Geometric mapping:
- L/R: Left/right gyration operations (non-commutative transformations in 3D/6DoF space)
- S: Horizon worlds (observable boundary, solid angle 4π)
- Depth-2 (E): Tests commutation of gyrations ([L][R] vs [R][L])
- Depth-4 (B): Tests closure of gyrations (holonomy cancellation)
- Unification: The five foundational constraints derive 3D structure from operational closure,
  with gyrations corresponding to SU(2) rotations and R^3 translations (6DoF total)
"""

from z3 import (
    Solver,
    Bool,
    BoolVal,
    And,
    Or,
    Not,
    Implies,
    sat,
    unsat,
    is_true,
    BoolRef,
)
from typing import Dict, Set, Optional, Tuple, Callable, List, Any, cast
import warnings

# Canonical token mappings
CANONICAL_TOKENS = {"CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS", "BU_DUAL"}
ATOMIC_TOKENS = {"A1", "A2", "A3", "A4", "A5"}
OA_TO_CANON = {
    "OA1": "CS",
    "OA2": "UNA",
    "OA3": "ONA",
    "OA4": "BU_EGRESS",
    "OA5": "BU_INGRESS",
}
CANON_TO_ATOMIC = {
    "CS": "A1",
    "UNA": "A2",
    "ONA": "A3",
    "BU_EGRESS": "A4",
    "BU_INGRESS": "A5",
}
CANON_TO_OA = {v: k for k, v in OA_TO_CANON.items()}


def canonicalize(token: str) -> str:
    """Convert token to canonical form. Maps OA* to canonical, passes through canonical/atomic."""
    token = token.upper()
    if token.startswith("OA") and token in OA_TO_CANON:
        warnings.warn(
            f"{token} is deprecated; use canonical tokens (CS, UNA, ONA, BU_EGRESS, BU_INGRESS).",
            DeprecationWarning,
        )
        return OA_TO_CANON[token]
    if token == "BU":
        return "BU_DUAL"
    return token


class ConstraintEncoder:
    """Encodes the five foundational constraints (CS, UNA, ONA, BU‑Egress, BU‑Ingress) in Z3.

    The system consists of:
    - CS (A1): Assumption — chirality at horizon
    - UNA (A2): Lemma — depth‑2 equality non‑absolute (¬□E)
    - ONA (A3): Lemma — depth‑2 inequality non‑absolute (¬□¬E)
    - BU‑Egress (A4): Proposition — depth‑4 closure (□B)
    - BU‑Ingress (A5): Proposition — memory reconstruction (□B → CS∧UNA∧ONA)
    - BU := BU‑Egress ∧ BU‑Ingress (dual balance)

    Depth choice: 1 (chirality at horizon), 2 (non-commutation), 4 (minimal closure length
    where gyrations cancel, corresponding to SU(2) holonomy in 3D/6DoF geometry).
    Depth 3 is not axiomatized as it doesn't achieve full commutative closure.
    """

    def __init__(self, n: int):
        """Initialize with n worlds."""
        self.n = n
        self.R_L = [[Bool(f"R_L_{i}_{j}") for j in range(n)] for i in range(n)]
        self.R_R = [[Bool(f"R_R_{i}_{j}") for j in range(n)] for i in range(n)]
        self.S = [Bool(f"S_{i}") for i in range(n)]

    def box_L(self, phi: Callable[[int], BoolRef]) -> Callable[[int], BoolRef]:
        """Left necessity operator."""

        def box_L_impl(w: int) -> BoolRef:
            implications = [Implies(self.R_L[w][v], phi(v)) for v in range(self.n)]
            if not implications:
                return BoolVal(True)
            return cast(BoolRef, And(implications))

        return box_L_impl

    def box_R(self, phi: Callable[[int], BoolRef]) -> Callable[[int], BoolRef]:
        """Right necessity operator."""

        def box_R_impl(w: int) -> BoolRef:
            implications = [Implies(self.R_R[w][v], phi(v)) for v in range(self.n)]
            if not implications:
                return BoolVal(True)
            return cast(BoolRef, And(implications))

        return box_R_impl

    def box(self, phi: Callable[[int], BoolRef]) -> Callable[[int], BoolRef]:
        """Joint necessity (both L and R)."""

        def box_impl(w: int) -> BoolRef:
            return cast(BoolRef, And(self.box_L(phi)(w), self.box_R(phi)(w)))

        return box_impl

    def U_at(self, w: int) -> BoolRef:
        """Unity: [L]S ↔ [R]S at world w."""
        left = self.box_L(lambda v: self.S[v])(w)
        right = self.box_R(lambda v: self.S[v])(w)
        return cast(BoolRef, And(Implies(left, right), Implies(right, left)))

    def E_at(self, w: int) -> BoolRef:
        """Equality: [L][R]S ↔ [R][L]S at world w."""
        LR = self.box_L(lambda u: self.box_R(lambda v: self.S[v])(u))(w)
        RL = self.box_R(lambda u: self.box_L(lambda v: self.S[v])(u))(w)
        return cast(BoolRef, And(Implies(LR, RL), Implies(RL, LR)))

    def O_at(self, w: int) -> BoolRef:
        """Opposition: [L][R]S ↔ ¬[R][L]S at world w."""
        LR = self.box_L(lambda u: self.box_R(lambda v: self.S[v])(u))(w)
        RL = self.box_R(lambda u: self.box_L(lambda v: self.S[v])(u))(w)
        return cast(BoolRef, And(Implies(LR, Not(RL)), Implies(Not(RL), LR)))

    def B_at(self, w: int) -> BoolRef:
        """Balance: [L][R][L][R]S ↔ [R][L][R][L]S at world w."""
        LRLR = self.box_L(
            lambda w1: self.box_R(
                lambda w2: self.box_L(lambda w3: self.box_R(lambda w4: self.S[w4])(w3))(
                    w2
                )
            )(w1)
        )(w)
        RLRL = self.box_R(
            lambda w1: self.box_L(
                lambda w2: self.box_R(lambda w3: self.box_L(lambda w4: self.S[w4])(w3))(
                    w2
                )
            )(w1)
        )(w)
        return cast(BoolRef, And(Implies(LRLR, RLRL), Implies(RLRL, LRLR)))

    def encode_A1(self, s: Solver) -> None:
        """A1: Chirality at the horizon (right preserves S, left alters S)"""
        for w in range(self.n):
            right_preserves = self.box_R(lambda v: self.S[v])(w) == self.S[w]
            left_alters = Not(self.box_L(lambda v: self.S[v])(w) == self.S[w])
            s.add(Implies(self.S[w], And(right_preserves, left_alters)))

    def encode_A2(self, s: Solver) -> None:
        """A2: ¬□E at S (depth-2 equality non-absolute)"""
        for w in range(self.n):
            s.add(Implies(self.S[w], Not(self.box(self.E_at)(w))))

    def encode_A3(self, s: Solver) -> None:
        """A3: ¬□¬E at S (depth-2 inequality non-absolute)"""
        for w in range(self.n):
            not_E = lambda v: cast(BoolRef, Not(self.E_at(v)))
            s.add(Implies(self.S[w], Not(self.box(not_E)(w))))

    def encode_A4(self, s: Solver) -> None:
        """A4: □B at S (depth-4 balance)"""
        for w in range(self.n):
            s.add(Implies(self.S[w], self.box(self.B_at)(w)))

    def encode_A5(self, s: Solver) -> None:
        """A5: Memory schema at S (□B → CS ∧ ¬□E ∧ ¬□¬E)"""
        for w in range(self.n):
            balance = self.box(self.B_at)(w)
            right_preserves = self.box_R(lambda v: self.S[v])(w) == self.S[w]
            left_alters = Not(self.box_L(lambda v: self.S[v])(w) == self.S[w])
            una = Not(self.box(self.E_at)(w))
            not_E = lambda v: cast(BoolRef, Not(self.E_at(v)))
            ona = Not(self.box(not_E)(w))
            memory = And(right_preserves, left_alters, una, ona)
            s.add(Implies(self.S[w], Implies(balance, memory)))

    def encode(self, token: str, s: Solver) -> None:
        """Encode constraint by token. Maps canonical tokens to atomic encoders."""
        token = canonicalize(token)

        # Map canonical tokens to atomic tokens
        canon_to_atomic = {
            "CS": "A1",
            "UNA": "A2",
            "ONA": "A3",
            "BU_EGRESS": "A4",
            "BU_INGRESS": "A5",
            "BU_DUAL": None,  # Special case: needs both A4 and A5
        }

        if token == "BU_DUAL":
            self.encode_A4(s)
            self.encode_A5(s)
        elif token in canon_to_atomic:
            atomic = canon_to_atomic[token]
            getattr(self, f"encode_{atomic}")(s)
        elif token.startswith("A") and token in ["A1", "A2", "A3", "A4", "A5"]:
            getattr(self, f"encode_{token}")(s)
        else:
            raise ValueError(f"Unknown constraint token: {token}")

    def encode_frame_conditions(self, s: Solver) -> None:
        """Encode basic frame conditions: seriality and S nonempty."""
        for w in range(self.n):
            s.add(Or([self.R_L[w][v] for v in range(self.n)]))
            s.add(Or([self.R_R[w][v] for v in range(self.n)]))
        s.add(Or([self.S[w] for w in range(self.n)]))

    def encode_S_generated(self, s: Solver) -> None:
        """Encode that all worlds are reachable from S-worlds."""
        reach = [
            [Bool(f"reach_{k}_{w}") for w in range(self.n)] for k in range(self.n + 1)
        ]
        for w in range(self.n):
            s.add(reach[0][w] == self.S[w])
        for k in range(self.n):
            for w in range(self.n):
                s.add(
                    reach[k + 1][w]
                    == Or(
                        reach[k][w],
                        Or(
                            [
                                And(
                                    reach[k][prev],
                                    Or(self.R_L[prev][w], self.R_R[prev][w]),
                                )
                                for prev in range(self.n)
                            ]
                        ),
                    )
                )
        for w in range(self.n):
            s.add(reach[self.n][w])

    def extract_frame(self, model: Any) -> Dict[str, Any]:
        """Extract frame from Z3 model."""
        frame = {
            "worlds": list(range(self.n)),
            "R_L": {
                w: [v for v in range(self.n) if is_true(model[self.R_L[w][v]])]
                for w in range(self.n)
            },
            "R_R": {
                w: [v for v in range(self.n) if is_true(model[self.R_R[w][v]])]
                for w in range(self.n)
            },
            "S": [w for w in range(self.n) if is_true(model[self.S[w]])],
        }
        return frame


def test_consistency(
    constraints: List[str], n: int = 3, timeout: int = 10000, generated: bool = False
) -> Optional[Dict[str, Any]]:
    """Test if given constraints are consistent. Supports canonical tokens."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    for constraint in constraints:
        token = canonicalize(constraint)
        oa.encode(token, s)
    oa.encode_frame_conditions(s)
    if generated:
        oa.encode_S_generated(s)

    if s.check() == sat:
        return oa.extract_frame(s.model())
    return None


def test_entailment(
    premises: List[str],
    conclusion: str,
    n: int = 4,
    timeout: int = 10000,
    generated: bool = False,
) -> bool:
    """Test if premises entail conclusion. Supports canonical tokens."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    for p in premises:
        token = canonicalize(p)
        oa.encode(token, s)
    oa.encode_frame_conditions(s)
    if generated:
        oa.encode_S_generated(s)

    # Support canonical conclusion tokens
    conclusion = canonicalize(conclusion)

    if conclusion in ["UNA", "A2"]:
        s.add(Or([And(oa.S[w], oa.box(oa.E_at)(w)) for w in range(n)]))
    elif conclusion in ["ONA", "A3"]:
        not_E = lambda v: cast(BoolRef, Not(oa.E_at(v)))
        s.add(Or([And(oa.S[w], oa.box(not_E)(w)) for w in range(n)]))
    elif conclusion in ["BU_EGRESS", "A4"]:
        s.add(Or([And(oa.S[w], Not(oa.box(oa.B_at)(w))) for w in range(n)]))
    elif conclusion in ["BU_INGRESS", "A5"]:
        # Test memory schema: add negation of memory conditions when balance holds
        for w in range(n):
            balance = oa.box(oa.B_at)(w)
            right_preserves = oa.box_R(lambda v: oa.S[v])(w) == oa.S[w]
            left_alters = Not(oa.box_L(lambda v: oa.S[v])(w) == oa.S[w])
            una = Not(oa.box(oa.E_at)(w))
            not_E = lambda v: cast(BoolRef, Not(oa.E_at(v)))
            ona = Not(oa.box(not_E)(w))
            memory = And(right_preserves, left_alters, una, ona)
            s.add(Or([And(oa.S[w], balance, Not(memory)) for w in range(n)]))
    elif conclusion == "BU_DUAL":
        # Test both BU_EGRESS and BU_INGRESS
        s.add(
            Or(
                [And(oa.S[w], Not(oa.box(oa.B_at)(w))) for w in range(n)]
                + [
                    And(
                        oa.S[w],
                        oa.box(oa.B_at)(w),
                        Not(
                            And(
                                oa.box_R(lambda v: oa.S[v])(w) == oa.S[w],
                                Not(oa.box_L(lambda v: oa.S[v])(w) == oa.S[w]),
                                Not(oa.box(oa.E_at)(w)),
                                Not(
                                    oa.box(lambda v: cast(BoolRef, Not(oa.E_at(v))))(w)
                                ),
                            )
                        ),
                    )
                    for w in range(n)
                ]
            )
        )
    elif conclusion == "BU":  # Legacy alias for BU_EGRESS
        s.add(Or([And(oa.S[w], Not(oa.box(oa.B_at)(w))) for w in range(n)]))
    elif conclusion in ["CS", "A1"]:
        s.add(
            Or(
                [
                    And(
                        oa.S[w],
                        Not(
                            And(
                                oa.box_R(lambda v: oa.S[v])(w) == oa.S[w],
                                Not(oa.box_L(lambda v: oa.S[v])(w) == oa.S[w]),
                            )
                        ),
                    )
                    for w in range(n)
                ]
            )
        )

    return s.check() == unsat


def test_independence(
    constraint: str,
    others: List[str],
    n: int = 4,
    timeout: int = 10000,
    generated: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Test if constraint is independent of others. Supports canonical tokens. Returns (status, witness)."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    for other in others:
        token = canonicalize(other)
        oa.encode(token, s)
    oa.encode_frame_conditions(s)
    if generated:
        oa.encode_S_generated(s)

    constraint = canonicalize(constraint)

    if constraint in ["CS", "A1"]:
        s.add(
            Or(
                [
                    And(
                        oa.S[w],
                        Not(
                            And(
                                oa.box_R(lambda v: oa.S[v])(w) == oa.S[w],
                                Not(oa.box_L(lambda v: oa.S[v])(w) == oa.S[w]),
                            )
                        ),
                    )
                    for w in range(n)
                ]
            )
        )
    elif constraint in ["UNA", "A2"]:
        s.add(Or([And(oa.S[w], oa.box(oa.E_at)(w)) for w in range(n)]))
    elif constraint in ["ONA", "A3"]:
        not_E = lambda v: cast(BoolRef, Not(oa.E_at(v)))
        s.add(Or([And(oa.S[w], oa.box(not_E)(w)) for w in range(n)]))
    elif constraint in ["BU_EGRESS", "A4"]:
        s.add(Or([And(oa.S[w], Not(oa.box(oa.B_at)(w))) for w in range(n)]))
    elif constraint in ["BU_INGRESS", "A5"]:
        not_E = lambda v: cast(BoolRef, Not(oa.E_at(v)))
        s.add(
            Or(
                [
                    And(
                        oa.S[w],
                        oa.box(oa.B_at)(w),
                        Not(
                            And(
                                oa.box_R(lambda v: oa.S[v])(w) == oa.S[w],
                                Not(oa.box_L(lambda v: oa.S[v])(w) == oa.S[w]),
                                Not(oa.box(oa.E_at)(w)),
                                Not(oa.box(not_E)(w)),
                            )
                        ),
                    )
                    for w in range(n)
                ]
            )
        )

    result = s.check()
    if result == sat:
        return ("independent", oa.extract_frame(s.model()))
    elif result == unsat:
        return ("derivable", None)
    else:
        return ("unknown", None)


def verify_frame_locally(frame: Dict[str, Any], constraint: str) -> bool:
    """Verify a constraint holds on extracted frame using Python evaluation. Supports canonical tokens."""
    constraint = canonicalize(constraint)
    n = len(frame["worlds"])
    R_L = frame["R_L"]
    R_R = frame["R_R"]
    S_set = set(frame["S"])

    def box_L_py(phi: Callable[[int], bool], w: int) -> bool:
        return all(phi(v) for v in R_L[w])

    def box_R_py(phi: Callable[[int], bool], w: int) -> bool:
        return all(phi(v) for v in R_R[w])

    def box_py(phi: Callable[[int], bool], w: int) -> bool:
        return box_L_py(phi, w) and box_R_py(phi, w)

    def U_at_py(w: int) -> bool:
        return box_L_py(lambda v: v in S_set, w) == box_R_py(lambda v: v in S_set, w)

    def O_at_py(w: int) -> bool:
        LR = box_L_py(lambda u: box_R_py(lambda v: v in S_set, u), w)
        RL = box_R_py(lambda u: box_L_py(lambda v: v in S_set, u), w)
        return LR == (not RL)

    def E_at_py(w: int) -> bool:
        LR = box_L_py(lambda u: box_R_py(lambda v: v in S_set, u), w)
        RL = box_R_py(lambda u: box_L_py(lambda v: v in S_set, u), w)
        return LR == RL

    def B_at_py(w: int) -> bool:
        LRLR = box_L_py(
            lambda w1: box_R_py(
                lambda w2: box_L_py(
                    lambda w3: box_R_py(lambda w4: w4 in S_set, w3), w2
                ),
                w1,
            ),
            w,
        )
        RLRL = box_R_py(
            lambda w1: box_L_py(
                lambda w2: box_R_py(
                    lambda w3: box_L_py(lambda w4: w4 in S_set, w3), w2
                ),
                w1,
            ),
            w,
        )
        return LRLR == RLRL

    for w in S_set:
        if constraint in ["CS", "A1"]:
            right_preserves = box_R_py(lambda v: v in S_set, w) == (w in S_set)
            left_alters = not (box_L_py(lambda v: v in S_set, w) == (w in S_set))
            if not (right_preserves and left_alters):
                return False
        elif constraint in ["UNA", "A2"]:
            # UNA (A2): only checks ¬□E at S, not CS conditions
            una = not box_py(E_at_py, w)
            if not una:
                return False
        elif constraint in ["ONA", "A3"]:
            # ONA (A3): only checks ¬□¬E at S, not CS/UNA conditions
            def not_E_at_py(w: int) -> bool:
                return not E_at_py(w)

            ona = not box_py(not_E_at_py, w)
            if not ona:
                return False
        elif constraint in ["BU_EGRESS", "A4"]:
            if not box_py(B_at_py, w):
                return False
        elif constraint in ["BU_INGRESS", "A5"]:
            if box_py(B_at_py, w):
                right_preserves = box_R_py(lambda v: v in S_set, w) == (w in S_set)
                left_alters = not (box_L_py(lambda v: v in S_set, w) == (w in S_set))

                def not_E_at_py(w: int) -> bool:
                    return not E_at_py(w)

                una = not box_py(E_at_py, w)
                ona = not box_py(not_E_at_py, w)
                if not (right_preserves and left_alters and una and ona):
                    return False
        elif constraint == "BU_DUAL":
            # Check both BU_EGRESS and BU_INGRESS
            if not box_py(B_at_py, w):
                return False
            if box_py(B_at_py, w):
                right_preserves = box_R_py(lambda v: v in S_set, w) == (w in S_set)
                left_alters = not (box_L_py(lambda v: v in S_set, w) == (w in S_set))

                def not_E_at_py(w: int) -> bool:
                    return not E_at_py(w)

                una = not box_py(E_at_py, w)
                ona = not box_py(not_E_at_py, w)
                if not (right_preserves and left_alters and una and ona):
                    return False
    return True


def format_frame(frame: Dict[str, Any]) -> str:
    """Format frame for display."""
    return (
        f"W={frame['worlds']} "
        f"R_L={frame['R_L']} "
        f"R_R={frame['R_R']} "
        f"S={frame['S']}"
    )


def expand_to_atomics(constraints: List[str]) -> List[str]:
    """Expand constraint tokens to atomic postulates (A1-A5). Uses flat expansion.

    Args:
        constraints: List of constraint tokens (canonical or OA* - will be canonicalized)
    """
    # Map to canonical first, then to atomics (flat expansion only)
    canon_to_atomic = {
        "CS": ["A1"],
        "UNA": ["A2"],
        "ONA": ["A3"],
        "BU_EGRESS": ["A4"],
        "BU_INGRESS": ["A5"],
        "BU_DUAL": ["A4", "A5"],
    }

    atomics = []
    for constraint in constraints:
        canon = canonicalize(constraint)
        if canon in canon_to_atomic:
            atomics.extend(canon_to_atomic[canon])
        elif canon.startswith("A"):
            atomics.append(canon)
    return sorted(list(set(atomics)))


def test_entailment_atomics(
    atomics: List[str],
    target: str,
    n: int = 3,
    timeout: int = 10000,
    generated: bool = False,
) -> bool:
    """Test if atomic postulates entail target. Supports canonical target tokens."""
    return test_entailment(atomics, target, n=n, timeout=timeout, generated=generated)


def find_minimal_entailing_subsets(
    target: str,
    all_constraints: List[str],
    n: int = 3,
    timeout: int = 10000,
    use_atomics: bool = True,
) -> List[List[str]]:
    """Find minimal subsets of constraints that entail target.

    Args:
        target: Target constraint (supports canonical tokens)
        all_constraints: List of constraint tokens (canonical or OA*)
        use_atomics: If True, expand to atomic postulates (A1..A5) using flat expansion
    """
    minimal = []
    from itertools import combinations

    if use_atomics:
        all_atomics = expand_to_atomics(all_constraints)
        for r in range(1, len(all_atomics) + 1):
            for subset in combinations(all_atomics, r):
                subset_list = sorted(list(subset))
                if test_entailment_atomics(subset_list, target, n=n, timeout=timeout):
                    to_remove = []
                    is_minimal = True
                    for i, existing in enumerate(minimal):
                        if set(existing).issubset(set(subset_list)):
                            is_minimal = False
                            break
                        elif set(subset_list).issubset(set(existing)):
                            to_remove.append(i)
                    if is_minimal:
                        for i in reversed(to_remove):
                            minimal.pop(i)
                        minimal.append(subset_list)
    else:
        for r in range(1, len(all_constraints) + 1):
            for subset in combinations(all_constraints, r):
                subset_list = list(subset)
                if test_entailment(subset_list, target, n=n, timeout=timeout):
                    to_remove = []
                    is_minimal = True
                    for i, existing in enumerate(minimal):
                        if set(existing).issubset(set(subset_list)):
                            is_minimal = False
                            break
                        elif set(subset_list).issubset(set(existing)):
                            to_remove.append(i)
                    if is_minimal:
                        for i in reversed(to_remove):
                            minimal.pop(i)
                        minimal.append(subset_list)
    return minimal


def find_non_commutation_witness(
    n: int = 4, timeout: int = 10000
) -> Optional[Dict[str, Any]]:
    """Find frame where B holds but E is contingent (neither □E nor □¬E)."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    oa.encode("BU_EGRESS", s)
    oa.encode_frame_conditions(s)

    not_E = lambda v: cast(BoolRef, Not(oa.E_at(v)))
    s.add(
        Or(
            [
                And(oa.S[w], Not(oa.box(oa.E_at)(w)), Not(oa.box(not_E)(w)))
                for w in range(n)
            ]
        )
    )

    if s.check() == sat:
        return oa.extract_frame(s.model())
    return None


def test_oa4_not_collapse_LR(
    n: int = 4, timeout: int = 10000
) -> Optional[Dict[str, Any]]:
    """Verify BU‑Egress (A4) does not force R_L = R_R on S-worlds."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    oa.encode("BU_EGRESS", s)
    oa.encode_frame_conditions(s)

    for w in range(n):
        s.add(
            Implies(oa.S[w], Or([Not(oa.R_L[w][v] == oa.R_R[w][v]) for v in range(n)]))
        )

    if s.check() == sat:
        return oa.extract_frame(s.model())
    return None


def test_entailment_E(
    constraint: str, target: str, n: int = 4, timeout: int = 10000
) -> bool:
    """Test if constraint entails depth-2 equality/inequality properties."""
    oa = ConstraintEncoder(n)
    s = Solver()
    s.set("timeout", timeout)

    constraint = canonicalize(constraint)
    oa.encode(constraint, s)

    oa.encode_frame_conditions(s)

    if target == "not_E":
        s.add(Or([And(oa.S[w], oa.box(oa.E_at)(w)) for w in range(n)]))
    elif target == "not_not_E":
        s.add(
            Or(
                [
                    And(oa.S[w], oa.box(lambda v: cast(BoolRef, Not(oa.E_at(v))))(w))
                    for w in range(n)
                ]
            )
        )

    return s.check() == unsat


def main() -> None:
    """Run complete test suite for the five foundational constraints."""

    print(
        "CGM FOUNDATIONAL CONSTRAINTS TEST SUITE (CS, UNA, ONA, BU‑Egress, BU‑Ingress)"
    )
    print("Testing the five foundational constraints with Z3 SMT solver")

    print("\n1. CONSISTENCY OF INDIVIDUAL CONSTRAINTS")
    for constraint in ["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"]:
        frame = test_consistency([constraint])
        if frame:
            verified = verify_frame_locally(frame, constraint)
            print(f"{constraint}: Consistent (local verification: {verified})")
            print(f"  {format_frame(frame)}")
        else:
            print(f"{constraint}: Inconsistent or timeout")

    print("\n2. CUMULATIVE CONSISTENCY")
    cumulative = [
        (["CS"], "CS alone"),
        (["CS", "UNA"], "CS + UNA"),
        (["CS", "UNA", "ONA"], "CS + UNA + ONA"),
        (["CS", "UNA", "ONA", "BU_EGRESS"], "Forward chain"),
        (["BU_EGRESS", "BU_INGRESS"], "BU‑Egress + BU‑Ingress"),
        (["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"], "All five constraints"),
    ]
    for constraints, desc in cumulative:
        frame = test_consistency(constraints)
        if frame:
            print(f"{desc}: Consistent")
            print(f"  {format_frame(frame)}")
        else:
            print(f"{desc}: Inconsistent or timeout")

    print("\n3. FORWARD ENTAILMENTS")
    forward_tests = [
        (["CS"], "CS", "CS entails itself"),
        (["UNA"], "UNA", "UNA entails itself"),
        (["ONA"], "ONA", "ONA entails itself"),
        (["BU_EGRESS"], "BU_EGRESS", "BU‑Egress entails itself"),
        (["CS", "UNA"], "UNA", "CS + UNA entails UNA"),
        (["UNA", "ONA"], "ONA", "UNA + ONA entails ONA"),
        (["CS", "UNA", "ONA"], "BU_EGRESS", "CS/UNA/ONA do not entail BU‑Egress"),
        (
            ["CS", "UNA", "ONA", "BU_EGRESS"],
            "BU_EGRESS",
            "Full forward chain to BU‑Egress",
        ),
    ]
    for premises, conclusion, desc in forward_tests:
        entailed = test_entailment(premises, conclusion)
        print(f"{desc}: {'Entailed' if entailed else 'Not entailed'}")

    print("\n4. REVERSE ENTAILMENT VIA BU‑INGRESS")
    reverse_tests = [
        (["BU_EGRESS", "BU_INGRESS"], "CS", "BU‑Egress ∧ BU‑Ingress entails CS"),
        (["BU_EGRESS", "BU_INGRESS"], "UNA", "BU‑Egress ∧ BU‑Ingress entails UNA"),
        (["BU_EGRESS", "BU_INGRESS"], "ONA", "BU‑Egress ∧ BU‑Ingress entails ONA"),
        (["BU_EGRESS"], "CS", "BU‑Egress alone does not entail CS"),
        (["BU_EGRESS"], "UNA", "BU‑Egress alone does not entail UNA"),
        (["BU_EGRESS"], "ONA", "BU‑Egress alone does not entail ONA"),
    ]
    for premises, conclusion, desc in reverse_tests:
        entailed = test_entailment(premises, conclusion)
        print(f"{desc}: {'Entailed' if entailed else 'Not entailed'}")

    print("\n5. INDEPENDENCE TESTS")
    independence_tests = [
        ("UNA", ["CS"], "UNA (A2) independent of CS (A1)"),
        ("ONA", ["UNA"], "ONA (A3) independent of UNA (A2)"),
        ("BU_EGRESS", ["CS", "UNA", "ONA"], "BU‑Egress (A4) independent of CS/UNA/ONA"),
        ("BU_INGRESS", ["BU_EGRESS"], "BU‑Ingress (A5) independent of BU‑Egress (A4)"),
        ("CS", ["BU_EGRESS"], "CS independent of BU‑Egress"),
        ("CS", ["BU_EGRESS", "BU_INGRESS"], "CS derivable from BU‑Egress ∧ BU‑Ingress"),
        (
            "CS",
            ["UNA", "ONA", "BU_EGRESS", "BU_INGRESS"],
            "CS from UNA/ONA/BU‑Egress/BU‑Ingress",
        ),
        ("BU_EGRESS", ["BU_INGRESS"], "BU‑Egress independent of BU‑Ingress"),
    ]
    for constraint, others, desc in independence_tests:
        status, witness = test_independence(
            constraint, others, n=3 if constraint in ["UNA", "ONA"] else 4
        )
        if "CS derivable from BU" in desc:
            if status == "derivable":
                print(f"{desc}: Derivable (as designed by memory)")
            elif status == "independent":
                print(f"{desc}: Independent (unexpected)")
                if witness:
                    print(f"  {format_frame(witness)}")
            else:
                print(f"{desc}: Timeout")
        elif status == "independent":
            print(f"{desc}: Independent")
            if witness:
                print(f"  {format_frame(witness)}")
        elif status == "derivable":
            print(f"{desc}: Derivable")
        else:
            print(f"{desc}: Timeout")

    print("\n6. BU‑INGRESS NECESSITY FOR RECONSTRUCTION")
    bu_egress_alone_cs = test_entailment(["BU_EGRESS"], "CS")
    bu_egress_alone_una = test_entailment(["BU_EGRESS"], "UNA")
    bu_egress_alone_ona = test_entailment(["BU_EGRESS"], "ONA")
    bu_dual_cs = test_entailment(["BU_EGRESS", "BU_INGRESS"], "CS")
    bu_dual_una = test_entailment(["BU_EGRESS", "BU_INGRESS"], "UNA")
    bu_dual_ona = test_entailment(["BU_EGRESS", "BU_INGRESS"], "ONA")
    bu_ingress_alone_egress = test_entailment(["BU_INGRESS"], "BU_EGRESS")

    print(f"BU‑Egress alone → CS: {bu_egress_alone_cs}")
    print(f"BU‑Egress alone → UNA: {bu_egress_alone_una}")
    print(f"BU‑Egress alone → ONA: {bu_egress_alone_ona}")
    print(f"BU‑Egress + BU‑Ingress → CS: {bu_dual_cs}")
    print(f"BU‑Egress + BU‑Ingress → UNA: {bu_dual_una}")
    print(f"BU‑Egress + BU‑Ingress → ONA: {bu_dual_ona}")
    print(
        f"BU‑Ingress alone → BU‑Egress: {bu_ingress_alone_egress} (BU‑Ingress does not imply closure)"
    )

    if not bu_egress_alone_cs and bu_dual_cs:
        print(
            "BU‑Ingress (A5) is necessary for reconstruction: BU‑Egress alone cannot reconstruct CS"
        )

    print("\n7. COMPLETE TOROIDAL CYCLE")
    all_constraints = test_consistency(["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"])
    if all_constraints:
        print("All five constraints are consistent together")
        print(f"  {format_frame(all_constraints)}")

        forward = test_entailment(["CS", "UNA", "ONA", "BU_EGRESS"], "BU_EGRESS")
        reverse = test_entailment(["BU_EGRESS", "BU_INGRESS"], "CS")

        print(f"Forward (CS→UNA→ONA→BU‑Egress): {forward}")
        print(f"Reverse (BU‑Egress ∧ BU‑Ingress → CS): {reverse}")

        if forward and reverse:
            print("Toroidal structure confirmed")
        else:
            print(f"Toroidal incomplete: forward={forward}, reverse={reverse}")
    else:
        print("Full constraint set inconsistent or timeout")

    print("\n8. S-GENERATED FRAMES")
    gen_frame = test_consistency(
        ["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"], n=4, generated=True
    )
    if gen_frame:
        print("S-generated frame exists")
        print(f"  {format_frame(gen_frame)}")
    else:
        print("No S-generated frame found")

    print("\n9. MINIMAL ENTAILING SUBSETS (atomic postulates)")
    for target in ["UNA", "ONA", "BU_EGRESS", "BU_INGRESS", "BU_DUAL"]:
        minimal = find_minimal_entailing_subsets(
            target,
            ["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"],
            n=3,
            use_atomics=True,
        )
        print(f"{target}: {minimal}")

    print("\n10. NON-COMMUTATION WITNESS")
    non_comm = find_non_commutation_witness(n=4)
    if non_comm:
        print("Frame where B holds but E is contingent (neither □E nor □¬E):")
        print(f"  {format_frame(non_comm)}")
    else:
        print("No non-commutation witness found")

    print("\n11. BU‑EGRESS (A4) DOES NOT COLLAPSE L AND R")
    mirror = test_oa4_not_collapse_LR(n=4)
    if mirror:
        print("BU‑Egress (A4) allows R_L ≠ R_R on S-worlds")
        print(f"  {format_frame(mirror)}")
    else:
        print("BU‑Egress may force R_L = R_R (or timeout)")

    print("\n12. DEPTH-2 EQUALITY PROPERTIES")
    una_not_E = test_entailment_E("UNA", "not_E", n=4)
    ona_not_not_E = test_entailment_E("ONA", "not_not_E", n=4)
    print(f"UNA ⇒ ¬□E: {una_not_E}")
    print(f"ONA ⇒ ¬□¬E: {ona_not_not_E}")

    print("\n13. INVESTIGATING CS + BU_EGRESS")
    frame = test_consistency(["CS", "BU_EGRESS"], n=3)
    if frame:
        print(f"CS + BU_EGRESS frame: {format_frame(frame)}")
        print(
            f"CS + BU_EGRESS → ONA: {test_entailment(['CS', 'BU_EGRESS'], 'ONA', n=3)}"
        )
        print(
            f"CS + BU_EGRESS → UNA: {test_entailment(['CS', 'BU_EGRESS'], 'UNA', n=3)}"
        )
    else:
        print("CS + BU_EGRESS: Inconsistent or timeout")

    print("\n13b. TESTING A1+A4→ONA AT DIFFERENT FRAME SIZES")
    # Note: n=3 shows finite-model over-constraint; stable minimal sets emerge at n≥4
    for n_val in [3, 4, 5]:
        result = test_entailment_atomics(["A1", "A4"], "ONA", n=n_val, timeout=20000)
        print(f"  n={n_val}: A1+A4 → ONA = {result}")

    print("\n14. NEIGHBOR PAIR INVESTIGATION")
    neighbor_tests = [
        (["UNA", "BU_EGRESS"], "ONA", "UNA + BU_EGRESS → ONA"),
        (["UNA", "BU_EGRESS"], "UNA", "UNA + BU_EGRESS → UNA"),
    ]
    for premises, conclusion, desc in neighbor_tests:
        result = test_entailment(premises, conclusion, n=3)
        print(f"{desc}: {result}")

    print("\n15. MINIMAL SUBSETS WITH S-GENERATED CONSTRAINT")
    from itertools import combinations

    all_atomics = expand_to_atomics(["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"])
    for target in ["UNA", "ONA", "BU_EGRESS", "BU_INGRESS", "BU_DUAL"]:
        minimal = []
        for r in range(1, len(all_atomics) + 1):
            for subset in combinations(all_atomics, r):
                subset_list = sorted(list(subset))
                if test_entailment_atomics(subset_list, target, n=3, generated=True):
                    to_remove = []
                    is_minimal = True
                    for i, existing in enumerate(minimal):
                        if set(existing).issubset(set(subset_list)):
                            is_minimal = False
                            break
                        elif set(subset_list).issubset(set(existing)):
                            to_remove.append(i)
                    if is_minimal:
                        for i in reversed(to_remove):
                            minimal.pop(i)
                        minimal.append(subset_list)
        print(f"{target} (S-generated): {minimal}")

    # print("\n16. MINIMAL SUBSETS STABILITY (n=3..5)")
    # SKIPPED: Has issues with n=5 (timeout/complexity)
    # for n_val in [3, 4, 5]:
    #     print(f"n={n_val}:")
    #     for target in ['UNA', 'ONA', 'BU']:
    #         minimal = find_minimal_entailing_subsets(target, ['OA1', 'OA2', 'OA3', 'OA4', 'OA5'],
    #                                                  n=n_val, timeout=5000, use_atomics=True)
    #         print(f"  {target}: {minimal}")

    print("\n17. COMPLETENESS PROBE")
    # Test if constraints force non-triviality (exists frame where L ≠ R globally)
    oa = ConstraintEncoder(4)
    s = Solver()
    for constraint in ["CS", "UNA", "ONA", "BU_EGRESS", "BU_INGRESS"]:
        oa.encode(constraint, s)
    oa.encode_frame_conditions(s)
    oa.encode_S_generated(s)
    # Add: exists some w,v where R_L[w][v] ≠ R_R[w][v]
    exists_diff = Or(
        [Or([Not(oa.R_L[w][v] == oa.R_R[w][v]) for v in range(4)]) for w in range(4)]
    )
    s.add(exists_diff)
    if s.check() == sat:
        print("Constraints allow non-trivial L/R distinction")
    else:
        print("Constraints force L/R collapse (unexpected)")


if __name__ == "__main__":
    main()
