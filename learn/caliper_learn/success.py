"""Composable success predicates: the honest answer to "did the grasp work?".

A manipulation run is scored by a PREDICATE over world state, not by a reward
threshold and not by a human watching the viewer. This module ships the two
that every pick-and-place task is built from — `Lifted` (the object left the
table) and `PlacedInZone` (the object ended up in the target box) — plus
`AllOf`/`AnyOf` to combine them. The same object is used by `VecSimEnv`
(`success=`, reported per step in `info["success"]`), the eval harness
(`EvalTask.success_predicate` — Wilson-95 over predicate-scored episodes) and
the autopsy verdict, so one definition of success is shared by the demo, the
rollout and the report instead of three that quietly disagree.

Design notes (read before touching):

- EVERY predicate is JSON-describable: `to_dict()` / `from_dict()` round-trip
  exactly, the same contract `randomize.sample` draws carry. That is what lets
  a success criterion be stored beside a scene (the task-artifact phase),
  diffed in review, and cloned per env — `VecSimEnv` clones the caller's
  predicate through this round-trip so N envs never share one predicate's
  captured baseline.

- Predicates are STATEFUL in exactly one place: `Lifted(ref="initial")` has to
  remember the object's height at episode start. `reset(state)` captures it;
  `reset()` clears it and the next call captures instead. Everything else is a
  pure function of the state, and `describe()` never depends on the captured
  value — the plain-English sentence is the same before and after an episode.

- Combinators evaluate EVERY term, never short-circuiting. `AllOf(a, b)` with
  a False `a` still calls `b`, because `b` may be a `Lifted` that has yet to
  capture its baseline: a short-circuit would make the captured reference
  depend on the order the terms happened to fail in, which is exactly the kind
  of silent, seed-dependent scoring bug this module exists to prevent.

- Thresholds are plain `>=` / `<=` comparisons on binary floats, with no
  epsilon anywhere: a decimal boundary is not a boundary a float can sit on
  (0.15 - 0.10 is 0.049999999999999996, so a 0.05 lift from 0.10 to 0.15 does
  NOT clear a 0.05 bar). State the threshold you actually mean rather than the
  knife edge; no tolerance is invented on your behalf.

- `SuccessState` addresses props by NAME, not by index. `state_from_mujoco`
  builds one from any MuJoCo model/data pair by finding the FREE bodies (a
  free-floating body IS a prop), keyed by the body name and — for the
  `prop_<name>` bodies caliper's MJCF exporter emits — also by the bare name,
  mirroring `caliper-sim-mujoco`'s own `prop_<mujoco_name(name)>` resolution.
  So `Lifted("cube")` matches a body named `cube` or `prop_cube`.

Pure python + numpy: no mujoco, no torch, no caliper. `state_from_mujoco` is
the one function that touches mujoco and imports it lazily, matching the
package rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

# Stable `kind` tags — the JSON schema's discriminator, never renamed.
KIND_LIFTED = "lifted"
KIND_PLACED_IN_ZONE = "placed_in_zone"
KIND_ALL_OF = "all_of"
KIND_ANY_OF = "any_of"

_PROP_PREFIX = "prop_"  # caliper's MJCF exporter body-name prefix for props


# ----- the state a predicate judges -------------------------------------------


@dataclass(frozen=True)
class SuccessState:
    """One world snapshot, as much of it as a predicate can ask about.

    `prop_pos` maps a prop name to its world-space center `(x, y, z)` in
    meters; `prop_vel` (optional) maps the same names to world-space LINEAR
    velocity in m/s — needed only by the settled variant of `PlacedInZone`.
    `tip_pos` is the robot tip's world position when the caller knows it;
    `VecSimEnv` leaves it None (no shipped predicate needs it, and running FK
    for every env on every step to fill a field nobody reads is not free).
    """

    prop_pos: Mapping[str, Sequence[float]]
    prop_vel: Optional[Mapping[str, Sequence[float]]] = None
    tip_pos: Optional[Sequence[float]] = None

    def pos(self, prop: str) -> Sequence[float]:
        """World center of `prop`; `ValueError` naming what IS there if absent."""
        try:
            return self.prop_pos[prop]
        except KeyError:
            raise ValueError(
                f"no prop named {prop!r} in this state — known props: "
                f"{sorted(self.prop_pos)}. Props are the scene's FREE bodies; "
                "caliper's MJCF exporter names them `prop_<name>` (both spellings "
                "resolve). Check the scene actually contains the object."
            ) from None

    def vel(self, prop: str) -> Sequence[float]:
        """World linear velocity of `prop`; `ValueError` when the state carries
        no velocities at all (a settled check on a velocity-less state cannot
        be answered, and answering it anyway would fabricate a success)."""
        if self.prop_vel is None:
            raise ValueError(
                f"this state carries no velocities, so 'settled' cannot be checked "
                f"for {prop!r} — build the state with prop_vel=... (VecSimEnv and "
                "state_from_mujoco always do) or drop settled_speed from the predicate"
            )
        try:
            return self.prop_vel[prop]
        except KeyError:
            raise ValueError(
                f"no velocity for prop {prop!r} — known: {sorted(self.prop_vel)}"
            ) from None


# ----- zones -------------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    """An axis-aligned box: `center` (x, y, z) and `half` extents, meters.

    Membership is INCLUSIVE on every face — a point exactly on the boundary is
    inside. Half extents must be finite and non-negative (a zero extent is a
    degenerate but legal plane/line/point constraint).
    """

    center: tuple[float, float, float]
    half: tuple[float, float, float]

    def __post_init__(self):
        for name in ("center", "half"):
            v = getattr(self, name)
            if len(v) != 3 or not all(math.isfinite(float(x)) for x in v):
                raise ValueError(f"Zone.{name} must be 3 finite numbers, got {v!r}")
            object.__setattr__(self, name, tuple(float(x) for x in v))
        if any(h < 0.0 for h in self.half):
            raise ValueError(f"Zone.half must be non-negative, got {self.half}")

    def contains(self, point: Sequence[float]) -> bool:
        """Is `point` inside the box (inclusive on every face)?"""
        if len(point) != 3:
            raise ValueError(f"Zone.contains needs a 3-vector, got {len(point)} values")
        return all(
            abs(float(p) - c) <= h for p, c, h in zip(point, self.center, self.half)
        )

    def to_dict(self) -> dict:
        return {"center": list(self.center), "half": list(self.half)}

    @staticmethod
    def from_dict(d: Mapping) -> "Zone":
        _check_keys(d, {"center", "half"}, "zone")
        for k in ("center", "half"):
            if k not in d:
                raise ValueError(f"zone is missing required key '{k}'")
        return Zone(center=tuple(d["center"]), half=tuple(d["half"]))

    def describe(self) -> str:
        sx, sy, sz = (2.0 * h for h in self.half)
        cx, cy, cz = self.center
        return (
            f"the {sx:.3f} x {sy:.3f} x {sz:.3f} m box centered at "
            f"({cx:.3f}, {cy:.3f}, {cz:.3f})"
        )


# ----- the predicate protocol ---------------------------------------------------


class SuccessPredicate:
    """Base class: `name`, `describe()`, `reset(state=None)`, `__call__(state)
    -> bool`, `to_dict()`.

    Subclasses are dataclasses so that equality and `repr` come for free and
    any captured baseline is declared `compare=False` — two predicates are the
    same predicate when their SPECIFICATION matches, regardless of what
    episode they are in the middle of.
    """

    @property
    def name(self) -> str:
        """Short stable identifier, e.g. `lifted:cube` — for log lines and
        `info` keys. `describe()` is the sentence a human reads."""
        raise NotImplementedError

    def describe(self) -> str:
        """One plain-English sentence stating what counts as success. Goes
        verbatim into the eval report and the autopsy verdict, so it reads as
        a claim: 'cube rises 0.050 m above its initial height'."""
        raise NotImplementedError

    def reset(self, state: Optional[SuccessState] = None) -> None:
        """Start a new episode. With `state`, capture any initial reference
        NOW; without it, clear the reference so the next call captures."""

    def __call__(self, state: SuccessState) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        """JSON-describable spec, `{"kind": ..., ...}` — round-trips exactly
        through `from_dict`."""
        raise NotImplementedError


@dataclass
class Lifted(SuccessPredicate):
    """The object left the surface.

    `ref="initial"` (default): success when the prop's center is `height`
    meters ABOVE where it was at the last `reset` — the honest lift test,
    independent of where the table is. `ref="absolute"`: success when the
    center's world z reaches `height`, for scenes whose heights are known.

    With `ref="initial"` and no reset, the reference is captured on the FIRST
    call, so the predicate is usable standalone over a stream of states.
    """

    prop: str
    height: float
    ref: str = "initial"
    _z0: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        _check_prop(self.prop)
        self.height = float(self.height)
        if not math.isfinite(self.height):
            raise ValueError(f"Lifted.height must be finite, got {self.height}")
        if self.ref not in ("initial", "absolute"):
            raise ValueError(
                f"Lifted.ref must be 'initial' or 'absolute', got {self.ref!r}"
            )
        if self.ref == "initial" and self.height <= 0.0:
            raise ValueError(
                f"Lifted.height must be > 0 for ref='initial' (a lift of "
                f"{self.height} m is not a lift); use ref='absolute' for a world-z "
                "threshold that can sit anywhere"
            )

    @property
    def name(self) -> str:
        return f"{KIND_LIFTED}:{self.prop}"

    def describe(self) -> str:
        if self.ref == "absolute":
            return f"{self.prop}'s center reaches z >= {self.height:.3f} m"
        return f"{self.prop} rises {self.height:.3f} m above its initial height"

    def reset(self, state: Optional[SuccessState] = None) -> None:
        self._z0 = None if state is None else float(state.pos(self.prop)[2])

    def __call__(self, state: SuccessState) -> bool:
        z = float(state.pos(self.prop)[2])
        if self.ref == "absolute":
            return z >= self.height
        if self._z0 is None:  # standalone use: first state IS the reference
            self._z0 = z
        return (z - self._z0) >= self.height

    def to_dict(self) -> dict:
        return {
            "kind": KIND_LIFTED,
            "prop": self.prop,
            "height": self.height,
            "ref": self.ref,
        }


@dataclass
class PlacedInZone(SuccessPredicate):
    """The object ended up in the target box.

    Success when the prop's center is inside `zone` (inclusive faces). With
    `settled_speed` set (m/s), it must ALSO be moving slower than that — the
    difference between "placed" and "flew through". A settled check on a state
    that carries no velocities raises rather than passing silently.
    """

    prop: str
    zone: Zone
    settled_speed: Optional[float] = None

    def __post_init__(self):
        _check_prop(self.prop)
        if not isinstance(self.zone, Zone):
            raise TypeError(
                f"PlacedInZone.zone must be a Zone, got {type(self.zone).__name__} "
                "(build one with Zone(center=(x, y, z), half=(hx, hy, hz)))"
            )
        if self.settled_speed is not None:
            self.settled_speed = float(self.settled_speed)
            if not math.isfinite(self.settled_speed) or self.settled_speed < 0.0:
                raise ValueError(
                    f"PlacedInZone.settled_speed must be finite and >= 0, got "
                    f"{self.settled_speed}"
                )

    @property
    def name(self) -> str:
        return f"{KIND_PLACED_IN_ZONE}:{self.prop}"

    def describe(self) -> str:
        s = f"{self.prop}'s center is inside {self.zone.describe()}"
        if self.settled_speed is not None:
            s += f" and has come to rest (speed <= {self.settled_speed:.3f} m/s)"
        return s

    def __call__(self, state: SuccessState) -> bool:
        if not self.zone.contains(state.pos(self.prop)):
            return False
        if self.settled_speed is None:
            return True
        v = state.vel(self.prop)
        speed = math.sqrt(sum(float(x) * float(x) for x in v))
        return speed <= self.settled_speed

    def to_dict(self) -> dict:
        return {
            "kind": KIND_PLACED_IN_ZONE,
            "prop": self.prop,
            "zone": self.zone.to_dict(),
            "settled_speed": self.settled_speed,
        }


@dataclass(init=False)  # variadic terms: hand-written __init__, dataclass eq/repr
class _Combinator(SuccessPredicate):
    """Shared machinery for AllOf/AnyOf (see the module doc: no short-circuit).

    Subclasses supply `_kind` (the JSON tag), `_word` (the English connective
    `describe()` joins terms with) and `_combine` (the boolean fold).
    """

    terms: tuple[SuccessPredicate, ...]

    def __init__(self, *terms):
        flat: list[SuccessPredicate] = []
        for t in terms:
            if isinstance(t, (list, tuple)):
                flat.extend(t)
            else:
                flat.append(t)
        if not flat:
            raise ValueError(f"{type(self).__name__} needs at least one term")
        for t in flat:
            if not isinstance(t, SuccessPredicate):
                raise TypeError(
                    f"{type(self).__name__} terms must be SuccessPredicate, got "
                    f"{type(t).__name__}"
                )
        self.terms = tuple(flat)

    @property
    def name(self) -> str:
        return f"{self._kind}({', '.join(t.name for t in self.terms)})"

    def describe(self) -> str:
        return f" {self._word} ".join(t.describe() for t in self.terms)

    def reset(self, state: Optional[SuccessState] = None) -> None:
        for t in self.terms:
            t.reset(state)

    def __call__(self, state: SuccessState) -> bool:
        # Evaluate EVERY term (see the module doc): a skipped term is a term
        # whose initial reference never got captured.
        results = [bool(t(state)) for t in self.terms]
        return self._combine(results)

    def to_dict(self) -> dict:
        return {"kind": self._kind, "terms": [t.to_dict() for t in self.terms]}


class AllOf(_Combinator):
    """Every term must hold (conjunction) — 'lifted AND placed'."""

    _kind = KIND_ALL_OF
    _word = "and"

    @staticmethod
    def _combine(results: list[bool]) -> bool:
        return all(results)


class AnyOf(_Combinator):
    """At least one term must hold (disjunction) — 'in either bin'."""

    _kind = KIND_ANY_OF
    _word = "or"

    @staticmethod
    def _combine(results: list[bool]) -> bool:
        return any(results)


# ----- (de)serialization ---------------------------------------------------------


def _check_prop(prop: str) -> None:
    if not isinstance(prop, str) or not prop.strip():
        raise ValueError(f"prop name must be a non-empty string, got {prop!r}")


def _check_keys(d: Mapping, allowed: set, what: str) -> None:
    if not isinstance(d, Mapping):
        raise TypeError(f"{what} spec must be a dict, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) in {what} spec: {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )


def from_dict(spec: Mapping) -> SuccessPredicate:
    """Rebuild a predicate from its `to_dict()` form.

    The round-trip is EXACT: `from_dict(p.to_dict()).to_dict() == p.to_dict()`
    for every predicate here — that equality is what makes a stored success
    criterion trustworthy. Unknown kinds and unknown keys raise rather than
    being ignored, so a typo in a task file is a loud failure, not a silently
    weaker success test.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"predicate spec must be a dict, got {type(spec).__name__}")
    kind = spec.get("kind")
    if kind == KIND_LIFTED:
        _check_keys(spec, {"kind", "prop", "height", "ref"}, KIND_LIFTED)
        return Lifted(
            prop=spec["prop"], height=spec["height"], ref=spec.get("ref", "initial")
        )
    if kind == KIND_PLACED_IN_ZONE:
        _check_keys(spec, {"kind", "prop", "zone", "settled_speed"}, KIND_PLACED_IN_ZONE)
        return PlacedInZone(
            prop=spec["prop"],
            zone=Zone.from_dict(spec["zone"]),
            settled_speed=spec.get("settled_speed"),
        )
    if kind in (KIND_ALL_OF, KIND_ANY_OF):
        _check_keys(spec, {"kind", "terms"}, kind)
        terms = [from_dict(t) for t in spec.get("terms", ())]
        return (AllOf if kind == KIND_ALL_OF else AnyOf)(*terms)
    raise ValueError(
        f"unknown success predicate kind {kind!r} — known kinds: "
        f"{[KIND_LIFTED, KIND_PLACED_IN_ZONE, KIND_ALL_OF, KIND_ANY_OF]}"
    )


def as_predicate(spec) -> SuccessPredicate:
    """Normalize what a caller may pass as `success=`: a predicate, a spec
    dict, or a sequence of either. A sequence becomes an `AllOf` — a task with
    several conditions succeeds when ALL of them hold."""
    if isinstance(spec, SuccessPredicate):
        return spec
    if isinstance(spec, Mapping):
        return from_dict(spec)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("success spec sequence is empty")
        terms = [as_predicate(s) for s in spec]
        return terms[0] if len(terms) == 1 else AllOf(*terms)
    raise TypeError(
        f"cannot read a success predicate from {type(spec).__name__} — pass a "
        "SuccessPredicate, its to_dict() form, or a sequence of those"
    )


def clone(pred: SuccessPredicate) -> SuccessPredicate:
    """An independent copy with NO captured baseline, via the JSON round-trip
    (so cloning can never share mutable state — the reason `VecSimEnv` can
    hand one predicate spec to N envs)."""
    return from_dict(pred.to_dict())


# ----- building a state from a live sim --------------------------------------------


def _prop_keys(body_name: str) -> Iterable[str]:
    """The name(s) a prop body answers to: its body name, plus the bare name
    when it carries the exporter's `prop_` prefix."""
    yield body_name
    if body_name.startswith(_PROP_PREFIX) and len(body_name) > len(_PROP_PREFIX):
        yield body_name[len(_PROP_PREFIX) :]


def state_from_mujoco(model, data, *, tip_pos: Optional[Sequence[float]] = None) -> SuccessState:
    """Read every prop's pose out of a live MuJoCo model/data pair.

    A PROP is a free-floating body — exactly the thing caliper's MJCF exporter
    emits for `PropSpec` and the only kind of body a manipulation task moves —
    so this collects the bodies carrying a `free` joint. Each is keyed by its
    body name and, for `prop_<name>` bodies, also by the bare name. Positions
    come from `data.xpos` (the body frame origin = the primitive's center) and
    velocities from the free joint's first three `qvel` entries, which MuJoCo
    defines as world-frame linear velocity.

    `data` must be current (any `mj_step`/`mj_forward` since the last write);
    stale `xpos` would be read as a stale success verdict.
    """
    import mujoco  # lazy: keep this module importable without mujoco

    pos: dict[str, tuple[float, float, float]] = {}
    vel: dict[str, tuple[float, float, float]] = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        bid = int(model.jnt_bodyid[jid])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if body is None:  # unnamed free body: nothing a predicate could address
            continue
        dadr = int(model.jnt_dofadr[jid])
        p = tuple(float(v) for v in data.xpos[bid])
        v = tuple(float(x) for x in data.qvel[dadr : dadr + 3])
        for key in _prop_keys(body):
            if key in pos:
                raise ValueError(
                    f"two prop bodies both answer to the name {key!r} — rename one "
                    "(a body named `x` and a body named `prop_x` collide, because "
                    "`prop_x` also answers to `x`)"
                )
            pos[key], vel[key] = p, v
    return SuccessState(prop_pos=pos, prop_vel=vel, tip_pos=tip_pos)
