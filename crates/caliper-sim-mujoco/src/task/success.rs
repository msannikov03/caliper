//! Success predicates — the Rust half of ONE definition of "did it work?".
//!
//! This is a port, not a parallel invention: the JSON schema, the thresholds
//! and the plain-English `describe()` sentences are byte-for-byte the ones
//! `caliper_learn.success` (the python learning sidecar) reads and writes, so a
//! `*.caliper-task.json` scored live in Studio and the same file scored by an
//! eval rollout cannot disagree. Where the two faces could drift, this module
//! copies python's exact rule and says so in a comment.
//!
//! What is shared:
//! - Kinds `lifted` / `placed_in_zone` / `all_of` / `any_of`, with the same
//!   keys and the same required-ness. Unknown keys and unknown kinds RAISE at
//!   every level ([`Predicate::from_value`]) — a typo must never silently
//!   weaken a success test.
//! - Plain `>=` / `<=` comparisons on binary floats, with NO epsilon anywhere.
//!   A decimal boundary is not a boundary a float can sit on (`0.15 - 0.10` is
//!   `0.049999999999999996`, so lifting from 0.10 to 0.15 does NOT clear a
//!   0.05 bar). State the threshold you mean.
//! - Zone membership INCLUSIVE on every face; a settled check against a state
//!   that carries no velocities is an ERROR, never a pass.
//! - Combinators evaluate their terms IN ORDER and never short-circuit on a
//!   `false`, because a later term may be a [`Predicate::Lifted`] that has yet
//!   to capture its baseline. They do stop at the first `Err`, which is what
//!   python's raise-inside-a-comprehension does.
//!
//! The one structural difference: python keeps a captured baseline INSIDE the
//! predicate object, and Rust cannot hand out a `&self` evaluator that mutates
//! itself. So the baseline lives beside the predicate in a
//! [`SuccessTracker`], keyed by prop name — which is the same value python
//! would capture, since `ref="initial"` always means "the z this prop had at
//! episode start" no matter how many terms ask about it.

use super::TaskError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Stable `kind` tags — the JSON schema's discriminator, never renamed.
pub const KIND_LIFTED: &str = "lifted";
pub const KIND_PLACED_IN_ZONE: &str = "placed_in_zone";
pub const KIND_ALL_OF: &str = "all_of";
pub const KIND_ANY_OF: &str = "any_of";

fn invalid(msg: impl Into<String>) -> TaskError {
    TaskError::Invalid(msg.into())
}

/// Reject unknown keys in one JSON object (python's `_check_keys`).
fn check_keys(
    obj: &serde_json::Map<String, Value>,
    allowed: &[&str],
    what: &str,
) -> Result<(), TaskError> {
    let unknown: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !allowed.contains(k))
        .collect();
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "unknown key(s) in {what} spec: {unknown:?} (allowed: {allowed:?})"
        )));
    }
    Ok(())
}

fn as_object<'a>(
    v: &'a Value,
    what: &str,
) -> Result<&'a serde_json::Map<String, Value>, TaskError> {
    v.as_object().ok_or_else(|| {
        invalid(format!(
            "{what} spec must be an object, got {}",
            json_kind(v)
        ))
    })
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// A required finite number.
fn number(obj: &serde_json::Map<String, Value>, key: &str, what: &str) -> Result<f64, TaskError> {
    let v = obj
        .get(key)
        .ok_or_else(|| invalid(format!("{what} is missing required key `{key}`")))?;
    let n = v.as_f64().ok_or_else(|| {
        invalid(format!(
            "{what} `{key}` must be a number, got {}",
            json_kind(v)
        ))
    })?;
    if !n.is_finite() {
        return Err(invalid(format!("{what} `{key}` must be finite, got {n}")));
    }
    Ok(n)
}

/// A required non-empty prop name (python's `_check_prop`).
fn prop_name(obj: &serde_json::Map<String, Value>, what: &str) -> Result<String, TaskError> {
    let v = obj
        .get("prop")
        .ok_or_else(|| invalid(format!("{what} is missing required key `prop`")))?;
    match v.as_str() {
        Some(s) if !s.trim().is_empty() => Ok(s.to_string()),
        _ => Err(invalid(format!(
            "{what} `prop` must be a non-empty string, got {}",
            json_kind(v)
        ))),
    }
}

/// Three finite numbers.
fn vec3(v: &Value, what: &str) -> Result<[f64; 3], TaskError> {
    let a = v
        .as_array()
        .ok_or_else(|| invalid(format!("{what} must be 3 numbers, got {}", json_kind(v))))?;
    if a.len() != 3 {
        return Err(invalid(format!(
            "{what} must be 3 numbers, got {}",
            a.len()
        )));
    }
    let mut out = [0.0; 3];
    for (i, x) in a.iter().enumerate() {
        let n = x
            .as_f64()
            .ok_or_else(|| invalid(format!("{what} must be 3 numbers, got {}", json_kind(x))))?;
        if !n.is_finite() {
            return Err(invalid(format!("{what} must be finite, got {n}")));
        }
        out[i] = n;
    }
    Ok(out)
}

// ===== zones =====

/// An axis-aligned box: `center` and `half` extents, meters.
///
/// Membership is INCLUSIVE on every face — a point exactly on the boundary is
/// inside. Half extents must be finite and non-negative (a zero extent is a
/// degenerate but legal plane/line/point constraint, exactly as in python).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    pub center: [f64; 3],
    pub half: [f64; 3],
}

impl Zone {
    pub fn new(center: [f64; 3], half: [f64; 3]) -> Result<Self, TaskError> {
        let z = Zone { center, half };
        z.validate()?;
        Ok(z)
    }

    pub fn validate(&self) -> Result<(), TaskError> {
        for (name, v) in [("center", self.center), ("half", self.half)] {
            if !v.iter().all(|x| x.is_finite()) {
                return Err(invalid(format!(
                    "zone {name} must be 3 finite numbers, got {v:?}"
                )));
            }
        }
        if self.half.iter().any(|h| *h < 0.0) {
            return Err(invalid(format!(
                "zone half extents must be non-negative, got {:?}",
                self.half
            )));
        }
        Ok(())
    }

    /// Is `point` inside the box (inclusive on every face)?
    pub fn contains(&self, point: [f64; 3]) -> bool {
        (0..3).all(|i| (point[i] - self.center[i]).abs() <= self.half[i])
    }

    pub fn from_value(v: &Value) -> Result<Self, TaskError> {
        let obj = as_object(v, "zone")?;
        check_keys(obj, &["center", "half"], "zone")?;
        let get = |key: &str| -> Result<[f64; 3], TaskError> {
            let x = obj
                .get(key)
                .ok_or_else(|| invalid(format!("zone is missing required key `{key}`")))?;
            vec3(x, &format!("zone {key}"))
        };
        Zone::new(get("center")?, get("half")?)
    }

    pub fn to_value(&self) -> Value {
        serde_json::json!({ "center": self.center.to_vec(), "half": self.half.to_vec() })
    }

    /// The sentence fragment `PlacedInZone::describe` embeds, worded exactly
    /// like `caliper_learn.success.Zone.describe`.
    pub fn describe(&self) -> String {
        let [cx, cy, cz] = self.center;
        let [hx, hy, hz] = self.half;
        format!(
            "the {:.3} x {:.3} x {:.3} m box centered at ({cx:.3}, {cy:.3}, {cz:.3})",
            2.0 * hx,
            2.0 * hy,
            2.0 * hz
        )
    }
}

/// A zone inside a predicate: either spelled out, or — the task-file
/// convenience — the NAME of a `scene.zones` entry.
///
/// A [`ZoneRef::Named`] exists only between parsing a task file and
/// [`crate::task::TaskSpec`]'s zone resolution: `load_task` never returns one,
/// and evaluating one is a loud error rather than a guess. The python face
/// (`caliper_learn.success.from_dict`) accepts the resolved form only, which is
/// why resolution happens before a predicate is ever handed over.
#[derive(Clone, Debug, PartialEq)]
pub enum ZoneRef {
    Named(String),
    Inline(Zone),
}

impl ZoneRef {
    pub fn zone(&self) -> Result<&Zone, TaskError> {
        match self {
            ZoneRef::Inline(z) => Ok(z),
            ZoneRef::Named(n) => Err(invalid(format!(
                "zone `{n}` was never resolved — a predicate used on its own must \
                 spell its zone out as {{center, half}}; the name form only works \
                 inside a task file that declares that zone in `scene.zones`"
            ))),
        }
    }

    fn from_value(v: &Value) -> Result<Self, TaskError> {
        match v {
            Value::String(s) if !s.trim().is_empty() => Ok(ZoneRef::Named(s.clone())),
            Value::String(_) => Err(invalid("a zone name must be a non-empty string")),
            _ => Ok(ZoneRef::Inline(Zone::from_value(v)?)),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            ZoneRef::Named(n) => Value::String(n.clone()),
            ZoneRef::Inline(z) => z.to_value(),
        }
    }
}

/// Which reference height a [`Predicate::Lifted`] measures against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LiftedRef {
    /// Above where the prop was when the episode started (the honest lift
    /// test, independent of where the table is). Needs a baseline, so it needs
    /// a [`SuccessTracker`].
    #[default]
    Initial,
    /// A world-z threshold, for scenes whose heights are known.
    Absolute,
}

impl LiftedRef {
    fn as_str(self) -> &'static str {
        match self {
            LiftedRef::Initial => "initial",
            LiftedRef::Absolute => "absolute",
        }
    }
}

// ===== the state a predicate judges =====

/// One world snapshot, as much of it as a predicate can ask about.
///
/// Positions are world-space prop CENTERS (meters) by name; velocities are
/// world-space LINEAR velocity (m/s), needed only by the settled variant of
/// [`Predicate::PlacedInZone`]. `tip_pos` is the robot tip when the caller
/// knows it — no shipped predicate reads it yet.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SuccessState {
    prop_pos: BTreeMap<String, [f64; 3]>,
    prop_vel: Option<BTreeMap<String, [f64; 3]>>,
    tip_pos: Option<[f64; 3]>,
}

impl SuccessState {
    /// A velocity-less state: every settled check against it is an ERROR.
    pub fn from_positions<I, S>(positions: I) -> Self
    where
        I: IntoIterator<Item = (S, [f64; 3])>,
        S: Into<String>,
    {
        Self {
            prop_pos: positions.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            prop_vel: None,
            tip_pos: None,
        }
    }

    pub fn with_velocities<I, S>(mut self, velocities: I) -> Self
    where
        I: IntoIterator<Item = (S, [f64; 3])>,
        S: Into<String>,
    {
        self.prop_vel = Some(velocities.into_iter().map(|(k, v)| (k.into(), v)).collect());
        self
    }

    pub fn with_tip(mut self, tip: [f64; 3]) -> Self {
        self.tip_pos = Some(tip);
        self
    }

    pub fn tip_pos(&self) -> Option<[f64; 3]> {
        self.tip_pos
    }

    /// Names this state carries positions for.
    pub fn prop_names(&self) -> Vec<&str> {
        self.prop_pos.keys().map(String::as_str).collect()
    }

    /// World center of `prop`; an error naming what IS there when absent.
    pub fn pos(&self, prop: &str) -> Result<[f64; 3], TaskError> {
        self.prop_pos.get(prop).copied().ok_or_else(|| {
            TaskError::State(format!(
                "no prop named `{prop}` in this state — known props: {:?}. Props are the \
                 scene's free bodies; check the scene actually contains the object.",
                self.prop_names()
            ))
        })
    }

    /// World linear velocity of `prop`. A state with no velocities at all
    /// cannot answer a settled check, and answering it anyway would fabricate
    /// a success — so this is an error, exactly like python's.
    pub fn vel(&self, prop: &str) -> Result<[f64; 3], TaskError> {
        let vels = self.prop_vel.as_ref().ok_or_else(|| {
            TaskError::State(format!(
                "this state carries no velocities, so `settled` cannot be checked for \
                 `{prop}` — build the state with velocities or drop settled_speed from \
                 the predicate"
            ))
        })?;
        vels.get(prop).copied().ok_or_else(|| {
            TaskError::State(format!(
                "no velocity for prop `{prop}` — known: {:?}",
                vels.keys().collect::<Vec<_>>()
            ))
        })
    }
}

// ===== the predicate =====

/// A composable success criterion over world state.
///
/// (De)serialization is hand-written rather than derived, for the same reason
/// python's is: every level checks its keys, so a misspelled `settle_speed`
/// fails loudly instead of quietly dropping the settle requirement.
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// The object left the surface.
    Lifted {
        prop: String,
        height: f64,
        reference: LiftedRef,
    },
    /// The object ended up in the target box (and, with `settled_speed`, has
    /// stopped moving — the difference between "placed" and "flew through").
    PlacedInZone {
        prop: String,
        zone: ZoneRef,
        settled_speed: Option<f64>,
    },
    /// Every term holds.
    AllOf(Vec<Predicate>),
    /// At least one term holds.
    AnyOf(Vec<Predicate>),
}

impl Predicate {
    /// Rebuild a predicate from its JSON form. The round-trip is EXACT:
    /// `from_value(&p.to_value()).to_value() == p.to_value()`.
    pub fn from_value(v: &Value) -> Result<Self, TaskError> {
        let obj = as_object(v, "predicate")?;
        let kind = obj.get("kind").and_then(Value::as_str).ok_or_else(|| {
            invalid(format!(
                "a success predicate needs a string `kind` — known kinds: \
                 [{KIND_LIFTED}, {KIND_PLACED_IN_ZONE}, {KIND_ALL_OF}, {KIND_ANY_OF}]"
            ))
        })?;
        match kind {
            KIND_LIFTED => {
                check_keys(obj, &["kind", "prop", "height", "ref"], KIND_LIFTED)?;
                let reference = match obj.get("ref") {
                    None | Some(Value::Null) => LiftedRef::Initial,
                    Some(Value::String(s)) if s == "initial" => LiftedRef::Initial,
                    Some(Value::String(s)) if s == "absolute" => LiftedRef::Absolute,
                    Some(other) => {
                        return Err(invalid(format!(
                            "lifted `ref` must be \"initial\" or \"absolute\", got {other}"
                        )));
                    }
                };
                let p = Predicate::Lifted {
                    prop: prop_name(obj, KIND_LIFTED)?,
                    height: number(obj, "height", KIND_LIFTED)?,
                    reference,
                };
                p.validate()?;
                Ok(p)
            }
            KIND_PLACED_IN_ZONE => {
                check_keys(
                    obj,
                    &["kind", "prop", "zone", "settled_speed"],
                    KIND_PLACED_IN_ZONE,
                )?;
                let zone = obj.get("zone").ok_or_else(|| {
                    invalid(format!(
                        "{KIND_PLACED_IN_ZONE} is missing required key `zone`"
                    ))
                })?;
                let settled_speed = match obj.get("settled_speed") {
                    None | Some(Value::Null) => None,
                    Some(_) => Some(number(obj, "settled_speed", KIND_PLACED_IN_ZONE)?),
                };
                let p = Predicate::PlacedInZone {
                    prop: prop_name(obj, KIND_PLACED_IN_ZONE)?,
                    zone: ZoneRef::from_value(zone)?,
                    settled_speed,
                };
                p.validate()?;
                Ok(p)
            }
            KIND_ALL_OF | KIND_ANY_OF => {
                check_keys(obj, &["kind", "terms"], kind)?;
                let terms = match obj.get("terms") {
                    None => Vec::new(),
                    Some(Value::Array(a)) => {
                        a.iter()
                            .map(Predicate::from_value)
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    Some(other) => {
                        return Err(invalid(format!(
                            "{kind} `terms` must be an array of predicates, got {}",
                            json_kind(other)
                        )));
                    }
                };
                let p = if kind == KIND_ALL_OF {
                    Predicate::AllOf(terms)
                } else {
                    Predicate::AnyOf(terms)
                };
                p.validate()?;
                Ok(p)
            }
            other => Err(invalid(format!(
                "unknown success predicate kind `{other}` — known kinds: \
                 [{KIND_LIFTED}, {KIND_PLACED_IN_ZONE}, {KIND_ALL_OF}, {KIND_ANY_OF}]"
            ))),
        }
    }

    /// The JSON form, key-for-key what `caliper_learn.success` writes: `ref`
    /// and `settled_speed` are always present (the latter as `null` when
    /// unset), so a stored criterion reads the same from either face.
    pub fn to_value(&self) -> Value {
        match self {
            Predicate::Lifted {
                prop,
                height,
                reference,
            } => serde_json::json!({
                "kind": KIND_LIFTED,
                "prop": prop,
                "height": height,
                "ref": reference.as_str(),
            }),
            Predicate::PlacedInZone {
                prop,
                zone,
                settled_speed,
            } => serde_json::json!({
                "kind": KIND_PLACED_IN_ZONE,
                "prop": prop,
                "zone": zone.to_value(),
                "settled_speed": settled_speed,
            }),
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => serde_json::json!({
                "kind": if matches!(self, Predicate::AllOf(_)) { KIND_ALL_OF } else { KIND_ANY_OF },
                "terms": terms.iter().map(Predicate::to_value).collect::<Vec<_>>(),
            }),
        }
    }

    /// Every rule python's constructors enforce, applied to a whole tree.
    pub fn validate(&self) -> Result<(), TaskError> {
        match self {
            Predicate::Lifted {
                prop,
                height,
                reference,
            } => {
                if prop.trim().is_empty() {
                    return Err(invalid("lifted `prop` must be a non-empty string"));
                }
                if !height.is_finite() {
                    return Err(invalid(format!(
                        "lifted `height` must be finite, got {height}"
                    )));
                }
                // A "lift" of zero or less is not a lift; `absolute` may sit anywhere.
                if *reference == LiftedRef::Initial && *height <= 0.0 {
                    return Err(invalid(format!(
                        "lifted `height` must be > 0 for ref=\"initial\" (a lift of {height} m \
                         is not a lift); use ref=\"absolute\" for a world-z threshold"
                    )));
                }
                Ok(())
            }
            Predicate::PlacedInZone {
                prop,
                zone,
                settled_speed,
            } => {
                if prop.trim().is_empty() {
                    return Err(invalid("placed_in_zone `prop` must be a non-empty string"));
                }
                if let ZoneRef::Inline(z) = zone {
                    z.validate()?;
                }
                if let Some(s) = settled_speed
                    && !(s.is_finite() && *s >= 0.0)
                {
                    return Err(invalid(format!(
                        "placed_in_zone `settled_speed` must be finite and >= 0, got {s}"
                    )));
                }
                Ok(())
            }
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => {
                if terms.is_empty() {
                    return Err(invalid(format!("{} needs at least one term", self.kind())));
                }
                for t in terms {
                    t.validate()?;
                }
                Ok(())
            }
        }
    }

    /// The JSON `kind` tag of this node.
    pub fn kind(&self) -> &'static str {
        match self {
            Predicate::Lifted { .. } => KIND_LIFTED,
            Predicate::PlacedInZone { .. } => KIND_PLACED_IN_ZONE,
            Predicate::AllOf(_) => KIND_ALL_OF,
            Predicate::AnyOf(_) => KIND_ANY_OF,
        }
    }

    /// Short stable identifier for log lines, e.g. `lifted:cube`.
    pub fn name(&self) -> String {
        match self {
            Predicate::Lifted { prop, .. } => format!("{KIND_LIFTED}:{prop}"),
            Predicate::PlacedInZone { prop, .. } => format!("{KIND_PLACED_IN_ZONE}:{prop}"),
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => format!(
                "{}({})",
                self.kind(),
                terms
                    .iter()
                    .map(Predicate::name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// One plain-English sentence stating what counts as success — the exact
    /// wording `caliper_learn.success.describe` produces, so an eval report
    /// and Studio's task panel read identically.
    pub fn describe(&self) -> String {
        match self {
            Predicate::Lifted {
                prop,
                height,
                reference,
            } => match reference {
                LiftedRef::Absolute => format!("{prop}'s center reaches z >= {height:.3} m"),
                LiftedRef::Initial => {
                    format!("{prop} rises {height:.3} m above its initial height")
                }
            },
            Predicate::PlacedInZone {
                prop,
                zone,
                settled_speed,
            } => {
                let where_ = match zone {
                    ZoneRef::Inline(z) => z.describe(),
                    ZoneRef::Named(n) => format!("the zone `{n}`"),
                };
                let mut s = format!("{prop}'s center is inside {where_}");
                if let Some(v) = settled_speed {
                    s.push_str(&format!(" and has come to rest (speed <= {v:.3} m/s)"));
                }
                s
            }
            Predicate::AllOf(terms) => join_describe(terms, "and"),
            Predicate::AnyOf(terms) => join_describe(terms, "or"),
        }
    }

    /// Every prop name this predicate reads — what a scene must contain for it
    /// to be answerable.
    pub fn prop_names(&self) -> BTreeSet<&str> {
        let mut out = BTreeSet::new();
        self.collect_props(&mut out);
        out
    }

    fn collect_props<'a>(&'a self, out: &mut BTreeSet<&'a str>) {
        match self {
            Predicate::Lifted { prop, .. } | Predicate::PlacedInZone { prop, .. } => {
                out.insert(prop.as_str());
            }
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => {
                for t in terms {
                    t.collect_props(out);
                }
            }
        }
    }

    /// The first zone still referenced by NAME, if any. A predicate handed to a
    /// runner must carry real numbers, and this is how a caller checks that
    /// without waiting for the first judged state to fail.
    pub fn unresolved_zone(&self) -> Option<&str> {
        match self {
            Predicate::Lifted { .. } => None,
            Predicate::PlacedInZone {
                zone: ZoneRef::Named(n),
                ..
            } => Some(n),
            Predicate::PlacedInZone { .. } => None,
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => {
                terms.iter().find_map(Predicate::unresolved_zone)
            }
        }
    }

    /// Hand every zone reference in the tree to `f` — how a task file resolves
    /// [`ZoneRef::Named`] against its own `scene.zones` (the resolution rule,
    /// and its error text, live in [`crate::task`], which knows the scene).
    pub fn visit_zones_mut(
        &mut self,
        f: &mut impl FnMut(&mut ZoneRef) -> Result<(), TaskError>,
    ) -> Result<(), TaskError> {
        match self {
            Predicate::Lifted { .. } => Ok(()),
            Predicate::PlacedInZone { zone, .. } => f(zone),
            Predicate::AllOf(terms) | Predicate::AnyOf(terms) => {
                for t in terms {
                    t.visit_zones_mut(f)?;
                }
                Ok(())
            }
        }
    }

    /// Judge one state with a FRESH episode baseline — the exact equivalent of
    /// calling a newly built python predicate: a `ref="initial"` lift measures
    /// against this very state, so it can only be `false` for a positive
    /// height. Use a [`SuccessTracker`] to judge a stream of states.
    pub fn eval(&self, state: &SuccessState) -> Result<bool, TaskError> {
        let mut baselines = BTreeMap::new();
        eval_with(self, state, &mut baselines)
    }
}

fn join_describe(terms: &[Predicate], word: &str) -> String {
    terms
        .iter()
        .map(Predicate::describe)
        .collect::<Vec<_>>()
        .join(&format!(" {word} "))
}

/// The evaluator. `baselines` maps a prop to the z it had at episode start,
/// captured lazily on first use exactly like python's `Lifted._z0`.
fn eval_with(
    p: &Predicate,
    state: &SuccessState,
    baselines: &mut BTreeMap<String, f64>,
) -> Result<bool, TaskError> {
    match p {
        Predicate::Lifted {
            prop,
            height,
            reference,
        } => {
            let z = state.pos(prop)?[2];
            match reference {
                LiftedRef::Absolute => Ok(z >= *height),
                LiftedRef::Initial => {
                    // Standalone use: the first state IS the reference.
                    let z0 = *baselines.entry(prop.clone()).or_insert(z);
                    Ok((z - z0) >= *height)
                }
            }
        }
        Predicate::PlacedInZone {
            prop,
            zone,
            settled_speed,
        } => {
            if !zone.zone()?.contains(state.pos(prop)?) {
                return Ok(false);
            }
            let Some(limit) = settled_speed else {
                return Ok(true);
            };
            let v = state.vel(prop)?;
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            Ok(speed <= *limit)
        }
        // Evaluate EVERY term (no short-circuit on false): a skipped term is a
        // term whose initial reference never got captured, which would make the
        // verdict depend on the order the terms happened to fail in.
        Predicate::AllOf(terms) => {
            let mut all = true;
            for t in terms {
                all &= eval_with(t, state, baselines)?;
            }
            Ok(all)
        }
        Predicate::AnyOf(terms) => {
            let mut any = false;
            for t in terms {
                any |= eval_with(t, state, baselines)?;
            }
            Ok(any)
        }
    }
}

// ===== serde (delegating to the checked JSON form) =====

impl Serialize for Predicate {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Predicate {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        Predicate::from_value(&v).map_err(serde::de::Error::custom)
    }
}

// ===== the stateful half =====

/// A predicate plus the per-episode baselines `ref="initial"` needs.
///
/// [`reset`](Self::reset) starts an episode — with a state it captures the
/// baselines NOW, without one it clears them so the next
/// [`judge`](Self::judge) captures instead. This is where python keeps
/// `Lifted._z0`; keeping it out here is what lets one [`Predicate`] be cloned
/// across N environments without sharing a captured height.
#[derive(Clone, Debug)]
pub struct SuccessTracker {
    predicate: Predicate,
    baselines: BTreeMap<String, f64>,
}

impl SuccessTracker {
    pub fn new(predicate: Predicate) -> Self {
        Self {
            predicate,
            baselines: BTreeMap::new(),
        }
    }

    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// Start a new episode. `Some(state)` captures every `ref="initial"`
    /// baseline from it (an error when the state lacks a referenced prop —
    /// silently scoring against a missing object is the bug this prevents);
    /// `None` clears them.
    pub fn reset(&mut self, state: Option<&SuccessState>) -> Result<(), TaskError> {
        self.baselines.clear();
        let Some(state) = state else { return Ok(()) };
        let mut props: Vec<String> = Vec::new();
        collect_initial_lift_props(&self.predicate, &mut props);
        for prop in props {
            let z = state.pos(&prop)?[2];
            self.baselines.insert(prop, z);
        }
        Ok(())
    }

    /// The verdict for `state`, capturing any baseline not yet captured.
    pub fn judge(&mut self, state: &SuccessState) -> Result<bool, TaskError> {
        eval_with(&self.predicate, state, &mut self.baselines)
    }

    /// Captured `ref="initial"` baselines, for diagnostics.
    pub fn baselines(&self) -> &BTreeMap<String, f64> {
        &self.baselines
    }
}

fn collect_initial_lift_props(p: &Predicate, out: &mut Vec<String>) {
    match p {
        Predicate::Lifted {
            prop,
            reference: LiftedRef::Initial,
            ..
        } => {
            if !out.iter().any(|x| x == prop) {
                out.push(prop.clone());
            }
        }
        Predicate::Lifted { .. } | Predicate::PlacedInZone { .. } => {}
        Predicate::AllOf(terms) | Predicate::AnyOf(terms) => {
            for t in terms {
                collect_initial_lift_props(t, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> Value {
        serde_json::from_str(s).expect("test json parses")
    }

    fn pred(s: &str) -> Predicate {
        Predicate::from_value(&json(s)).expect("predicate parses")
    }

    #[test]
    fn json_round_trip_is_exact() {
        for text in [
            r#"{"kind":"lifted","prop":"cube","height":0.05,"ref":"initial"}"#,
            r#"{"kind":"lifted","prop":"cube","height":0.3,"ref":"absolute"}"#,
            r#"{"kind":"placed_in_zone","prop":"cube",
                "zone":{"center":[0.4,0.2,0.02],"half":[0.05,0.05,0.02]},
                "settled_speed":0.01}"#,
            r#"{"kind":"placed_in_zone","prop":"cube",
                "zone":{"center":[0.0,0.0,0.0],"half":[1.0,1.0,1.0]},
                "settled_speed":null}"#,
            r#"{"kind":"all_of","terms":[
                {"kind":"lifted","prop":"cube","height":0.05,"ref":"initial"},
                {"kind":"any_of","terms":[
                  {"kind":"placed_in_zone","prop":"cube",
                   "zone":{"center":[0.4,0.2,0.02],"half":[0.05,0.05,0.02]},
                   "settled_speed":null}]}]}"#,
        ] {
            let v = json(text);
            let p = Predicate::from_value(&v).expect("parses");
            assert_eq!(p.to_value(), v, "round-trip drifted for {text}");
            // and through serde, which is the path a task file takes
            let via_serde: Predicate =
                serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(via_serde, p);
        }
    }

    #[test]
    fn a_missing_ref_defaults_to_initial() {
        let p = pred(r#"{"kind":"lifted","prop":"cube","height":0.05}"#);
        assert_eq!(
            p,
            Predicate::Lifted {
                prop: "cube".into(),
                height: 0.05,
                reference: LiftedRef::Initial
            }
        );
        // ... and comes BACK spelled out, like python's to_dict
        assert_eq!(p.to_value()["ref"], Value::String("initial".into()));
    }

    #[test]
    fn unknown_keys_and_kinds_raise_at_every_level() {
        let cases = [
            (
                r#"{"kind":"lifted","prop":"c","height":0.05,"reff":"initial"}"#,
                "unknown key",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c","zone":{"center":[0,0,0],"half":[1,1,1]},"settle_speed":0.1}"#,
                "unknown key",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c","zone":{"center":[0,0,0],"half":[1,1,1],"size":[1,1,1]}}"#,
                "unknown key",
            ),
            (r#"{"kind":"all_of","terms":[],"mode":"x"}"#, "unknown key"),
            (
                r#"{"kind":"all_of","terms":[{"kind":"lifted","prop":"c","height":1,"nope":1}]}"#,
                "unknown key",
            ),
            (
                r#"{"kind":"lifted_up","prop":"c","height":0.05}"#,
                "unknown success predicate kind",
            ),
            (r#"{"prop":"c","height":0.05}"#, "needs a string `kind`"),
            (r#"{"kind":"all_of","terms":[]}"#, "at least one term"),
            (r#"{"kind":"any_of"}"#, "at least one term"),
            (r#"{"kind":"lifted","prop":"c","height":0.0}"#, "not a lift"),
            (
                r#"{"kind":"lifted","prop":"","height":0.05}"#,
                "non-empty string",
            ),
            (
                r#"{"kind":"lifted","prop":"c"}"#,
                "missing required key `height`",
            ),
            (
                r#"{"kind":"lifted","prop":"c","height":"tall"}"#,
                "must be a number",
            ),
            (
                r#"{"kind":"lifted","prop":"c","height":1,"ref":"up"}"#,
                "must be \"initial\"",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c","zone":{"center":[0,0,0],"half":[1,-1,1]}}"#,
                "non-negative",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c","zone":{"center":[0,0],"half":[1,1,1]}}"#,
                "must be 3 numbers",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c"}"#,
                "missing required key `zone`",
            ),
            (
                r#"{"kind":"placed_in_zone","prop":"c","zone":{"center":[0,0,0],"half":[1,1,1]},"settled_speed":-1}"#,
                ">= 0",
            ),
            (r#"[1,2,3]"#, "must be an object"),
        ];
        for (text, want) in cases {
            let err = Predicate::from_value(&json(text))
                .map(|_| ())
                .expect_err(&format!("must be refused: {text}"))
                .to_string();
            assert!(
                err.contains(want),
                "for {text}\n  got: {err}\n  want: {want}"
            );
        }
    }

    #[test]
    fn describe_matches_the_python_wording() {
        assert_eq!(
            pred(r#"{"kind":"lifted","prop":"cube","height":0.05}"#).describe(),
            "cube rises 0.050 m above its initial height"
        );
        assert_eq!(
            pred(r#"{"kind":"lifted","prop":"cube","height":0.3,"ref":"absolute"}"#).describe(),
            "cube's center reaches z >= 0.300 m"
        );
        assert_eq!(
            pred(
                r#"{"kind":"placed_in_zone","prop":"cube",
                    "zone":{"center":[0.4,0.2,0.02],"half":[0.05,0.05,0.02]},
                    "settled_speed":0.01}"#
            )
            .describe(),
            "cube's center is inside the 0.100 x 0.100 x 0.040 m box centered at \
             (0.400, 0.200, 0.020) and has come to rest (speed <= 0.010 m/s)"
        );
        let both = pred(
            r#"{"kind":"all_of","terms":[
                {"kind":"lifted","prop":"cube","height":0.05},
                {"kind":"placed_in_zone","prop":"cube",
                 "zone":{"center":[0,0,0],"half":[1,1,1]}}]}"#,
        );
        assert_eq!(
            both.describe(),
            "cube rises 0.050 m above its initial height and cube's center is inside \
             the 2.000 x 2.000 x 2.000 m box centered at (0.000, 0.000, 0.000)"
        );
        assert_eq!(both.name(), "all_of(lifted:cube, placed_in_zone:cube)");
    }

    #[test]
    fn zone_faces_are_inclusive() {
        let z = Zone::new([0.0, 0.0, 0.0], [0.5, 0.25, 0.125]).unwrap();
        assert!(z.contains([0.5, 0.25, 0.125]));
        assert!(z.contains([-0.5, -0.25, -0.125]));
        assert!(!z.contains([0.5, 0.25, 0.126]));
        // a zero extent is a legal degenerate constraint (python allows it)
        let plane = Zone::new([0.0, 0.0, 1.0], [1.0, 1.0, 0.0]).unwrap();
        assert!(plane.contains([0.5, 0.5, 1.0]));
        assert!(!plane.contains([0.5, 0.5, 1.0 + f64::EPSILON]));
    }

    #[test]
    fn a_named_zone_cannot_be_evaluated() {
        let mut p = pred(r#"{"kind":"placed_in_zone","prop":"cube","zone":"bin"}"#);
        let state = SuccessState::from_positions([("cube", [0.0, 0.0, 0.0])]);
        let err = p.eval(&state).unwrap_err().to_string();
        assert!(err.contains("never resolved"), "got: {err}");
        // resolving it makes the same predicate answerable
        p.visit_zones_mut(&mut |z| {
            *z = ZoneRef::Inline(Zone::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])?);
            Ok(())
        })
        .unwrap();
        assert!(p.eval(&state).unwrap());
    }

    #[test]
    fn a_settled_check_without_velocities_errors() {
        let p = pred(
            r#"{"kind":"placed_in_zone","prop":"cube",
                "zone":{"center":[0,0,0],"half":[1,1,1]},"settled_speed":0.01}"#,
        );
        let no_vel = SuccessState::from_positions([("cube", [0.0, 0.0, 0.0])]);
        let err = p.eval(&no_vel).unwrap_err().to_string();
        assert!(err.contains("no velocities"), "got: {err}");
        // with velocities it answers
        let with_vel = no_vel.clone().with_velocities([("cube", [0.0, 0.0, 0.0])]);
        assert!(p.eval(&with_vel).unwrap());
        // an unknown prop is an error, never a false
        let elsewhere = SuccessState::from_positions([("ball", [0.0, 0.0, 0.0])]);
        let err = p.eval(&elsewhere).unwrap_err().to_string();
        assert!(err.contains("no prop named `cube`"), "got: {err}");
    }

    #[test]
    fn the_tracker_captures_and_reclears_the_lift_baseline() {
        let mut t = SuccessTracker::new(pred(
            r#"{"kind":"lifted","prop":"cube","height":0.05,"ref":"initial"}"#,
        ));
        let low = SuccessState::from_positions([("cube", [0.0, 0.0, 0.10])]);
        let high = SuccessState::from_positions([("cube", [0.0, 0.0, 0.30])]);

        t.reset(Some(&low)).unwrap();
        assert_eq!(t.baselines().get("cube"), Some(&0.10));
        assert!(!t.judge(&low).unwrap());
        assert!(t.judge(&high).unwrap());

        // Re-reset AT the high pose: the same 0.05 bar is now measured from
        // there, so the high state stops being a success.
        t.reset(Some(&high)).unwrap();
        assert!(!t.judge(&high).unwrap());
        assert!(
            !t.judge(&low).unwrap(),
            "falling below the baseline is not a lift"
        );

        // reset(None) clears: the next judged state becomes the baseline.
        t.reset(None).unwrap();
        assert!(t.baselines().is_empty());
        assert!(!t.judge(&low).unwrap());
        assert_eq!(t.baselines().get("cube"), Some(&0.10));
        assert!(t.judge(&high).unwrap());

        // resetting against a state that lacks the prop is loud
        let err = t
            .reset(Some(&SuccessState::from_positions([(
                "ball",
                [0.0, 0.0, 0.0],
            )])))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no prop named `cube`"), "got: {err}");
    }

    #[test]
    fn absolute_lift_needs_no_baseline() {
        let p = pred(r#"{"kind":"lifted","prop":"cube","height":0.2,"ref":"absolute"}"#);
        assert!(
            !p.eval(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.19])]))
                .unwrap()
        );
        assert!(
            p.eval(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.2])]))
                .unwrap()
        );
        // and the tracker captures nothing for it
        let mut t = SuccessTracker::new(p);
        t.reset(Some(&SuccessState::from_positions([(
            "cube",
            [0.0, 0.0, 0.0],
        )])))
        .unwrap();
        assert!(t.baselines().is_empty());
    }

    #[test]
    fn combinators_evaluate_every_term() {
        // The `any_of` is false on its first term but must still evaluate the
        // second, whose baseline capture is the whole point.
        let mut t = SuccessTracker::new(pred(
            r#"{"kind":"any_of","terms":[
                {"kind":"placed_in_zone","prop":"cube",
                 "zone":{"center":[9,9,9],"half":[0.1,0.1,0.1]}},
                {"kind":"lifted","prop":"cube","height":0.05}]}"#,
        ));
        let low = SuccessState::from_positions([("cube", [0.0, 0.0, 0.10])]);
        assert!(!t.judge(&low).unwrap());
        assert_eq!(
            t.baselines().get("cube"),
            Some(&0.10),
            "the second term never got to capture its baseline"
        );
        assert!(
            t.judge(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.20])]))
                .unwrap()
        );

        // all_of: both terms must hold of the SAME state
        let mut t = SuccessTracker::new(pred(
            r#"{"kind":"all_of","terms":[
                {"kind":"lifted","prop":"cube","height":0.05},
                {"kind":"placed_in_zone","prop":"cube",
                 "zone":{"center":[0.0,0.0,0.2],"half":[0.05,0.05,0.05]}}]}"#,
        ));
        t.reset(Some(&low)).unwrap();
        // lifted 0.10 m and inside the box at z = 0.20
        assert!(
            t.judge(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.20])]))
                .unwrap()
        );
        // lifted, but nowhere near the box
        assert!(
            !t.judge(&SuccessState::from_positions([("cube", [1.0, 0.0, 0.20])]))
                .unwrap()
        );
    }

    /// The knife edge python's module doc calls out: 0.15 − 0.10 is
    /// 0.049999999999999996, which does NOT clear a 0.05 bar. Both faces do
    /// the same binary arithmetic, so both must say `false` here.
    #[test]
    fn no_epsilon_is_invented_at_the_threshold() {
        let mut t = SuccessTracker::new(pred(r#"{"kind":"lifted","prop":"cube","height":0.05}"#));
        t.reset(Some(&SuccessState::from_positions([(
            "cube",
            [0.0, 0.0, 0.10],
        )])))
        .unwrap();
        assert!(
            !t.judge(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.15])]))
                .unwrap()
        );
        // and 0.16 clears it comfortably
        assert!(
            t.judge(&SuccessState::from_positions([("cube", [0.0, 0.0, 0.16])]))
                .unwrap()
        );
    }

    #[test]
    fn prop_names_are_collected_from_the_whole_tree() {
        let p = pred(
            r#"{"kind":"all_of","terms":[
                {"kind":"lifted","prop":"cube","height":0.05},
                {"kind":"any_of","terms":[
                  {"kind":"placed_in_zone","prop":"ball",
                   "zone":{"center":[0,0,0],"half":[1,1,1]}}]}]}"#,
        );
        assert_eq!(
            p.prop_names().into_iter().collect::<Vec<_>>(),
            vec!["ball", "cube"]
        );
    }
}
