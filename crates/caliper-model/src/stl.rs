//! Tiny, pure-Rust STL loader (binary + ASCII) → a vertex cloud.
//!
//! Returns every triangle vertex (no deduplication, no topology): the only
//! downstream consumer is [`crate::hull::convex_hull`], whose support over a
//! point cloud equals the support over its convex hull, so faces/normals are
//! irrelevant. Dependency-free; all parsing is bounds-checked and returns
//! `None` on any malformed input (the caller then keeps the mesh DROPPED — the
//! pre-existing, safe behavior).

use nalgebra::Point3;

/// Parse STL `bytes` into the cloud of all triangle vertices. `None` if the
/// buffer is not a recognizable / well-formed STL or contains no finite vertex.
pub fn parse_stl(bytes: &[u8]) -> Option<Vec<Point3<f64>>> {
    if looks_binary(bytes) {
        parse_binary(bytes)
    } else {
        parse_ascii(bytes)
    }
}

/// Binary STL is `84 + 50·n` bytes, where `n` is the little-endian `u32` at
/// offset 80. An ASCII file's header text makes that size relation fail, so the
/// exact-size match is the standard robust discriminator (an ASCII file that
/// merely starts with "solid" will not satisfy it).
fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    // Guard against overflow on absurd counts before the multiply.
    match n.checked_mul(50) {
        Some(body) => bytes.len() == 84 + body,
        None => false,
    }
}

fn parse_binary(bytes: &[u8]) -> Option<Vec<Point3<f64>>> {
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let mut out = Vec::with_capacity(n * 3);
    let mut off = 84usize;
    let rd_f32 =
        |b: &[u8], o: usize| -> f32 { f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    for _ in 0..n {
        // 50-byte record: 3 floats normal, 3×3 floats verts, 2 bytes attr.
        if off + 50 > bytes.len() {
            return None;
        }
        // skip the 12-byte normal; read three vertices (9 floats)
        for v in 0..3 {
            let base = off + 12 + v * 12;
            let x = rd_f32(bytes, base) as f64;
            let y = rd_f32(bytes, base + 4) as f64;
            let z = rd_f32(bytes, base + 8) as f64;
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                return None;
            }
            out.push(Point3::new(x, y, z));
        }
        off += 50;
    }
    (!out.is_empty()).then_some(out)
}

fn parse_ascii(bytes: &[u8]) -> Option<Vec<Point3<f64>>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut out = Vec::new();
    // Tokenize on whitespace; every "vertex" token is followed by three floats.
    let mut it = text.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == "vertex" {
            let x: f64 = it.next()?.parse().ok()?;
            let y: f64 = it.next()?.parse().ok()?;
            let z: f64 = it.next()?.parse().ok()?;
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                return None;
            }
            out.push(Point3::new(x, y, z));
        }
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal binary STL for one triangle.
    fn bin_tri(verts: [[f32; 3]; 3]) -> Vec<u8> {
        let mut b = vec![0u8; 80]; // header
        b.extend_from_slice(&1u32.to_le_bytes()); // count
        b.extend_from_slice(&[0u8; 12]); // normal
        for v in verts {
            for c in v {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 2]); // attribute
        b
    }

    #[test]
    fn parses_binary_triangle() {
        let b = bin_tri([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(looks_binary(&b));
        let v = parse_stl(&b).unwrap();
        assert_eq!(v.len(), 3);
        assert!((v[1] - Point3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn parses_ascii_triangle() {
        let s = "solid t\nfacet normal 0 0 1\nouter loop\n\
                 vertex 0 0 0\nvertex 2 0 0\nvertex 0 3 0\n\
                 endloop\nendfacet\nendsolid t\n";
        assert!(!looks_binary(s.as_bytes()));
        let v = parse_stl(s.as_bytes()).unwrap();
        assert_eq!(v.len(), 3);
        assert!((v[2] - Point3::new(0.0, 3.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn truncated_binary_rejected() {
        let mut b = bin_tri([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        // claim two triangles but only ship one body → size mismatch → ASCII path → no verts
        b[80] = 2;
        assert!(parse_stl(&b).is_none());
    }

    #[test]
    fn empty_or_garbage_is_none() {
        assert!(parse_stl(b"").is_none());
        assert!(parse_stl(b"solid nothing here endsolid").is_none());
    }
}
