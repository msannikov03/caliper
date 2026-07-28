//! Minimal XML DOM (parse + pretty re-emit) for URDF doctoring.
//!
//! Hand-rolled for the same reason `caliper_model::xacro` hand-rolls its
//! parser (quick-xml is not a workspace dependency), but DIFFERENT in intent:
//! the doctor must read files that urdf-rs REJECTS outright (a `<limit>`
//! without `velocity=`, xacro leftovers, negative masses, …) — a lenient DOM
//! is the whole point of the crate. Supported: elements, attributes (either
//! quote style; a repeated attribute keeps its FIRST value, matching lookup
//! order — re-emitting both would produce a file no strict parser accepts),
//! text, comments (preserved verbatim), the XML declaration and processing
//! instructions (skipped). Rejected loudly: DOCTYPE and CDATA, exactly like
//! the xacro expander.
//!
//! The writer re-emits a normalized 2-space-indented document. Comments are
//! kept verbatim (except that `--` inside one gains a space and a trailing
//! `-` a padding space — XML forbids them, and the repaired copy must stay
//! loadable); insignificant whitespace is not. It is used to produce repaired
//! COPIES only — an input file is never overwritten.

/// Parse/shape error, positioned by byte offset into the input.
#[derive(thiserror::Error, Debug)]
#[error("XML at byte {at}: {msg}")]
pub struct XmlError {
    pub at: usize,
    pub msg: String,
}

/// An element: name, attributes in document order, children in document order.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

/// A child node. Whitespace-only text is dropped at parse time.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Element(Element),
    Text(String),
    Comment(String),
}

impl Element {
    pub fn new(name: &str) -> Self {
        Element {
            name: name.to_string(),
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
    /// Replace `key`'s value, or append the attribute if absent.
    pub fn set_attr(&mut self, key: &str, value: &str) {
        match self.attrs.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value.to_string(),
            None => self.attrs.push((key.to_string(), value.to_string())),
        }
    }
    /// First child element named `name`.
    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children_named(name).next()
    }
    pub fn child_mut(&mut self, name: &str) -> Option<&mut Element> {
        self.children_named_mut(name).next()
    }
    /// All child elements (any name), in document order.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            _ => None,
        })
    }
    pub fn children_named<'s>(&'s self, name: &str) -> impl Iterator<Item = &'s Element> + 's {
        // Own the name so the returned iterator borrows only `self` (callers
        // like `child()` pass transient &str names).
        let name = name.to_owned();
        self.elements().filter(move |e| e.name == name)
    }
    pub fn children_named_mut<'s>(
        &'s mut self,
        name: &str,
    ) -> impl Iterator<Item = &'s mut Element> + 's {
        let name = name.to_owned();
        self.children.iter_mut().filter_map(move |n| match n {
            Node::Element(e) if e.name == name => Some(e),
            _ => None,
        })
    }
}

// ===== parsing =====

struct Parser<'a> {
    s: &'a str,
    i: usize,
}

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b':' | b'_' | b'-' | b'.')
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }
    fn err(&self, msg: impl Into<String>) -> XmlError {
        XmlError {
            at: self.i,
            msg: msg.into(),
        }
    }
    fn skip_ws(&mut self) {
        let b = self.s.as_bytes();
        while self.i < b.len() && b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, tok: &str) -> bool {
        if self.rest().starts_with(tok) {
            self.i += tok.len();
            true
        } else {
            false
        }
    }
    /// Consume up to and including `tok`, returning the text before it.
    fn until(&mut self, tok: &str) -> Result<&'a str, XmlError> {
        match self.rest().find(tok) {
            Some(j) => {
                let out = &self.rest()[..j];
                self.i += j + tok.len();
                Ok(out)
            }
            None => Err(self.err(format!("expected `{tok}` before end of input"))),
        }
    }
    fn read_name(&mut self) -> Result<String, XmlError> {
        let b = self.s.as_bytes();
        let start = self.i;
        while self.i < b.len() && is_name_byte(b[self.i]) {
            self.i += 1;
        }
        if self.i == start {
            return Err(self.err("expected a name"));
        }
        Ok(self.s[start..self.i].to_string())
    }

    fn parse_element(&mut self, depth: usize) -> Result<Element, XmlError> {
        if depth > 128 {
            return Err(self.err("element nesting deeper than 128"));
        }
        if !self.eat("<") {
            return Err(self.err("expected `<`"));
        }
        let name = self.read_name()?;
        let mut attrs = Vec::new();
        loop {
            self.skip_ws();
            if self.eat("/>") {
                return Ok(Element {
                    name,
                    attrs,
                    children: Vec::new(),
                });
            }
            if self.eat(">") {
                break;
            }
            let key = self.read_name()?;
            self.skip_ws();
            if !self.eat("=") {
                return Err(self.err(format!("attribute `{key}`: expected `=`")));
            }
            self.skip_ws();
            let raw = if self.eat("\"") {
                self.until("\"")?
            } else if self.eat("'") {
                self.until("'")?
            } else {
                return Err(self.err(format!("attribute `{key}`: expected a quoted value")));
            };
            // a repeated attribute keeps its FIRST value (matches `attr()`
            // lookup); re-emitting both would break every strict parser
            if !attrs.iter().any(|(k, _)| *k == key) {
                attrs.push((key, unescape(raw)));
            }
        }
        let mut children = Vec::new();
        loop {
            if self.eat("</") {
                let close = self.read_name()?;
                if close != name {
                    return Err(self.err(format!("`</{close}>` closes `<{name}>`")));
                }
                self.skip_ws();
                if !self.eat(">") {
                    return Err(self.err(format!("expected `>` after `</{close}`")));
                }
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if self.eat("<!--") {
                let c = self.until("-->")?;
                children.push(Node::Comment(c.to_string()));
                continue;
            }
            if self.rest().starts_with("<![") {
                return Err(self.err("CDATA is not supported"));
            }
            if self.eat("<?") {
                self.until("?>")?;
                continue;
            }
            if self.rest().starts_with("<!") {
                return Err(self.err("DOCTYPE / `<!` declarations are not supported"));
            }
            if self.rest().starts_with('<') {
                children.push(Node::Element(self.parse_element(depth + 1)?));
                continue;
            }
            if self.i >= self.s.len() {
                return Err(self.err(format!("unclosed element `<{name}>`")));
            }
            let j = self.rest().find('<').unwrap_or(self.rest().len());
            let text = unescape(&self.rest()[..j]);
            self.i += j;
            let text = text.trim();
            if !text.is_empty() {
                children.push(Node::Text(text.to_string()));
            }
        }
    }
}

/// Parse a whole document; returns the root element. Prolog (XML declaration,
/// comments) is skipped; trailing comments after the root are allowed.
pub fn parse_document(text: &str) -> Result<Element, XmlError> {
    let mut p = Parser { s: text, i: 0 };
    loop {
        p.skip_ws();
        if p.eat("<?") {
            p.until("?>")?;
            continue;
        }
        if p.eat("<!--") {
            p.until("-->")?;
            continue;
        }
        if p.rest().starts_with("<!") {
            return Err(p.err("DOCTYPE is not supported"));
        }
        break;
    }
    let root = p.parse_element(0)?;
    loop {
        p.skip_ws();
        if p.eat("<!--") {
            p.until("-->")?;
            continue;
        }
        if p.i < p.s.len() {
            return Err(p.err("trailing content after the root element"));
        }
        return Ok(root);
    }
}

fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    const KNOWN: [(&str, char); 5] = [
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&amp;", '&'),
        ("&quot;", '"'),
        ("&apos;", '\''),
    ];
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if let Some((tok, ch)) = KNOWN.iter().find(|(t, _)| rest.starts_with(t)) {
            out.push(*ch);
            rest = &rest[tok.len()..];
        } else if let Some((ch, len)) = numeric_ref(rest) {
            // decode &#NN; / &#xHH; here, or the writer re-escapes the `&`
            // into `&amp;#NN;` — a silent identifier rename in every copy
            out.push(ch);
            rest = &rest[len..];
        } else {
            // unknown entity: kept verbatim — the doctor is lenient by design
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// A numeric character reference (`&#45;` / `&#x2F;`) at the start of `rest`:
/// the decoded char and the reference's byte length. `None` when it is not a
/// valid reference (kept verbatim by the lenient path instead).
fn numeric_ref(rest: &str) -> Option<(char, usize)> {
    let body = rest.strip_prefix("&#")?;
    let end = body.find(';')?;
    let digits = &body[..end];
    let code = match digits.strip_prefix(['x', 'X']) {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None => digits.parse::<u32>().ok()?,
    };
    Some((char::from_u32(code)?, 2 + end + 1))
}

fn escape(s: &str, quote: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if quote => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

// ===== writing =====

/// Comment text made legal for re-emission: XML forbids `--` inside a comment
/// and a `-` right before the closing `-->`, and the lenient parser can have
/// accepted both. A space is inserted between consecutive dashes and after a
/// trailing dash — the ONE place the writer alters preserved content, and only
/// because the alternative is a repaired copy no parser loads.
fn comment_safe(c: &str) -> String {
    let mut out = String::with_capacity(c.len());
    for ch in c.chars() {
        if ch == '-' && out.ends_with('-') {
            out.push(' ');
        }
        out.push(ch);
    }
    if out.ends_with('-') {
        out.push(' ');
    }
    out
}

/// Re-emit a document from its root element (see the module docs for what is
/// normalized). The output always re-parses to a structurally equal tree,
/// except that a comment containing `--` (or ending in `-`) re-parses with
/// the spaces [`comment_safe`] inserted to keep the document well-formed.
pub fn write_document(root: &Element) -> String {
    let mut out = String::from("<?xml version=\"1.0\"?>\n");
    write_element(&mut out, root, 0);
    out
}

fn write_element(out: &mut String, e: &Element, depth: usize) {
    let pad = "  ".repeat(depth);
    out.push_str(&pad);
    out.push('<');
    out.push_str(&e.name);
    for (k, v) in &e.attrs {
        out.push(' ');
        out.push_str(k);
        out.push_str("=\"");
        out.push_str(&escape(v, true));
        out.push('"');
    }
    if e.children.is_empty() {
        out.push_str("/>\n");
        return;
    }
    // a single text child stays inline: <tag>text</tag>
    if let [Node::Text(t)] = e.children.as_slice() {
        out.push('>');
        out.push_str(&escape(t, false));
        out.push_str("</");
        out.push_str(&e.name);
        out.push_str(">\n");
        return;
    }
    out.push_str(">\n");
    for child in &e.children {
        match child {
            Node::Element(c) => write_element(out, c, depth + 1),
            Node::Text(t) => {
                out.push_str(&pad);
                out.push_str("  ");
                out.push_str(&escape(t, false));
                out.push('\n');
            }
            Node::Comment(c) => {
                out.push_str(&pad);
                out.push_str("  <!--");
                out.push_str(&comment_safe(c));
                out.push_str("-->\n");
            }
        }
    }
    out.push_str(&pad);
    out.push_str("</");
    out.push_str(&e.name);
    out.push_str(">\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_attrs_entities_and_comments() {
        let doc = r#"<?xml version="1.0"?>
<!-- prolog comment -->
<robot name="a &amp; b">
  <!-- kept -->
  <link name='l1'/>
  <note>hi &lt;there&gt;</note>
</robot>"#;
        let root = parse_document(doc).unwrap();
        assert_eq!(root.name, "robot");
        assert_eq!(root.attr("name"), Some("a & b"));
        assert_eq!(root.child("link").unwrap().attr("name"), Some("l1"));
        assert_eq!(
            root.child("note").unwrap().children,
            vec![Node::Text("hi <there>".to_string())]
        );
        assert!(
            root.children
                .iter()
                .any(|n| matches!(n, Node::Comment(c) if c == " kept ")),
            "comment text is preserved verbatim, whitespace included"
        );
    }

    /// Regression (#16, 2026-07-28 audit): numeric character refs were left
    /// verbatim by unescape, so the writer re-escaped `&#45;` into
    /// `&amp;#45;` — a silent identifier rename in every repaired copy.
    #[test]
    fn numeric_character_refs_decode_and_round_trip() {
        let root = parse_document(r##"<robot name="a&#45;b&#x2F;c&#X41;"/>"##).unwrap();
        assert_eq!(root.attr("name"), Some("a-b/cA"));
        let emitted = write_document(&root);
        assert!(emitted.contains("a-b/cA"), "{emitted}");
        assert!(!emitted.contains("&amp;#"), "no re-escaped refs: {emitted}");
        assert_eq!(parse_document(&emitted).unwrap(), root);
        // malformed refs stay verbatim (lenient), like unknown entities
        let bad = parse_document(r#"<a x="&#zz;&#x;"/>"#).unwrap();
        assert_eq!(bad.attr("x"), Some("&#zz;&#x;"));
    }

    /// Regression (#17, 2026-07-28 audit): duplicate attributes were all
    /// parsed AND all re-emitted, turning a leniently-loadable file into one
    /// no strict parser accepts. The first value wins (matching `attr()`).
    #[test]
    fn duplicate_attributes_dedupe_to_the_first_value() {
        let doc = r#"<robot name="a"><link name="l1" name="l2"/></robot>"#;
        let root = parse_document(doc).unwrap();
        let link = root.child("link").unwrap();
        assert_eq!(link.attrs, vec![("name".to_string(), "l1".to_string())]);
        let emitted = write_document(&root);
        assert_eq!(
            emitted.matches("name=").count(),
            2,
            "one per element: {emitted}"
        );
        parse_document(&emitted).unwrap();
    }

    /// Regression (#30, 2026-07-28 audit): comment whitespace was trimmed at
    /// parse, and a comment containing `--` was re-emitted as-is — illegal
    /// XML that strict parsers (and our own `until("-->")`) misread.
    #[test]
    fn comments_preserve_whitespace_and_double_dashes_are_made_legal() {
        // the third comment's `--->` closes at the FIRST `-->`, leaving a
        // trailing `-` in its text — the other illegal-on-write shape
        let doc = "<a><!--  spaced  out  --><!-- x -- y --><!-- z --->hi</a>";
        let root = parse_document(doc).unwrap();
        assert!(
            root.children
                .iter()
                .any(|n| matches!(n, Node::Comment(c) if c == "  spaced  out  "))
        );
        let emitted = write_document(&root);
        let back = parse_document(&emitted).expect("emitted document must stay parseable");
        // the verbatim comment round-trips exactly; the illegal one only
        // gained the legalizing spaces
        assert!(
            back.children
                .iter()
                .any(|n| matches!(n, Node::Comment(c) if c == "  spaced  out  "))
        );
        assert!(
            back.children
                .iter()
                .any(|n| matches!(n, Node::Comment(c) if c.contains("x - - y"))),
            "{emitted}"
        );
        assert!(!emitted.contains("---"), "no illegal dash runs: {emitted}");
        assert!(
            back.children
                .iter()
                .any(|n| matches!(n, Node::Text(t) if t == "hi")),
            "text child survives beside comments"
        );
    }

    #[test]
    fn self_closing_and_nesting() {
        let root = parse_document("<a><b x=\"1\"><c/></b><b x=\"2\"/></a>").unwrap();
        assert_eq!(root.children_named("b").count(), 2);
        assert!(root.child("b").unwrap().child("c").is_some());
    }

    #[test]
    fn mismatched_tag_is_error() {
        let e = parse_document("<a><b></a></b>").unwrap_err();
        assert!(e.to_string().contains("closes"), "{e}");
        assert!(parse_document("<a>").is_err(), "unclosed root");
        assert!(parse_document("<a/><b/>").is_err(), "two roots");
    }

    #[test]
    fn doctype_and_cdata_rejected() {
        assert!(parse_document("<!DOCTYPE x><a/>").is_err());
        assert!(parse_document("<a><![CDATA[x]]></a>").is_err());
    }

    #[test]
    fn write_then_reparse_round_trips() {
        let doc = r#"<robot name="r"><link name="l&quot;1"><inertial><mass value="1.5"/></inertial><!-- note --></link><joint type="revolute">text</joint></robot>"#;
        let root = parse_document(doc).unwrap();
        let emitted = write_document(&root);
        let back = parse_document(&emitted).unwrap();
        assert_eq!(root, back, "structural round-trip:\n{emitted}");
    }

    #[test]
    fn set_attr_replaces_or_appends() {
        let mut e = Element::new("axis");
        e.set_attr("xyz", "0 0 2");
        e.set_attr("xyz", "0 0 1");
        assert_eq!(e.attrs, vec![("xyz".to_string(), "0 0 1".to_string())]);
    }
}
