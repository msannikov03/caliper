//! Vendored MINIMAL xacro expander — pure Rust, no external `xacro` binary.
//!
//! Most real robot descriptions ship as `.xacro`, not plain URDF.
//! [`Model::from_urdf`](crate::Model::from_urdf) detects xacro input (a
//! `.xacro` extension or a root `<robot xmlns:xacro=..>`) and runs it through
//! [`expand`] first, so the ~90% subset those files actually use loads
//! directly. Supported:
//!
//! - `<xacro:property name=.. value=../>` — EAGER evaluation: the value is
//!   substituted at definition time, so properties must be defined before use
//!   (real files are ordered this way; lazy re-evaluation is NOT emulated).
//! - `${..}` substitution: a bare property reference, or a numeric expression
//!   over `+ - * / ( )`, float literals, numeric properties and the builtin
//!   `pi`. `$$` escapes a literal `$`.
//! - `$(find <pkg>)` — resolved to the package DIRECTORY through the same
//!   search roots the mesh resolver uses ([`crate::package_search_dirs`]:
//!   the URDF dir's ancestors, then `CALIPER_PACKAGE_PATH`).
//! - `<xacro:macro name=.. params="a b:=default *block">` definition and
//!   `<xacro:NAME ../>` instantiation. Block params are the SIMPLE form only:
//!   each `*param` binds exactly one child element of the call, spliced back
//!   via `<xacro:insert_block name=../>`.
//! - `<xacro:include filename=../>` — resolved relative to the INCLUDING file
//!   (the filename may use `$(find ..)` / `${..}`).
//! - `<xacro:if value=..>` / `<xacro:unless value=..>` where the substituted
//!   value is exactly `true`/`false`/`1`/`0`.
//!
//! Everything else is REJECTED with a [`XacroError`] naming the construct —
//! never silent wrong output. Known-unsupported (the other ~10%):
//! `<xacro:arg>` / `$(arg ..)`; `$(env ..)`, `$(optenv ..)`, `$(eval ..)`,
//! `$(anon ..)`; python expressions beyond numeric arithmetic (string
//! methods, comparisons/boolean operators, `cos(..)` etc., conditionals,
//! lists/dicts); lazy/recursive property evaluation; `scope=`/`default=` and
//! block-value forms of `<xacro:property>`; `^` param forwarding and quoted
//! macro-param defaults containing spaces; `<xacro:element>` /
//! `<xacro:attribute>`; namespaced includes (`ns=`); YAML loading; CDATA and
//! DOCTYPE internal subsets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Depth cap shared by macro instantiation and includes: catches recursive
/// macros and include cycles with a clear error instead of a stack overflow.
const MAX_DEPTH: usize = 64;

#[derive(thiserror::Error, Debug)]
pub enum XacroError {
    #[error("malformed XML: {0}")]
    Xml(String),
    #[error("unsupported xacro construct `{construct}`: {detail}")]
    Unsupported { construct: String, detail: String },
    #[error("undefined xacro property `{0}`")]
    UndefinedProperty(String),
    #[error("xacro expression `${{{0}}}`: {1}")]
    Expression(String, String),
    #[error("xacro macro `{0}`: {1}")]
    Macro(String, String),
    #[error("xacro include `{0}`: {1}")]
    Include(String, String),
    #[error(
        "$(find {0}): package directory not found (searched the URDF dir's \
         ancestors and CALIPER_PACKAGE_PATH) or not valid UTF-8"
    )]
    FindPackage(String),
    #[error("xacro condition value `{0}`: only true/false/1/0 (after substitution) is supported")]
    Condition(String),
}

/// True when `path`/`text` should be routed through [`expand`]: a `.xacro`
/// extension, or a root `<robot ..>` tag carrying an `xmlns:xacro` declaration.
pub(crate) fn is_xacro(path: &Path, text: &str) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("xacro"))
    {
        return true;
    }
    // Only look INSIDE the first `<robot ..>` tag, so a mention of xacro in a
    // comment or elsewhere does not trigger expansion.
    if let Some(i) = text.find("<robot") {
        let tag = match text[i..].find('>') {
            Some(j) => &text[i..i + j],
            None => &text[i..],
        };
        return tag.contains("xmlns:xacro");
    }
    false
}

/// Expand xacro `text` to a plain-URDF string. `dir` is the source file's
/// directory (resolves relative `<xacro:include>` and `$(find ..)`).
pub fn expand(text: &str, dir: Option<&Path>) -> Result<String, XacroError> {
    let root = XmlParser::new(text).parse_document()?;
    let mut ex = Expander::new();
    // Children first, THEN the root's own attributes: properties defined in
    // the body are visible to a parametrized root attribute.
    let children = ex.process_children(&root.children, dir)?;
    let mut attrs = Vec::with_capacity(root.attrs.len());
    for (k, v) in &root.attrs {
        if k == "xmlns:xacro" {
            continue; // consumed: the output is plain URDF
        }
        attrs.push((k.clone(), ex.substitute(v, dir)?));
    }
    let out_root = Element {
        name: root.name,
        attrs,
        children,
    };
    let mut out = String::with_capacity(text.len());
    out.push_str("<?xml version=\"1.0\"?>\n");
    write_element(&mut out, &out_root);
    Ok(out)
}

// ===== minimal XML tree (hand-rolled: quick-xml is not a direct dependency) =====

#[derive(Clone, Debug)]
enum Node {
    Element(Element),
    /// Raw text, entities untouched (they pass through to the URDF verbatim).
    Text(String),
}

#[derive(Clone, Debug)]
struct Element {
    /// Qualified name incl. any prefix, e.g. `xacro:property`.
    name: String,
    /// Raw attribute values in document order, entities untouched.
    attrs: Vec<(String, String)>,
    children: Vec<Node>,
}

fn attr<'e>(e: &'e Element, key: &str) -> Option<&'e str> {
    e.attrs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Recursive-descent XML parser for the WELL-FORMED subset URDF/xacro files
/// live in: elements, attributes (either quote style), text, comments
/// (dropped), `<?..?>` instructions (dropped). CDATA and DOCTYPE internal
/// subsets are rejected loudly.
struct XmlParser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> XmlParser<'a> {
    fn new(s: &'a str) -> Self {
        XmlParser { s, pos: 0 }
    }
    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }
    fn err(&self, msg: &str) -> XacroError {
        XacroError::Xml(format!("{msg} (at byte {})", self.pos))
    }
    fn eat(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }
    fn skip_ws(&mut self) {
        let r = self.rest();
        self.pos += r.len() - r.trim_start().len();
    }
    fn skip_until(&mut self, end: &str) -> Result<(), XacroError> {
        match self.rest().find(end) {
            Some(i) => {
                self.pos += i + end.len();
                Ok(())
            }
            None => Err(self.err(&format!("missing closing `{end}`"))),
        }
    }
    /// Skip whitespace, comments, `<?..?>` and `<!DOCTYPE..>` outside the root.
    fn skip_misc(&mut self) -> Result<(), XacroError> {
        loop {
            self.skip_ws();
            if self.eat("<!--") {
                self.skip_until("-->")?;
            } else if self.rest().starts_with("<?") {
                self.skip_until("?>")?;
            } else if self.rest().starts_with("<!") {
                let end = self
                    .rest()
                    .find('>')
                    .ok_or_else(|| self.err("unterminated `<!` declaration"))?;
                if self.rest()[..end].contains('[') {
                    return Err(XacroError::Unsupported {
                        construct: "DOCTYPE internal subset".into(),
                        detail: "declarations with `[..]` are not parsed".into(),
                    });
                }
                self.pos += end + 1;
            } else {
                return Ok(());
            }
        }
    }
    fn parse_document(mut self) -> Result<Element, XacroError> {
        self.eat("\u{feff}"); // BOM
        self.skip_misc()?;
        if !self.rest().starts_with('<') {
            return Err(self.err("expected a root element"));
        }
        let root = self.parse_element()?;
        self.skip_misc()?;
        if self.pos != self.s.len() {
            return Err(self.err("unexpected content after the root element"));
        }
        Ok(root)
    }
    fn name(&mut self) -> Result<String, XacroError> {
        let end = self
            .rest()
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')))
            .unwrap_or(self.rest().len());
        if end == 0 {
            return Err(self.err("expected an XML name"));
        }
        let n = self.rest()[..end].to_string();
        self.pos += end;
        Ok(n)
    }
    fn parse_element(&mut self) -> Result<Element, XacroError> {
        if !self.eat("<") {
            return Err(self.err("expected `<`"));
        }
        if self.rest().starts_with('$') {
            return Err(XacroError::Unsupported {
                construct: "element-name substitution".into(),
                detail: "`<${..}..>` dynamic tag names are not supported".into(),
            });
        }
        let name = self.name()?;
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
            let k = self.name()?;
            self.skip_ws();
            if !self.eat("=") {
                return Err(self.err("expected `=` after attribute name"));
            }
            self.skip_ws();
            let q = match self.rest().chars().next() {
                Some(c @ ('"' | '\'')) => c,
                _ => return Err(self.err("expected a quoted attribute value")),
            };
            self.pos += 1;
            let end = self
                .rest()
                .find(q)
                .ok_or_else(|| self.err("unterminated attribute value"))?;
            let v = self.rest()[..end].to_string();
            self.pos += end + 1;
            attrs.push((k, v));
        }
        let mut children = Vec::new();
        loop {
            if self.rest().is_empty() {
                return Err(self.err(&format!("unterminated element `<{name}>`")));
            }
            if self.eat("</") {
                let close = self.name()?;
                if close != name {
                    return Err(self.err(&format!("`<{name}>` closed by `</{close}>`")));
                }
                self.skip_ws();
                if !self.eat(">") {
                    return Err(self.err("expected `>` after closing tag name"));
                }
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if self.eat("<!--") {
                self.skip_until("-->")?;
            } else if self.rest().starts_with("<![CDATA[") {
                return Err(XacroError::Unsupported {
                    construct: "CDATA section".into(),
                    detail: "`<![CDATA[..]]>` is not parsed".into(),
                });
            } else if self.rest().starts_with("<?") {
                self.skip_until("?>")?;
            } else if self.rest().starts_with('<') {
                children.push(Node::Element(self.parse_element()?));
            } else {
                let end = self.rest().find('<').unwrap_or(self.rest().len());
                children.push(Node::Text(self.rest()[..end].to_string()));
                self.pos += end;
            }
        }
    }
}

fn write_node(out: &mut String, n: &Node) {
    match n {
        Node::Text(t) => out.push_str(t),
        Node::Element(e) => write_element(out, e),
    }
}

fn write_element(out: &mut String, e: &Element) {
    out.push('<');
    out.push_str(&e.name);
    for (k, v) in &e.attrs {
        out.push(' ');
        out.push_str(k);
        out.push('=');
        // Substitution can introduce quotes; pick a quote style that stays valid.
        if !v.contains('"') {
            out.push('"');
            out.push_str(v);
            out.push('"');
        } else if !v.contains('\'') {
            out.push('\'');
            out.push_str(v);
            out.push('\'');
        } else {
            out.push('"');
            out.push_str(&v.replace('"', "&quot;"));
            out.push('"');
        }
    }
    if e.children.is_empty() {
        out.push_str("/>");
    } else {
        out.push('>');
        for c in &e.children {
            write_node(out, c);
        }
        out.push_str("</");
        out.push_str(&e.name);
        out.push('>');
    }
}

// ===== expander =====

#[derive(Clone, Debug)]
struct Param {
    name: String,
    default: Option<String>,
    block: bool,
}

#[derive(Clone, Debug)]
struct Macro {
    params: Vec<Param>,
    /// Body stored RAW; substituted per instantiation.
    body: Vec<Node>,
    /// Directory of the DEFINING file — includes/`$(find)` inside the body
    /// resolve relative to where the macro was written, not where it's called.
    dir: Option<PathBuf>,
}

#[derive(Default)]
struct Scope {
    props: HashMap<String, String>,
    /// Block-param bindings (`*name` → the call-site child element).
    blocks: HashMap<String, Vec<Node>>,
}

struct Expander {
    /// Scope stack: `[global, macro frame, ..]`; lookups search top-down.
    scopes: Vec<Scope>,
    macros: HashMap<String, Macro>,
    depth: usize,
}

impl Expander {
    fn new() -> Self {
        Expander {
            scopes: vec![Scope::default()],
            macros: HashMap::new(),
            depth: 0,
        }
    }

    fn lookup(&self, name: &str) -> Option<&str> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.props.get(name))
            .map(String::as_str)
    }

    fn process_children(
        &mut self,
        nodes: &[Node],
        dir: Option<&Path>,
    ) -> Result<Vec<Node>, XacroError> {
        let mut out = Vec::with_capacity(nodes.len());
        for n in nodes {
            match n {
                Node::Text(t) => out.push(Node::Text(self.substitute(t, dir)?)),
                Node::Element(e) => self.process_element(e, dir, &mut out)?,
            }
        }
        Ok(out)
    }

    fn process_element(
        &mut self,
        e: &Element,
        dir: Option<&Path>,
        out: &mut Vec<Node>,
    ) -> Result<(), XacroError> {
        let Some(tag) = e.name.strip_prefix("xacro:") else {
            // Plain element: substitute attributes, recurse into children.
            let mut attrs = Vec::with_capacity(e.attrs.len());
            for (k, v) in &e.attrs {
                attrs.push((k.clone(), self.substitute(v, dir)?));
            }
            let children = self.process_children(&e.children, dir)?;
            out.push(Node::Element(Element {
                name: e.name.clone(),
                attrs,
                children,
            }));
            return Ok(());
        };
        match tag {
            "property" => self.define_property(e, dir),
            "macro" => self.define_macro(e, dir),
            "include" => self.include(e, dir, out),
            "if" => self.conditional(e, dir, out, true),
            "unless" => self.conditional(e, dir, out, false),
            "insert_block" => self.insert_block(e, dir, out),
            "arg" | "element" | "attribute" | "loop" => Err(XacroError::Unsupported {
                construct: format!("<xacro:{tag}>"),
                detail: "not implemented by the vendored minimal expander".into(),
            }),
            other => {
                let Some(mac) = self.macros.get(other).cloned() else {
                    return Err(XacroError::Unsupported {
                        construct: format!("<xacro:{other}>"),
                        detail: "not a defined <xacro:macro> and not a supported xacro construct"
                            .into(),
                    });
                };
                self.instantiate(other, &mac, e, dir, out)
            }
        }
    }

    fn define_property(&mut self, e: &Element, dir: Option<&Path>) -> Result<(), XacroError> {
        let name = attr(e, "name")
            .ok_or_else(|| XacroError::Xml("<xacro:property> without name=".into()))?
            .to_string();
        if attr(e, "scope").is_some()
            || attr(e, "default").is_some()
            || !e.children.iter().all(is_blank_text)
        {
            return Err(XacroError::Unsupported {
                construct: "<xacro:property>".into(),
                detail: format!(
                    "property `{name}` uses scope=/default=/block form; only \
                     `<xacro:property name=.. value=../>` is supported"
                ),
            });
        }
        let value = attr(e, "value").ok_or_else(|| {
            XacroError::Xml(format!("<xacro:property name=\"{name}\"> without value="))
        })?;
        // EAGER evaluation (see the module docs): define-before-use.
        let v = self.substitute(value, dir)?;
        self.scopes
            .last_mut()
            .expect("scope stack never empty")
            .props
            .insert(name, v);
        Ok(())
    }

    fn define_macro(&mut self, e: &Element, dir: Option<&Path>) -> Result<(), XacroError> {
        let name = attr(e, "name")
            .ok_or_else(|| XacroError::Xml("<xacro:macro> without name=".into()))?
            .to_string();
        let mut params = Vec::new();
        for tok in attr(e, "params").unwrap_or("").split_whitespace() {
            if let Some(b) = tok.strip_prefix('*') {
                params.push(Param {
                    name: b.to_string(),
                    default: None,
                    block: true,
                });
            } else if let Some((n, d)) = tok.split_once(":=") {
                if d.starts_with('^') {
                    return Err(XacroError::Unsupported {
                        construct: format!("macro param `{tok}`"),
                        detail: "`^` caller-property forwarding is not supported".into(),
                    });
                }
                // Single-token defaults only; a quoted default containing
                // whitespace was already split apart → reject loudly.
                if d.starts_with('\'') && (d.len() < 2 || !d.ends_with('\'')) {
                    return Err(XacroError::Unsupported {
                        construct: format!("macro param `{tok}`"),
                        detail: "quoted defaults containing spaces are not supported".into(),
                    });
                }
                params.push(Param {
                    name: n.to_string(),
                    default: Some(d.trim_matches('\'').to_string()),
                    block: false,
                });
            } else if tok.starts_with('^') {
                return Err(XacroError::Unsupported {
                    construct: format!("macro param `{tok}`"),
                    detail: "`^` caller-property forwarding is not supported".into(),
                });
            } else {
                params.push(Param {
                    name: tok.to_string(),
                    default: None,
                    block: false,
                });
            }
        }
        self.macros.insert(
            name,
            Macro {
                params,
                body: e.children.clone(),
                dir: dir.map(Path::to_path_buf),
            },
        );
        Ok(())
    }

    fn include(
        &mut self,
        e: &Element,
        dir: Option<&Path>,
        out: &mut Vec<Node>,
    ) -> Result<(), XacroError> {
        if attr(e, "ns").is_some() {
            return Err(XacroError::Unsupported {
                construct: "<xacro:include ns=..>".into(),
                detail: "namespaced includes are not supported".into(),
            });
        }
        let raw = attr(e, "filename")
            .ok_or_else(|| XacroError::Xml("<xacro:include> without filename=".into()))?;
        let f = self.substitute(raw, dir)?;
        let p = Path::new(&f);
        let path = if p.is_absolute() {
            p.to_path_buf()
        } else {
            match dir {
                Some(d) => d.join(p),
                None => p.to_path_buf(),
            }
        };
        let text = std::fs::read_to_string(&path)
            .map_err(|err| XacroError::Include(f.clone(), err.to_string()))?;
        let inc_root = XmlParser::new(&text)
            .parse_document()
            .map_err(|err| XacroError::Include(f.clone(), err.to_string()))?;
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(XacroError::Include(
                f,
                "include depth exceeded (cycle?)".into(),
            ));
        }
        // Splice the included root's children; its properties/macros register
        // in the CURRENT scope, matching xacro's flat include semantics.
        let nodes = self.process_children(&inc_root.children, path.parent());
        self.depth -= 1;
        out.extend(nodes?);
        Ok(())
    }

    fn conditional(
        &mut self,
        e: &Element,
        dir: Option<&Path>,
        out: &mut Vec<Node>,
        want: bool,
    ) -> Result<(), XacroError> {
        let raw = attr(e, "value")
            .ok_or_else(|| XacroError::Xml("<xacro:if>/<xacro:unless> without value=".into()))?;
        let v = self.substitute(raw, dir)?;
        let b = match v.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => return Err(XacroError::Condition(v)),
        };
        if b == want {
            let nodes = self.process_children(&e.children, dir)?;
            out.extend(nodes);
        }
        Ok(())
    }

    fn insert_block(
        &mut self,
        e: &Element,
        dir: Option<&Path>,
        out: &mut Vec<Node>,
    ) -> Result<(), XacroError> {
        let name = attr(e, "name")
            .ok_or_else(|| XacroError::Xml("<xacro:insert_block> without name=".into()))?;
        let nodes = self
            .scopes
            .iter()
            .rev()
            .find_map(|s| s.blocks.get(name))
            .cloned()
            .ok_or_else(|| {
                XacroError::Macro(
                    name.to_string(),
                    "<xacro:insert_block>: no such block parameter in scope".into(),
                )
            })?;
        let nodes = self.process_children(&nodes, dir)?;
        out.extend(nodes);
        Ok(())
    }

    fn instantiate(
        &mut self,
        name: &str,
        mac: &Macro,
        call: &Element,
        dir: Option<&Path>,
        out: &mut Vec<Node>,
    ) -> Result<(), XacroError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return Err(XacroError::Macro(
                name.to_string(),
                format!("recursion depth exceeded ({MAX_DEPTH})"),
            ));
        }
        let mut scope = Scope::default();
        let mut used = vec![false; call.attrs.len()];
        // Block params bind the call's child ELEMENTS in declaration order
        // (whitespace between them is ignored; comments were dropped at parse).
        let mut block_args = call.children.iter().filter(|n| !is_blank_text(n));
        for p in &mac.params {
            if p.block {
                let arg = block_args.next().ok_or_else(|| {
                    XacroError::Macro(
                        name.to_string(),
                        format!("missing block argument `*{}`", p.name),
                    )
                })?;
                let Node::Element(be) = arg else {
                    return Err(XacroError::Macro(
                        name.to_string(),
                        format!("block argument `*{}` must be an element, not text", p.name),
                    ));
                };
                scope
                    .blocks
                    .insert(p.name.clone(), vec![Node::Element(be.clone())]);
            } else {
                let raw = match call.attrs.iter().position(|(k, _)| k == &p.name) {
                    Some(i) => {
                        used[i] = true;
                        call.attrs[i].1.clone()
                    }
                    None => p.default.clone().ok_or_else(|| {
                        XacroError::Macro(
                            name.to_string(),
                            format!("missing required parameter `{}`", p.name),
                        )
                    })?,
                };
                // Substituted in the CALLER's scope, before the new frame is pushed.
                let v = self.substitute(&raw, dir)?;
                scope.props.insert(p.name.clone(), v);
            }
        }
        if let Some(i) = used.iter().position(|u| !u) {
            return Err(XacroError::Macro(
                name.to_string(),
                format!("unexpected attribute `{}`", call.attrs[i].0),
            ));
        }
        if block_args.next().is_some() {
            return Err(XacroError::Macro(
                name.to_string(),
                "more child elements than `*` block parameters".into(),
            ));
        }
        self.scopes.push(scope);
        let nodes = self.process_children(&mac.body, mac.dir.as_deref().or(dir));
        self.scopes.pop();
        self.depth -= 1;
        out.extend(nodes?);
        Ok(())
    }

    // ===== `${..}` / `$(..)` substitution =====

    fn substitute(&self, s: &str, dir: Option<&Path>) -> Result<String, XacroError> {
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(i) = rest.find('$') {
            out.push_str(&rest[..i]);
            let tail = &rest[i..];
            if let Some(t) = tail.strip_prefix("$$") {
                out.push('$'); // `$$` escapes a literal `$`
                rest = t;
            } else if let Some(t) = tail.strip_prefix("${") {
                let end = t.find('}').ok_or_else(|| {
                    XacroError::Expression(t.to_string(), "unterminated `${`".into())
                })?;
                let inner = &t[..end];
                if inner.contains('{') {
                    return Err(XacroError::Expression(
                        inner.to_string(),
                        "nested braces are not supported".into(),
                    ));
                }
                out.push_str(&self.eval(inner)?);
                rest = &t[end + 1..];
            } else if let Some(t) = tail.strip_prefix("$(") {
                let end = t.find(')').ok_or_else(|| {
                    XacroError::Expression(t.to_string(), "unterminated `$(`".into())
                })?;
                out.push_str(&dollar_paren(&t[..end], dir)?);
                rest = &t[end + 1..];
            } else {
                out.push('$');
                rest = &tail[1..];
            }
        }
        out.push_str(rest);
        Ok(out)
    }

    /// Evaluate the inside of `${..}`: a bare property reference (any string
    /// value), or a numeric `+ - * / ( )` expression over literals, numeric
    /// properties and `pi`.
    fn eval(&self, raw: &str) -> Result<String, XacroError> {
        let e = raw.trim();
        if e.is_empty() {
            return Err(XacroError::Expression(
                raw.to_string(),
                "empty expression".into(),
            ));
        }
        if is_ident(e) {
            if let Some(v) = self.lookup(e) {
                return Ok(v.to_string());
            }
            if e == "pi" {
                return Ok(std::f64::consts::PI.to_string());
            }
            return Err(XacroError::UndefinedProperty(e.to_string()));
        }
        let v = NumParser {
            s: e,
            pos: 0,
            ex: self,
        }
        .parse()
        .map_err(|m| XacroError::Expression(e.to_string(), m))?;
        if !v.is_finite() {
            return Err(XacroError::Expression(
                e.to_string(),
                "non-finite result".into(),
            ));
        }
        Ok(v.to_string())
    }
}

/// `$(..)` dispatch: only `$(find <pkg>)` is supported.
fn dollar_paren(inner: &str, dir: Option<&Path>) -> Result<String, XacroError> {
    let mut it = inner.split_whitespace();
    match (it.next(), it.next(), it.next()) {
        (Some("find"), Some(pkg), None) => {
            let p = resolve_package_dir(pkg, dir)
                .ok_or_else(|| XacroError::FindPackage(pkg.to_string()))?;
            p.into_os_string()
                .into_string()
                .map_err(|_| XacroError::FindPackage(pkg.to_string()))
        }
        (Some(cmd), ..) => Err(XacroError::Unsupported {
            construct: format!("$({cmd} ..)"),
            detail: "only $(find <pkg>) is supported".into(),
        }),
        (None, ..) => Err(XacroError::Expression(
            inner.to_string(),
            "empty `$(..)`".into(),
        )),
    }
}

/// `$(find <pkg>)` → the package DIRECTORY, through the same search roots the
/// mesh resolver uses ([`crate::package_search_dirs`]): under each root `A`,
/// first `A/<pkg>`, then `A` itself when its last component IS `pkg` (the
/// dir-level twins of the resolver's `A/<pkg>/<rest>` and `A/<rest>`).
fn resolve_package_dir(pkg: &str, urdf_dir: Option<&Path>) -> Option<PathBuf> {
    for d in crate::package_search_dirs(urdf_dir) {
        let sub = d.join(pkg);
        if sub.is_dir() {
            return Some(crate::absolutize(sub));
        }
        if d.file_name().and_then(|n| n.to_str()) == Some(pkg) && d.is_dir() {
            return Some(crate::absolutize(d));
        }
    }
    None
}

fn is_blank_text(n: &Node) -> bool {
    matches!(n, Node::Text(t) if t.trim().is_empty())
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Tiny numeric evaluator: `expr := term (('+'|'-') term)*`,
/// `term := factor (('*'|'/') factor)*`, `factor := '-' factor | number |
/// ident | '(' expr ')'`. Anything else errors (see the module docs).
struct NumParser<'a, 'e> {
    s: &'a str,
    pos: usize,
    ex: &'e Expander,
}

impl NumParser<'_, '_> {
    fn rest(&self) -> &str {
        &self.s[self.pos..]
    }
    fn skip_ws(&mut self) {
        let r = self.rest();
        self.pos += r.len() - r.trim_start().len();
    }
    fn eat(&mut self, c: char) -> bool {
        if self.rest().starts_with(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }
    fn parse(mut self) -> Result<f64, String> {
        let v = self.expr()?;
        self.skip_ws();
        if !self.rest().is_empty() {
            return Err(format!(
                "unexpected `{}` (supported: numbers, properties, pi, + - * / and parentheses)",
                self.rest()
            ));
        }
        Ok(v)
    }
    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        loop {
            self.skip_ws();
            if self.eat('+') {
                v += self.term()?;
            } else if self.eat('-') {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.factor()?;
        loop {
            self.skip_ws();
            if self.eat('*') {
                v *= self.factor()?;
            } else if self.eat('/') {
                v /= self.factor()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn factor(&mut self) -> Result<f64, String> {
        self.skip_ws();
        if self.eat('-') {
            return Ok(-self.factor()?);
        }
        if self.eat('(') {
            let v = self.expr()?;
            self.skip_ws();
            if !self.eat(')') {
                return Err("missing `)`".into());
            }
            return Ok(v);
        }
        let c = self
            .rest()
            .chars()
            .next()
            .ok_or_else(|| "unexpected end of expression".to_string())?;
        if c.is_ascii_digit() || c == '.' {
            return self.number();
        }
        if c.is_ascii_alphabetic() || c == '_' {
            return self.ident();
        }
        Err(format!(
            "unexpected `{c}` (supported: numbers, properties, pi, + - * / and parentheses)"
        ))
    }
    fn number(&mut self) -> Result<f64, String> {
        let b = self.rest().as_bytes();
        let mut end = 0;
        while end < b.len() && (b[end].is_ascii_digit() || b[end] == b'.') {
            end += 1;
        }
        if end < b.len() && (b[end] == b'e' || b[end] == b'E') {
            let mut j = end + 1;
            if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                j += 1;
            }
            let digits = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits {
                end = j;
            }
        }
        let tok = &self.rest()[..end];
        let v: f64 = tok
            .parse()
            .map_err(|_| format!("malformed number `{tok}`"))?;
        self.pos += end;
        Ok(v)
    }
    fn ident(&mut self) -> Result<f64, String> {
        let end = self
            .rest()
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(self.rest().len());
        let name = &self.rest()[..end];
        let v = match self.ex.lookup(name) {
            Some(val) => val
                .trim()
                .parse()
                .map_err(|_| format!("property `{name}` = `{val}` is not numeric"))?,
            None if name == "pi" => std::f64::consts::PI,
            None => return Err(format!("undefined property `{name}`")),
        };
        self.pos += end;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_s(s: &str) -> Result<String, XacroError> {
        expand(s, None)
    }
    fn robots_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/robots")
    }

    #[test]
    fn properties_arithmetic_and_escape() {
        let out = expand_s(
            r#"<robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro">
                 <xacro:property name="l" value="0.25"/>
                 <xacro:property name="tag" value="arm"/>
                 <link name="${tag}_base"/>
                 <joint name="j" type="revolute">
                   <origin xyz="0 0 ${l/2}"/>
                   <limit lower="${-pi/2}" upper="${(l + 0.75) * 2}" effort="1" velocity="1"/>
                 </joint>
                 <literal note="$${not_a_property}"/>
               </robot>"#,
        )
        .unwrap();
        assert!(out.contains(r#"<link name="arm_base"/>"#), "{out}");
        assert!(out.contains(r#"xyz="0 0 0.125""#), "{out}");
        assert!(out.contains(r#"lower="-1.5707963267948966""#), "{out}");
        assert!(out.contains(r#"upper="2""#), "{out}");
        assert!(
            out.contains(r#"note="${not_a_property}""#),
            "$$ escape: {out}"
        );
        assert!(!out.contains("xmlns:xacro"), "xmlns:xacro consumed: {out}");
    }

    #[test]
    fn conditionals_keep_and_drop() {
        let out = expand_s(
            r#"<robot name="r" xmlns:xacro="x">
                 <xacro:property name="flag" value="true"/>
                 <xacro:if value="${flag}"><link name="kept"/></xacro:if>
                 <xacro:unless value="${flag}"><link name="dropped"/></xacro:unless>
                 <xacro:if value="0"><link name="also_dropped"/></xacro:if>
                 <xacro:unless value="0"><link name="also_kept"/></xacro:unless>
               </robot>"#,
        )
        .unwrap();
        assert!(out.contains("kept") && out.contains("also_kept"), "{out}");
        assert!(!out.contains("dropped"), "{out}");
        let err =
            expand_s(r#"<robot xmlns:xacro="x"><xacro:if value="maybe"><a/></xacro:if></robot>"#)
                .unwrap_err();
        assert!(matches!(err, XacroError::Condition(_)), "got {err:?}");
    }

    #[test]
    fn macro_with_defaults_and_block_param() {
        let out = expand_s(
            r#"<robot name="r" xmlns:xacro="x">
                 <xacro:macro name="seg" params="name len:=0.5 *shape">
                   <link name="${name}">
                     <visual>
                       <origin xyz="0 0 ${len/2}"/>
                       <geometry><xacro:insert_block name="shape"/></geometry>
                     </visual>
                   </link>
                 </xacro:macro>
                 <xacro:seg name="a"><box size="1 1 ${1+1}"/></xacro:seg>
                 <xacro:seg name="b" len="0.2"><sphere radius="0.1"/></xacro:seg>
               </robot>"#,
        )
        .unwrap();
        assert!(out.contains(r#"<link name="a">"#), "{out}");
        assert!(out.contains(r#"xyz="0 0 0.25""#), "default len: {out}");
        assert!(
            out.contains(r#"<box size="1 1 2"/>"#),
            "block substituted: {out}"
        );
        assert!(out.contains(r#"xyz="0 0 0.1""#), "override len: {out}");
        assert!(out.contains(r#"<sphere radius="0.1"/>"#), "{out}");
    }

    #[test]
    fn macro_errors_are_specific() {
        let def = r#"<xacro:macro name="m" params="a *b"><x><xacro:insert_block name="b"/></x></xacro:macro>"#;
        // missing required value param
        let err = expand_s(&format!(
            r#"<robot xmlns:xacro="x">{def}<xacro:m><c/></xacro:m></robot>"#
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("missing required parameter `a`"),
            "{err}"
        );
        // missing block argument
        let err = expand_s(&format!(
            r#"<robot xmlns:xacro="x">{def}<xacro:m a="1"/></robot>"#
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("missing block argument `*b`"),
            "{err}"
        );
        // unexpected attribute
        let err = expand_s(&format!(
            r#"<robot xmlns:xacro="x">{def}<xacro:m a="1" zz="2"><c/></xacro:m></robot>"#
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("unexpected attribute `zz`"),
            "{err}"
        );
        // extra block children
        let err = expand_s(&format!(
            r#"<robot xmlns:xacro="x">{def}<xacro:m a="1"><c/><d/></xacro:m></robot>"#
        ))
        .unwrap_err();
        assert!(err.to_string().contains("more child elements"), "{err}");
    }

    #[test]
    fn unsupported_constructs_error_naming_the_construct() {
        let cases: &[(&str, &str)] = &[
            (r#"<xacro:arg name="v" default="1"/>"#, "xacro:arg"),
            (r#"<link name="$(arg v)"/>"#, "$(arg ..)"),
            (r#"<link name="$(env HOME)"/>"#, "$(env ..)"),
            (r#"<xacro:nonexistent_macro/>"#, "xacro:nonexistent_macro"),
            (
                r#"<xacro:property name="x" value="1" scope="parent"/>"#,
                "xacro:property",
            ),
            (r#"<xacro:include filename="f.xacro" ns="foo"/>"#, "ns="),
        ];
        for (body, needle) in cases {
            let err = expand_s(&format!(r#"<robot xmlns:xacro="x">{body}</robot>"#)).unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "error for {body} must name `{needle}`, got: {err}"
            );
        }
        // unsupported python expression inside ${..}
        let err =
            expand_s(r#"<robot xmlns:xacro="x"><link name="${cos(0)}"/></robot>"#).unwrap_err();
        assert!(matches!(err, XacroError::Expression(..)), "got {err:?}");
        // undefined property
        let err =
            expand_s(r#"<robot xmlns:xacro="x"><link name="${ghost}"/></robot>"#).unwrap_err();
        assert!(
            matches!(&err, XacroError::UndefinedProperty(p) if p == "ghost"),
            "got {err:?}"
        );
    }

    #[test]
    fn recursion_and_missing_include_are_caught() {
        let err = expand_s(
            r#"<robot xmlns:xacro="x">
                 <xacro:macro name="loopy" params=""><xacro:loopy/></xacro:macro>
                 <xacro:loopy/>
               </robot>"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("recursion depth"), "{err}");
        let err = expand_s(
            r#"<robot xmlns:xacro="x"><xacro:include filename="no_such_file.xacro"/></robot>"#,
        )
        .unwrap_err();
        assert!(matches!(err, XacroError::Include(..)), "got {err:?}");
    }

    #[test]
    fn find_resolves_via_package_search() {
        // demo_pkg sits one level above robots/ (the fixtures dir) — the same
        // ancestor walk resolve_mesh_path uses must find it here too.
        let dir = robots_dir();
        let p = resolve_package_dir("demo_pkg", Some(&dir)).expect("demo_pkg must resolve");
        assert!(p.is_dir() && p.ends_with("demo_pkg"), "got {p:?}");
        assert_eq!(resolve_package_dir("ghost_pkg", Some(&dir)), None);
        let out = expand(
            r#"<robot name="r" xmlns:xacro="x">
                 <mesh filename="$(find demo_pkg)/meshes/part.stl"/>
               </robot>"#,
            Some(&dir),
        )
        .unwrap();
        assert!(out.contains("demo_pkg/meshes/part.stl"), "{out}");
        let err = expand(
            r#"<robot xmlns:xacro="x"><mesh filename="$(find ghost_pkg)/m.stl"/></robot>"#,
            Some(&dir),
        )
        .unwrap_err();
        assert!(
            matches!(&err, XacroError::FindPackage(p) if p == "ghost_pkg"),
            "got {err:?}"
        );
    }

    #[test]
    fn detection_by_extension_and_root_xmlns() {
        let plain = r#"<robot name="r"><link name="a"/></robot>"#;
        let with_ns = r#"<robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro"/>"#;
        assert!(is_xacro(Path::new("a.xacro"), plain), "extension wins");
        assert!(is_xacro(Path::new("a.urdf.xacro"), plain));
        assert!(is_xacro(Path::new("a.urdf"), with_ns), "root xmlns");
        assert!(!is_xacro(Path::new("a.urdf"), plain));
        // a mention outside the root tag must NOT trigger expansion
        let comment = r#"<robot name="r"><!-- xmlns:xacro --><link name="a"/></robot>"#;
        assert!(!is_xacro(Path::new("a.urdf"), comment));
    }

    #[test]
    fn cdata_is_rejected_loudly() {
        let err = expand_s(r#"<robot xmlns:xacro="x"><a><![CDATA[hi]]></a></robot>"#).unwrap_err();
        assert!(err.to_string().contains("CDATA"), "{err}");
    }
}
