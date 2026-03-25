#[allow(unused_imports)]
use crate::tokenizer::Tokenizer;

#[allow(unused_imports)]
use crate::types::*;
use std::collections::HashMap;

pub struct Decoder {
    vocab_size: usize,
    hidden_size: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub last_n_tokens: usize,
}

impl Decoder {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            last_n_tokens: 64,
        }
    }

    pub fn sample(&self, logits: &[f32], _tokenizer: &Tokenizer) -> Token {
        self.sample_with_params(logits, _tokenizer, 1.0, 0.9)
    }

    pub fn sample_with_params(
        &self,
        logits: &[f32],
        _tokenizer: &Tokenizer,
        temperature: f32,
        top_p: f32,
    ) -> Token {
        let mut logits = logits.to_vec();

        if temperature > 0.0 {
            let max_logit = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            logits
                .iter_mut()
                .for_each(|l| *l = (*l - max_logit) / temperature);
            logits.iter_mut().for_each(|l| *l = l.exp());

            let sum: f32 = logits.iter().sum();
            if sum > 0.0 {
                logits.iter_mut().for_each(|l| *l /= sum);
            }
        }

        if top_p < 1.0 {
            let mut sorted: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            let cutoff = sorted
                .iter()
                .position(|(_, p)| {
                    cumsum += *p;
                    cumsum >= top_p
                })
                .unwrap_or(sorted.len());

            for (i, (_, p)) in sorted.iter().enumerate() {
                if i > cutoff {
                    logits[i] = 0.0;
                }
            }

            let sum: f32 = logits.iter().sum();
            if sum > 0.0 {
                logits.iter_mut().for_each(|l| *l /= sum);
            }
        }

        let r = rand_secure();
        let mut cumsum = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return i as Token;
            }
        }

        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        max_idx as Token
    }

    fn apply_repeat_penalty(&self, logits: &mut [f32]) {
        for logit in logits.iter_mut() {
            if *logit < 0.0 {
                *logit *= self.repeat_penalty;
            } else {
                *logit /= self.repeat_penalty;
            }
        }
    }

    fn top_k_logits(&self, logits: Vec<f32>, k: usize) -> (Vec<Token>, Vec<f32>) {
        let mut indexed_logits: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();

        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(indexed_logits.len());
        let (indices, values): (Vec<_>, Vec<_>) = indexed_logits.into_iter().take(k).unzip();

        let indices: Vec<Token> = indices.into_iter().map(|i| i as Token).collect();
        (indices, values)
    }

    fn sample_top_p(&self, indices: &[Token], probs: &[f32], top_p: f32) -> Token {
        let mut cumsum = 0.0f32;
        let cutoff = probs
            .iter()
            .position(|&p| {
                cumsum += p;
                cumsum >= top_p
            })
            .unwrap_or(probs.len());

        let cutoff = cutoff.min(probs.len().saturating_sub(1));

        let valid_probs = &probs[..=cutoff];
        let valid_indices = &indices[..=cutoff];

        let sum: f32 = valid_probs.iter().sum();
        let normalized: Vec<f32> = valid_probs.iter().map(|p| p / sum).collect();

        let sampled_idx = self.sample_multinomial(&normalized);
        valid_indices[sampled_idx]
    }

    fn sample_multinomial(&self, probs: &[f32]) -> usize {
        let r = rand_secure();
        let mut cumsum = 0.0f32;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    pub fn greedy_sample(&self, logits: &[f32]) -> Token {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as Token)
            .unwrap_or(0)
    }
}

fn rand_secure() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::Instant;

    let instant = Instant::now();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();

    let mut hasher = DefaultHasher::new();
    instant.hash(&mut hasher);
    nanos.hash(&mut hasher);

    let hash = hasher.finish();
    let value = (hash % 1000000) as f32;
    value / 1000000.0
}

pub struct BeamSearch {
    beam_width: usize,
    max_length: usize,
    length_penalty: f32,
    early_stopping: bool,
}

impl BeamSearch {
    pub fn new(beam_width: usize, max_length: usize) -> Self {
        Self {
            beam_width,
            max_length,
            length_penalty: 1.0,
            early_stopping: true,
        }
    }

    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty;
        self
    }

    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    pub fn search<F>(&self, logits_fn: F, eos_token: Token) -> Vec<(Vec<Token>, f32)>
    where
        F: Fn(&[Token]) -> Vec<f32>,
    {
        let mut beams: Vec<BeamHypothesis> = vec![BeamHypothesis {
            tokens: vec![],
            score: 0.0,
            is_complete: false,
        }];

        for _step in 0..self.max_length {
            let mut all_candidates: Vec<BeamHypothesis> = Vec::new();

            for beam in beams.iter() {
                if beam.is_complete {
                    all_candidates.push(beam.clone());
                    continue;
                }

                let logits = logits_fn(&beam.tokens);
                let top_k = self.beam_width * 2;

                let mut indexed_logits: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed_logits
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed_logits.truncate(top_k);

                for (token_id, log_prob) in indexed_logits {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id as Token);

                    let new_score = beam.score + log_prob;
                    let length_penalty = self.length_penalty(&new_tokens);

                    all_candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        score: new_score / length_penalty,
                        is_complete: token_id as Token == eos_token,
                    });
                }
            }

            all_candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_candidates.truncate(self.beam_width);

            let complete_count = all_candidates.iter().filter(|b| b.is_complete).count();
            if self.early_stopping && complete_count >= self.beam_width {
                break;
            }

            beams = all_candidates;
        }

        beams.into_iter().map(|b| (b.tokens, b.score)).collect()
    }

    fn length_penalty(&self, tokens: &[Token]) -> f32 {
        let len = tokens.len() as f32;
        if self.length_penalty == 1.0 {
            return 1.0;
        }
        ((len + 6.0) / 6.0).powf(self.length_penalty)
    }
}

#[derive(Clone)]
struct BeamHypothesis {
    tokens: Vec<Token>,
    score: f32,
    is_complete: bool,
}

pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub mirostat: bool,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            mirostat: false,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    grammar: Option<Grammar>,
    prev_tokens: Vec<Token>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self {
            params,
            grammar: None,
            prev_tokens: Vec::with_capacity(64),
        }
    }

    pub fn with_grammar(mut self, grammar: Grammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    pub fn sample(&mut self, logits: &[f32]) -> Token {
        let mut decoder = Decoder::new(logits.len(), 0);
        decoder.temperature = self.params.temperature;
        decoder.top_p = self.params.top_p;
        decoder.top_k = self.params.top_k;
        decoder.repeat_penalty = self.params.repeat_penalty;

        let token = if let Some(ref grammar) = self.grammar {
            self.sample_with_grammar(logits, grammar)
        } else {
            decoder.sample(logits, &Tokenizer::new())
        };

        if self.prev_tokens.len() >= 64 {
            self.prev_tokens.remove(0);
        }
        self.prev_tokens.push(token);

        token
    }

    fn sample_with_grammar(&self, logits: &[f32], grammar: &Grammar) -> Token {
        let mut allowed_tokens = grammar.get_allowed_tokens(&self.prev_tokens);

        if allowed_tokens.is_empty() {
            allowed_tokens = (0..logits.len()).collect();
        }

        let mut filtered_logits: Vec<(usize, f32)> =
            allowed_tokens.iter().map(|&i| (i, logits[i])).collect();

        filtered_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((token_id, _)) = filtered_logits.first() {
            *token_id as Token
        } else {
            0
        }
    }
}

pub struct Grammar {
    rules: Vec<GrammarRule>,
    start_rule: usize,
    name_to_idx: HashMap<String, usize>,
}

impl Grammar {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            start_rule: 0,
            name_to_idx: HashMap::new(),
        }
    }

    pub fn from_ggnf(ggnf: &str) -> Result<Self, String> {
        let mut grammar = Self::new();
        grammar.parse_ggnf(ggnf)?;
        Ok(grammar)
    }

    fn parse_ggnf(&mut self, ggnf: &str) -> Result<(), String> {
        let lines: Vec<&str> = ggnf.lines().collect();

        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((name_str, def)) = line.split_once("::=") {
                let name = name_str.trim().to_string();
                let rule_name = name.clone();
                let def = def.trim();

                let variants = self.parse_rule_variants(def);
                let rule_idx = self.rules.len();

                self.name_to_idx.insert(name, rule_idx);
                self.rules.push(GrammarRule {
                    name: rule_name,
                    variants,
                });
            }
        }

        if self.rules.is_empty() {
            return Err("No rules found in grammar".to_string());
        }

        Ok(())
    }

    fn parse_rule_variants(&self, def: &str) -> Vec<GrammarVariant> {
        let mut variants = Vec::new();

        for part in def.split('|') {
            let part = part.trim();
            let variant = self.parse_variant(part);
            variants.push(variant);
        }

        variants
    }

    fn parse_variant(&self, s: &str) -> GrammarVariant {
        let s = s.trim();

        if s.starts_with('"') && s.ends_with('"') {
            return GrammarVariant::Literal(s[1..s.len() - 1].to_string());
        }

        if s.starts_with('[') && s.ends_with(']') {
            let inner = &s[1..s.len() - 1];
            let tokens: Vec<usize> = inner
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            return GrammarVariant::CharClass(tokens);
        }

        if let Some(&idx) = self.name_to_idx.get(s) {
            return GrammarVariant::RuleRef(idx);
        }

        GrammarVariant::RuleRef(0)
    }

    pub fn get_allowed_tokens(&self, prev_tokens: &[Token]) -> Vec<usize> {
        let mut state = GrammarState::new(self.rules.len());

        for &token in prev_tokens.iter().rev().take(10) {
            state = self.apply_token(state, token as usize);
        }

        self.get_tokens_for_state(&state)
    }

    fn apply_token(&self, state: GrammarState, token: usize) -> GrammarState {
        let mut new_state = vec![false; self.rules.len()];

        for (rule_idx, rule) in self.rules.iter().enumerate() {
            for variant in &rule.variants {
                if self.variant_matches(variant, token) {
                    new_state[rule_idx] = true;
                }
            }
        }

        GrammarState(new_state)
    }

    fn variant_matches(&self, variant: &GrammarVariant, token: usize) -> bool {
        match variant {
            GrammarVariant::Literal(s) => s.chars().next().map(|c| c as usize) == Some(token),
            GrammarVariant::RuleRef(idx) => *idx == token,
            _ => false,
        }
    }

    fn get_tokens_for_state(&self, state: &GrammarState) -> Vec<usize> {
        let mut allowed = Vec::new();

        for (idx, is_active) in state.0.iter().enumerate() {
            if *is_active {
                if let Some(rule) = self.rules.get(idx) {
                    for variant in &rule.variants {
                        match variant {
                            GrammarVariant::Literal(s) => {
                                for c in s.chars() {
                                    allowed.push(c as usize);
                                }
                            }
                            GrammarVariant::CharClass(chars) => {
                                allowed.extend(chars.clone());
                            }
                            GrammarVariant::RuleRef(r) => {
                                allowed.push(*r);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        allowed.sort();
        allowed.dedup();
        allowed
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct GrammarState(Vec<bool>);

impl GrammarState {
    fn new(size: usize) -> Self {
        Self(vec![false; size])
    }
}

pub struct GrammarRule {
    pub name: String,
    pub variants: Vec<GrammarVariant>,
}

pub enum GrammarVariant {
    Literal(String),
    RuleRef(usize),
    Sequence(Vec<usize>),
    Choice(Vec<usize>),
    Repeat(usize, Option<usize>),
    CharClass(Vec<usize>),
}

pub fn create_json_grammar() -> Grammar {
    let ggnf = r#"
json ::= object | array | string | number | boolean | null
object ::= "{" "}" | "{" members "}"
members ::= pair | pair "," members
pair ::= string ":" value
array ::= "[" "]" | "[" elements "]"
elements ::= value | value "," elements
value ::= string | number | object | array | boolean | null
string ::= '"' characters '"'
characters ::= character characters | ""
character ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9A-Fa-f]
number ::= "-"? int frac exp
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+
boolean ::= "true" | "false"
null ::= "null"
"#;

    Grammar::from_ggnf(ggnf).unwrap_or_default()
}

pub fn create_code_grammar() -> Grammar {
    let ggnf = r#"
code ::= statement+
statement ::= expression ";" | compound_statement | function_def
compound_statement ::= "{" statement* "}"
function_def ::= "def" identifier "(" params ")" ":" statement
params ::= identifier "," params | identifier | ""
expression ::= term (("+" | "-") term)*
term ::= factor (("*" | "/" | "%") factor)*
factor ::= number | identifier | "(" expression ")" | function_call
function_call ::= identifier "(" arguments ")"
arguments ::= expression "," arguments | expression | ""
identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
number ::= [0-9]+ ("." [0-9]+)?
"#;

    Grammar::from_ggnf(ggnf).unwrap_or_default()
}
