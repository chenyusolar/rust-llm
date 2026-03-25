#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Batch {
    pub n_tokens: usize,
    pub tokens: Vec<Token>,
    pub positions: Vec<Pos>,
    pub seq_ids: Vec<Vec<SeqId>>,
    pub logits: Vec<bool>,
    pub embeddings: Option<Vec<f32>>,
}

impl Batch {
    pub fn new(capacity: usize) -> Self {
        Self {
            n_tokens: 0,
            tokens: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            seq_ids: Vec::new(),
            logits: Vec::with_capacity(capacity),
            embeddings: None,
        }
    }

    pub fn clear(&mut self) {
        self.n_tokens = 0;
        self.tokens.clear();
        self.positions.clear();
        self.seq_ids.clear();
        self.logits.clear();
        self.embeddings = None;
    }

    pub fn add_sequence(
        &mut self,
        seq_id: SeqId,
        tokens: &[Token],
        pos_start: Pos,
        output_logits: bool,
    ) {
        for (i, &token) in tokens.iter().enumerate() {
            self.tokens.push(token);
            self.positions.push(pos_start + i as Pos);
            self.seq_ids.push(vec![seq_id]);
            self.logits.push(output_logits && i == tokens.len() - 1);
        }
        self.n_tokens = self.tokens.len();
    }

    pub fn add_token(&mut self, token: Token, pos: Pos, seq_id: SeqId, output_logit: bool) {
        self.tokens.push(token);
        self.positions.push(pos);
        self.seq_ids.push(vec![seq_id]);
        self.logits.push(output_logit);
        self.n_tokens = self.tokens.len();
    }

    pub fn get_sequence_tokens(&self, seq_id: SeqId) -> Vec<Token> {
        self.tokens
            .iter()
            .zip(self.seq_ids.iter())
            .filter(|(_, ids)| ids.contains(&seq_id))
            .map(|(t, _)| *t)
            .collect()
    }

    pub fn num_sequences(&self) -> usize {
        let mut unique_seqs = std::collections::HashSet::new();
        for ids in &self.seq_ids {
            for &id in ids {
                unique_seqs.insert(id);
            }
        }
        unique_seqs.len()
    }
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<Token>,
    pub position: Pos,
    pub finished: bool,
    pub last_token: Token,
}

impl Sequence {
    pub fn new(id: SeqId) -> Self {
        Self {
            id,
            tokens: Vec::new(),
            position: 0,
            finished: false,
            last_token: 0,
        }
    }

    pub fn append(&mut self, token: Token) {
        self.tokens.push(token);
        self.last_token = token;
        self.position += 1;
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct BatchManager {
    max_batch_size: usize,
    max_seq_len: usize,
    sequences: HashMap<SeqId, Sequence>,
    pending_sequences: Vec<SeqId>,
    current_batch: Batch,
}

impl BatchManager {
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            max_batch_size,
            max_seq_len,
            sequences: HashMap::new(),
            pending_sequences: Vec::new(),
            current_batch: Batch::new(max_batch_size),
        }
    }

    pub fn add_sequence(&mut self, mut sequence: Sequence) -> SeqId {
        let id = sequence.id;
        self.sequences.insert(id, sequence);
        self.pending_sequences.push(id);
        id
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn remove_sequence(&mut self, seq_id: SeqId) -> Option<Sequence> {
        self.pending_sequences.retain(|&id| id != seq_id);
        self.sequences.remove(&seq_id)
    }

    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }

    pub fn prepare_batch(&mut self) -> &Batch {
        self.current_batch.clear();

        let batch_size = std::cmp::min(self.pending_sequences.len(), self.max_batch_size);

        for i in 0..batch_size {
            let seq_id = self.pending_sequences[i];
            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.position < self.max_seq_len as Pos && !seq.finished {
                    self.current_batch
                        .add_token(seq.last_token, seq.position, seq_id, true);
                }
            }
        }

        &self.current_batch
    }

    pub fn update_after_inference(&mut self, results: &[Token]) {
        for (i, &token) in results.iter().enumerate() {
            for (j, seq_id) in self.current_batch.seq_ids.iter().enumerate() {
                if j == i && !seq_id.is_empty() {
                    if let Some(seq) = self.sequences.get_mut(&seq_id[0]) {
                        seq.append(token);
                    }
                }
            }
        }

        self.pending_sequences.retain(|id| {
            if let Some(seq) = self.sequences.get(id) {
                !seq.finished && seq.position < self.max_seq_len as Pos
            } else {
                false
            }
        });
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub fn num_active_sequences(&self) -> usize {
        self.sequences.len()
    }
}
