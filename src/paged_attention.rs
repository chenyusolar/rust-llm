use std::collections::HashMap;

use crate::types::*;

const PAGE_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct Page {
    pub data: Vec<f32>,
    pub is_full: bool,
}

impl Page {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0f32; capacity],
            is_full: false,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub page_indices: Vec<usize>,
    pub block_size: usize,
}

impl Block {
    pub fn new(block_size: usize) -> Self {
        Self {
            page_indices: Vec::new(),
            block_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequencePageTable {
    pub seq_id: SeqId,
    pub blocks: Vec<Block>,
    pub num_tokens: usize,
}

impl SequencePageTable {
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            blocks: Vec::new(),
            num_tokens: 0,
        }
    }

    pub fn add_block(&mut self, page_idx: usize, block_size: usize) {
        let mut block = Block::new(block_size);
        block.page_indices.push(page_idx);
        self.blocks.push(block);
        self.num_tokens += block_size;
    }

    pub fn get_page_indices(&self) -> Vec<usize> {
        self.blocks
            .iter()
            .flat_map(|b| b.page_indices.clone())
            .collect()
    }
}

pub struct PagedAttentionCache {
    num_kv_heads: usize,
    head_dim: usize,
    max_num_pages: usize,
    page_size: usize,
    pages: Vec<Page>,
    page_tables: HashMap<SeqId, SequencePageTable>,
    free_pages: Vec<usize>,
    block_size: usize,
}

impl PagedAttentionCache {
    pub fn new(num_kv_heads: usize, head_dim: usize, max_seq_len: usize, num_pages: usize) -> Self {
        let page_size = num_kv_heads * head_dim * PAGE_SIZE;

        Self {
            num_kv_heads,
            head_dim,
            max_num_pages: num_pages,
            page_size,
            pages: Vec::with_capacity(num_pages),
            page_tables: HashMap::new(),
            free_pages: (0..num_pages).collect(),
            block_size: PAGE_SIZE,
        }
    }

    pub fn allocate_pages(&mut self, num_tokens: usize) -> Vec<usize> {
        let num_pages_needed = (num_tokens + PAGE_SIZE - 1) / PAGE_SIZE;

        let mut page_indices = Vec::new();

        for _ in 0..num_pages_needed {
            if let Some(page_idx) = self.free_pages.pop() {
                if self.pages.len() <= page_idx {
                    self.pages.resize(page_idx + 1, Page::new(self.page_size));
                }
                page_indices.push(page_idx);
            } else {
                break;
            }
        }

        page_indices
    }

    pub fn write_kv(
        &mut self,
        seq_id: SeqId,
        page_indices: &[usize],
        k_data: &[f32],
        v_data: &[f32],
    ) {
        let table = self
            .page_tables
            .entry(seq_id)
            .or_insert_with(|| SequencePageTable::new(seq_id));

        for (block_idx, &page_idx) in page_indices.iter().enumerate() {
            let token_start = block_idx * self.block_size;
            let token_end = (token_start + self.block_size).min(k_data.len() / self.num_kv_heads);

            for h in 0..self.num_kv_heads {
                for t in token_start..token_end {
                    let src_k = h * self.head_dim + t * self.num_kv_heads * self.head_dim;
                    let src_v = h * self.head_dim + t * self.num_kv_heads * self.head_dim;

                    let dst_k = page_idx * self.page_size
                        + h * self.head_dim
                        + (t - token_start) * self.num_kv_heads * self.head_dim;
                    let dst_v = page_idx * self.page_size
                        + self.page_size / 2
                        + h * self.head_dim
                        + (t - token_start) * self.num_kv_heads * self.head_dim;

                    if dst_k < self.pages[page_idx].data.len() && src_k < k_data.len() {
                        self.pages[page_idx].data[dst_k] = k_data[src_k];
                    }
                    if dst_v < self.pages[page_idx].data.len() && src_v < v_data.len() {
                        self.pages[page_idx].data[dst_v] = v_data[src_v];
                    }
                }
            }

            table.add_block(page_idx, self.block_size);
        }
    }

    pub fn read_kv(&self, seq_id: SeqId, num_tokens: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let table = self.page_tables.get(&seq_id)?;

        let mut k_data = vec![0.0f32; num_tokens * self.num_kv_heads * self.head_dim];
        let mut v_data = vec![0.0f32; num_tokens * self.num_kv_heads * self.head_dim];

        for (block_idx, block) in table.blocks.iter().enumerate() {
            for &page_idx in &block.page_indices {
                let token_start = block_idx * self.block_size;

                for h in 0..self.num_kv_heads {
                    for t in 0..self.block_size {
                        if token_start + t >= num_tokens {
                            break;
                        }

                        let src_k = page_idx * self.page_size
                            + h * self.head_dim
                            + t * self.num_kv_heads * self.head_dim;
                        let src_v = page_idx * self.page_size
                            + self.page_size / 2
                            + h * self.head_dim
                            + t * self.num_kv_heads * self.head_dim;

                        let dst_k = (token_start + t) * self.num_kv_heads * self.head_dim
                            + h * self.head_dim;
                        let dst_v = (token_start + t) * self.num_kv_heads * self.head_dim
                            + h * self.head_dim;

                        if src_k < self.pages[page_idx].data.len() {
                            k_data[dst_k..dst_k + self.head_dim].copy_from_slice(
                                &self.pages[page_idx].data[src_k..src_k + self.head_dim],
                            );
                        }
                        if src_v < self.pages[page_idx].data.len() {
                            v_data[dst_v..dst_v + self.head_dim].copy_from_slice(
                                &self.pages[page_idx].data[src_v..src_v + self.head_dim],
                            );
                        }
                    }
                }
            }
        }

        Some((k_data, v_data))
    }

    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(table) = self.page_tables.remove(&seq_id) {
            for block in &table.blocks {
                for &page_idx in &block.page_indices {
                    self.free_pages.push(page_idx);
                }
            }
        }
    }

    pub fn num_free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn get_kv_cache_size(&self) -> usize {
        self.pages.len() * self.page_size * 2
    }
}

pub struct DynamicBatcher {
    max_batch_size: usize,
    max_waiting_time_ms: u64,
    pending_requests: Vec<InferenceRequest>,
    running_requests: HashMap<SeqId, RunningRequest>,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub seq_id: SeqId,
    pub prompt_tokens: Vec<Token>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone)]
pub struct RunningRequest {
    pub seq_id: SeqId,
    pub generated_tokens: Vec<Token>,
    pub prompt_len: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl DynamicBatcher {
    pub fn new(max_batch_size: usize, max_waiting_time_ms: u64) -> Self {
        Self {
            max_batch_size,
            max_waiting_time_ms,
            pending_requests: Vec::new(),
            running_requests: HashMap::new(),
        }
    }

    pub fn add_request(&mut self, request: InferenceRequest) {
        self.pending_requests.push(request);
    }

    pub fn get_batch(&mut self) -> Vec<InferenceRequest> {
        if self.pending_requests.is_empty() {
            return Vec::new();
        }

        let batch_size = std::cmp::min(self.pending_requests.len(), self.max_batch_size);

        let batch: Vec<_> = self.pending_requests.drain(..batch_size).collect();

        for req in &batch {
            self.running_requests.insert(
                req.seq_id,
                RunningRequest {
                    seq_id: req.seq_id,
                    generated_tokens: req.prompt_tokens.clone(),
                    prompt_len: req.prompt_tokens.len(),
                    max_tokens: req.max_tokens,
                    temperature: req.temperature,
                    top_p: req.top_p,
                },
            );
        }

        batch
    }

    pub fn mark_completed(&mut self, seq_id: SeqId, generated_token: Token) {
        if let Some(req) = self.running_requests.get_mut(&seq_id) {
            req.generated_tokens.push(generated_token);

            if req.generated_tokens.len() >= req.prompt_len + req.max_tokens {
                self.running_requests.remove(&seq_id);
            }
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.pending_requests.is_empty() || !self.running_requests.is_empty()
    }

    pub fn num_running(&self) -> usize {
        self.running_requests.len()
    }
}
