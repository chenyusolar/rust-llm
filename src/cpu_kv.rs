#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use std::collections::HashMap;

#[allow(unused_imports)]
use anyhow::Result;

#[derive(Clone)]
pub struct KVCacheEntry {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub position: usize,
    pub max_position: usize,
}

impl KVCacheEntry {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let capacity = max_seq_len * num_kv_heads * head_dim;
        Self {
            k_cache: vec![0.0f32; capacity],
            v_cache: vec![0.0f32; capacity],
            position: 0,
            max_position: max_seq_len,
        }
    }

    pub fn update_k(&mut self, k: &[f32], start_pos: usize, num_heads: usize, head_dim: usize) {
        let offset = start_pos * num_heads * head_dim;
        let copy_size = k.len().min(self.k_cache.len().saturating_sub(offset));

        if copy_size > 0 {
            self.k_cache[offset..offset + copy_size].copy_from_slice(&k[..copy_size]);
            self.position = start_pos;
        }
    }

    pub fn update_v(&mut self, v: &[f32], start_pos: usize, num_heads: usize, head_dim: usize) {
        let offset = start_pos * num_heads * head_dim;
        let copy_size = v.len().min(self.v_cache.len().saturating_sub(offset));

        if copy_size > 0 {
            self.v_cache[offset..offset + copy_size].copy_from_slice(&v[..copy_size]);
        }
    }

    pub fn get_k(&self, start: usize, len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        let offset = start * num_heads * head_dim;
        let size = len * num_heads * head_dim;

        if offset + size > self.k_cache.len() {
            return Vec::new();
        }

        self.k_cache[offset..offset + size].to_vec()
    }

    pub fn get_v(&self, start: usize, len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        let offset = start * num_heads * head_dim;
        let size = len * num_heads * head_dim;

        if offset + size > self.v_cache.len() {
            return Vec::new();
        }

        self.v_cache[offset..offset + size].to_vec()
    }

    pub fn clear(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.position = 0;
    }
}

pub struct CpuKvCache {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    layer_caches: Vec<Vec<KVCacheEntry>>,
    seq_to_entry: HashMap<SeqId, Vec<usize>>,
}

impl CpuKvCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        max_seqs: usize,
    ) -> Self {
        let mut layer_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let mut layer = Vec::with_capacity(max_seqs);
            for _ in 0..max_seqs {
                layer.push(KVCacheEntry::new(max_seq_len, num_kv_heads, head_dim));
            }
            layer_caches.push(layer);
        }

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            layer_caches,
            seq_to_entry: HashMap::new(),
        }
    }

    pub fn register_sequence(&mut self, seq_id: SeqId) -> Option<usize> {
        if let Some(entries) = self.seq_to_entry.get(&seq_id) {
            return Some(entries[0]);
        }

        for layer_idx in 0..self.num_layers {
            let entry_idx = self.seq_to_entry.len();
            if entry_idx >= self.layer_caches[0].len() {
                return None;
            }

            self.seq_to_entry
                .entry(seq_id)
                .or_insert_with(Vec::new)
                .push(entry_idx);
        }

        self.seq_to_entry.get(&seq_id).map(|v| v[0])
    }

    pub fn unregister_sequence(&mut self, seq_id: SeqId) {
        if let Some(entries) = self.seq_to_entry.remove(&seq_id) {
            for &entry_idx in &entries {
                for layer in &mut self.layer_caches {
                    if entry_idx < layer.len() {
                        layer[entry_idx].clear();
                    }
                }
            }
        }
    }

    pub fn update_k(&mut self, layer_idx: usize, seq_id: SeqId, k: &[f32], start_pos: usize) {
        if let Some(entries) = self.seq_to_entry.get(&seq_id) {
            if let Some(&entry_idx) = entries.get(layer_idx) {
                if layer_idx < self.layer_caches.len()
                    && entry_idx < self.layer_caches[layer_idx].len()
                {
                    self.layer_caches[layer_idx][entry_idx].update_k(
                        k,
                        start_pos,
                        self.num_kv_heads,
                        self.head_dim,
                    );
                }
            }
        }
    }

    pub fn update_v(&mut self, layer_idx: usize, seq_id: SeqId, v: &[f32], start_pos: usize) {
        if let Some(entries) = self.seq_to_entry.get(&seq_id) {
            if let Some(&entry_idx) = entries.get(layer_idx) {
                if layer_idx < self.layer_caches.len()
                    && entry_idx < self.layer_caches[layer_idx].len()
                {
                    self.layer_caches[layer_idx][entry_idx].update_v(
                        v,
                        start_pos,
                        self.num_kv_heads,
                        self.head_dim,
                    );
                }
            }
        }
    }

    pub fn get_k(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        if let Some(entries) = self.seq_to_entry.get(&seq_id) {
            if let Some(&entry_idx) = entries.get(layer_idx) {
                if layer_idx < self.layer_caches.len()
                    && entry_idx < self.layer_caches[layer_idx].len()
                {
                    return self.layer_caches[layer_idx][entry_idx].get_k(
                        start,
                        len,
                        self.num_kv_heads,
                        self.head_dim,
                    );
                }
            }
        }
        Vec::new()
    }

    pub fn get_v(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        if let Some(entries) = self.seq_to_entry.get(&seq_id) {
            if let Some(&entry_idx) = entries.get(layer_idx) {
                if layer_idx < self.layer_caches.len()
                    && entry_idx < self.layer_caches[layer_idx].len()
                {
                    return self.layer_caches[layer_idx][entry_idx].get_v(
                        start,
                        len,
                        self.num_kv_heads,
                        self.head_dim,
                    );
                }
            }
        }
        Vec::new()
    }

    pub fn memory_usage(&self) -> usize {
        let per_entry = self.num_kv_heads * self.head_dim * self.max_seq_len * 2 * 4;
        let total_entries = self.layer_caches.len() * self.layer_caches[0].len();
        per_entry * total_entries
    }
}

pub struct GpuKvCache {
    device_id: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    max_seqs: usize,
    gpu_ptrs: Vec<u64>,
}

impl GpuKvCache {
    pub fn new(
        device_id: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        max_seqs: usize,
    ) -> Self {
        Self {
            device_id,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            max_seqs,
            gpu_ptrs: Vec::new(),
        }
    }

    pub fn allocate(&mut self) -> Result<()> {
        tracing::warn!("GPU KV cache allocation skipped (GPU not available)");
        Ok(())
    }

    pub fn memory_needed(&self) -> usize {
        let per_entry = self.num_kv_heads * self.head_dim * self.max_seq_len * 2 * 4;
        let total = self.num_layers * self.max_seqs;
        per_entry * total
    }

    pub fn can_fit_in_vram(&self, vram_available: usize) -> bool {
        self.memory_needed() <= vram_available
    }
}
