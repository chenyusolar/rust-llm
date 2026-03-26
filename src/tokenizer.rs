use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::types::*;
use anyhow::{anyhow, Result};

#[derive(Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, Token>,
    id_to_token: Vec<String>,
    vocab_scores: Vec<f32>,
    pub vocab_size: usize,
    pub bos_token: Token,
    pub eos_token: Token,
    pub eot_token: Token,
    pub sep_token: Token,
    pub nl_token: Token,
    pub pad_token: Token,
    pub unk_token: Token,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            vocab_scores: Vec::new(),
            vocab_size: 0,
            bos_token: 1,
            eos_token: 2,
            eot_token: 2,
            sep_token: -1,
            nl_token: 13,
            pad_token: -1,
            unk_token: 0,
            add_bos: true,
            add_eos: false,
        }
    }

    pub fn load_from_gguf(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.seek(SeekFrom::End(0))?;
        tracing::debug!("Tokenizer: file_size = {}", file_size);

        let mut tokenizer = Self::new();

        let mut header = [0u8; 4];
        file.seek(SeekFrom::Start(0))?;
        if let Err(e) = file.read_exact(&mut header) {
            tracing::error!("Failed to read header: {}", e);
            return Err(anyhow::anyhow!("Failed to read header: {}", e));
        }

        let magic = u32::from_le_bytes(header);
        if magic != 0x46554747 {
            return Err(anyhow!("Not a GGUF file"));
        }

        file.seek(SeekFrom::End(-8))?;
        let mut metadata_count_bytes = [0u8; 8];
        file.read_exact(&mut metadata_count_bytes)?;
        let metadata_count = u64::from_le_bytes(metadata_count_bytes);

        file.seek(SeekFrom::Start(8))?;

        for _ in 0..metadata_count {
            let _key = read_string_gguf(&mut file)?;
            let value_type = read_u32(&mut file)?;

            match value_type {
                0 => {
                    let _ = read_u32(&mut file)?;
                }
                1 => {
                    let _ = read_u64(&mut file)?;
                }
                2 => {
                    let _ = read_f32(&mut file)?;
                }
                3 => {
                    let _ = read_bool_gguf(&mut file)?;
                }
                4 => {
                    let _ = read_string_gguf(&mut file)?;
                }
                5 => {
                    let _ = read_array_gguf(&mut file)?;
                }
                _ => {}
            }
        }

        let tensor_count = read_u32(&mut file)? as usize;
        let _alignment = read_u32(&mut file)?;

        let mut max_offset = 0usize;
        let mut vocab_size = 0;
        let mut vocab_offset = 0usize;

        for _ in 0..tensor_count {
            let name = read_string_gguf(&mut file)?;
            let n_dims = read_u32(&mut file)? as usize;
            let mut shape = Vec::new();
            for _ in 0..n_dims {
                shape.push(read_u64(&mut file)? as usize);
            }
            let dtype = read_u32(&mut file)?;
            let offset = read_u64(&mut file)? as usize;

            if name.contains("token_embd") || name.contains("embed_tokens") {
                if shape.len() >= 2 {
                    vocab_size = shape[0];
                    vocab_offset = offset;
                }
            }

            let size = shape.iter().product::<usize>() * 4;
            if offset + size > max_offset {
                max_offset = offset + size;
            }
        }

        if vocab_size == 0 {
            vocab_size = 32000;
        }

        tokenizer.vocab_size = vocab_size;

        if vocab_offset > 0 && vocab_size > 0 && vocab_offset < file_size as usize {
            if let Err(e) = file.seek(SeekFrom::Start(vocab_offset as u64)) {
                tracing::warn!("Failed to seek to vocab offset {}: {}", vocab_offset, e);
            } else {
                if let Err(e) = tokenizer.load_vocab_from_binary(&mut file, vocab_size) {
                    tracing::warn!("Failed to load vocab: {}", e);
                }
            }
        }

        if tokenizer.id_to_token.is_empty() {
            tokenizer.generate_basic_vocab();
        }

        for (i, token) in tokenizer.id_to_token.iter().enumerate() {
            tokenizer.vocab.insert(token.clone(), i as Token);
        }

        tracing::info!("Loaded tokenizer: vocab_size={}", tokenizer.vocab_size);

        Ok(tokenizer)
    }

    fn load_vocab_from_binary(&mut self, file: &mut File, vocab_size: usize) -> Result<()> {
        self.id_to_token.clear();
        self.vocab_scores.clear();

        let max_try = vocab_size.min(200000);

        tracing::debug!(
            "Loading vocabulary from binary, target size: {}",
            vocab_size
        );

        for _i in 0..max_try {
            let len = read_varint_gguf(file)?;

            if len == 0 || len > 1000 {
                break;
            }

            let mut token_bytes = vec![0u8; len];
            if file.read_exact(&mut token_bytes).is_err() {
                break;
            }

            let token = String::from_utf8_lossy(&token_bytes).to_string();

            let score_len = read_varint_gguf(file)?;
            let score = if score_len > 0 && score_len <= 10 {
                let mut score_bytes = vec![0u8; score_len];
                if file.read_exact(&mut score_bytes).is_ok() {
                    if score_len == 4 {
                        f32::from_le_bytes([
                            score_bytes[0],
                            score_bytes[1],
                            score_bytes[2],
                            score_bytes[3],
                        ])
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            self.id_to_token.push(token);
            self.vocab_scores.push(score);
        }

        tracing::debug!("Loaded {} tokens from vocabulary", self.id_to_token.len());

        while self.id_to_token.len() < vocab_size {
            self.id_to_token
                .push(format!("tok_{}", self.id_to_token.len()));
            self.vocab_scores.push(0.0);
        }

        Ok(())
    }

    fn generate_basic_vocab(&mut self) {
        self.id_to_token.clear();
        self.vocab_scores.clear();

        tracing::warn!(
            "Generating fallback vocabulary, target size: {}",
            self.vocab_size
        );

        self.id_to_token.push("<unk>".to_string());
        self.id_to_token.push("<s>".to_string());
        self.id_to_token.push("</s>".to_string());
        self.vocab_scores.push(0.0);
        self.vocab_scores.push(0.0);
        self.vocab_scores.push(0.0);

        let common_chars = " \n\t\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        for c in common_chars.chars() {
            if self.id_to_token.len() < self.vocab_size {
                self.id_to_token.push(c.to_string());
                self.vocab_scores.push(0.0);
            }
        }

        let special_tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<mask>",
            "<s>",
            "</s>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ];
        for tok in special_tokens.iter() {
            if self.id_to_token.len() < self.vocab_size
                && !self.id_to_token.contains(&tok.to_string())
            {
                self.id_to_token.push(tok.to_string());
                self.vocab_scores.push(0.0);
            }
        }

        while self.id_to_token.len() < self.vocab_size {
            self.id_to_token
                .push(format!("tok{:05}", self.id_to_token.len()));
            self.vocab_scores.push(0.0);
        }

        tracing::warn!(
            "Generated fallback vocabulary with {} tokens",
            self.id_to_token.len()
        );
    }

    pub fn encode(&self, text: &str, add_bos: Option<bool>, add_eos: Option<bool>) -> Vec<Token> {
        let mut tokens = Vec::new();

        let add_bos = add_bos.unwrap_or(self.add_bos);
        let add_eos = add_eos.unwrap_or(self.add_eos);

        if add_bos {
            tokens.push(self.bos_token);
        }

        let text_bytes = text.as_bytes();
        let mut i = 0;
        while i < text_bytes.len() {
            let (token, len) = self.longest_match(text_bytes, i);
            tokens.push(token);
            i += len.max(1);
        }

        if add_eos {
            tokens.push(self.eos_token);
        }

        tokens
    }

    fn longest_match(&self, text: &[u8], pos: usize) -> (Token, usize) {
        if self.vocab.is_empty() {
            if pos < text.len() {
                return (text[pos] as Token, 1);
            }
            return (self.unk_token, 1);
        }

        let mut best_token = self.unk_token;
        let mut best_len = 1;

        for end in (pos + 1)..=text.len().min(pos + 20) {
            let slice = &text[pos..end];
            if let Ok(s) = std::str::from_utf8(slice) {
                if let Some(&token) = self.vocab.get(s) {
                    best_token = token;
                    best_len = end - pos;
                }
            }
        }

        (best_token, best_len)
    }

    pub fn decode(&self, tokens: &[Token]) -> String {
        let mut result = String::new();

        for &token in tokens {
            if token == self.eos_token || token == self.eot_token {
                break;
            }

            if token >= 0 && (token as usize) < self.id_to_token.len() {
                let text = &self.id_to_token[token as usize];
                if !text.is_empty() {
                    result.push_str(text);
                }
            }
        }

        result
    }

    pub fn decode_token(&self, token: Token) -> Option<&str> {
        if token < 0 || (token as usize) >= self.id_to_token.len() {
            return None;
        }
        let text = &self.id_to_token[token as usize];
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    pub fn token_score(&self, token: Token) -> f32 {
        if token < 0 || (token as usize) >= self.vocab_scores.len() {
            return 0.0;
        }
        self.vocab_scores[token as usize]
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

fn read_string_gguf(file: &mut File) -> Result<String> {
    let len = read_varint_gguf(file)?;
    if len == 0 || len > 10000 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(file: &mut File) -> Result<f32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_bool_gguf(file: &mut File) -> Result<bool> {
    let mut bytes = [0u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0] != 0)
}

fn read_array_gguf(file: &mut File) -> Result<()> {
    let n_elements = read_varint_gguf(file)?;
    let _array_type = read_u32(file)?;
    for _ in 0..n_elements {
        let _ = read_u32(file)?;
    }
    Ok(())
}

fn read_varint_gguf(file: &mut File) -> Result<usize> {
    let mut result = 0;
    for i in 0..10 {
        let mut byte = [0u8; 1];
        if file.read_exact(&mut byte).is_err() {
            return Ok(result);
        }
        result |= ((byte[0] & 0x7F) as usize) << (7 * i);
        if byte[0] & 0x80 == 0 {
            return Ok(result);
        }
    }
    Ok(result)
}

pub struct Encoding {
    pub tokens: Vec<Token>,
    pub token_scores: Vec<f32>,
    pub attention_mask: Vec<i64>,
}

impl Encoding {
    pub fn new(tokens: Vec<Token>) -> Self {
        let token_scores = tokens.iter().map(|_| 0.0).collect();
        let attention_mask = tokens.iter().map(|_| 1).collect();
        Self {
            tokens,
            token_scores,
            attention_mask,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
