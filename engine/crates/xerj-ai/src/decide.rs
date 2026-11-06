//! Local zero-shot decision head — tier 2 of the System One decide ladder.
//!
//! [`crate::neural`] answers *what is this text about* with vectors; this
//! module answers *what is this text* with probabilities over labels, in
//! process, via the same candle path. It is the rung below the history vote
//! in `xerj-api`'s decide ladder (issue #1057): when the `[decisions]`
//! history index has support the vote wins, and this head answers what the
//! history cannot — a node with no labelled history at all, or a question
//! whose payload retrieves no labelled neighbour.
//!
//! # The contract: pair scoring, not a fixed vocabulary
//!
//! A classifier head has a fixed `id2label`; decision questions arrive with
//! arbitrary labels (`noul` positive labels, `choice` criteria options), so a
//! fixed-vocabulary head cannot answer them. The head therefore loads an
//! NLI-shaped sequence-pair scorer — `id2label` must name `entailment` — and
//! scores one (premise, hypothesis) pair per candidate label, premise = the
//! payload text, hypothesis = [`hypothesis`] applied to the label. The
//! entailment probability of each hypothesis is renormalised across the
//! question's labels to a distribution that sums to 1. This is the
//! zero-shot-classification design the HF pipeline made standard, and it is
//! the shape the open `xerj-decide` model (issue #1064) is trained against:
//! v1.1 loads through this path unchanged.
//!
//! # Local files only — no egress
//!
//! Unlike [`crate::neural`], this module has **no hub download**: there is no
//! `hf-hub` dependency and no network code, so it adds nothing to the
//! published egress inventory. Weights are read from a configured directory
//! holding `config.json`, `tokenizer.json` and `model.safetensors` — the same
//! three files [`crate::neural::NeuralConfig::local_dir`] expects. A
//! deployment gets a checkpoint onto the machine the same way it gets any
//! other air-gapped asset.
//!
//! Weights load as F32 safetensors. candle 0.9 no longer ships the quantized
//! safetensors `VarBuilder` that earlier versions had (`gguf` is the only
//! quantized container left), so "quantized weights" in the issue's sense
//! would mean a gguf loader — deliberately not built here; the ≤ 300 MB
//! download budget is a property of the trained artifact (#1064), not of the
//! loader.
//!
//! Compiled only under the `decide-local` cargo feature.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBertForSequenceClassification};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

use crate::microbatch::group_by_padded_cost;

/// Cap on tokens per (premise, hypothesis) pair. Decision payloads are SMS-
/// and ticket-shaped, and every extra token is paid by every pair in the
/// batch; 512 covers the payload shapes the benchmarks use with headroom.
pub const MAX_TOKENS: usize = 512;

/// Rows in one forward pass — the same knee the neural embedder measured
/// ([`crate::neural`]): throughput climbs to 64 rows on CPU and then
/// flattens.
const MAX_BATCH_ROWS: usize = 64;

/// Ceiling on `rows × padded_sequence_length` for one forward pass, bounding
/// the activation memory one call can allocate. Same value as the neural
/// embedder's.
const PADDED_TOKEN_BUDGET: usize = 4_096;

/// How one label becomes an NLI hypothesis. Deterministic and part of the
/// model contract (#1064 trains against exactly this sentence); a template
/// is required because a bare label word is not a premise-hypothesis
/// statement an NLI head was trained to judge.
pub fn hypothesis(label: &str) -> String {
    format!("This example is {label}.")
}

/// The competing hypothesis for a `noul`: a binary question is scored as two
/// candidate statements, the positive label and its negation, so the answer
/// is a comparison rather than one unopposed score.
pub fn negation_hypothesis(label: &str) -> String {
    format!("This example is not {label}.")
}

/// How to obtain the decision model. A directory, not a hub id — see the
/// module header: this tier has no download path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecideConfig {
    /// Directory holding `config.json`, `tokenizer.json` and
    /// `model.safetensors`.
    pub model_dir: PathBuf,
}

/// One scoring request: a payload and the candidate labels to judge it
/// against. `hypotheses` are full hypothesis strings (build them with
/// [`hypothesis`] / [`negation_hypothesis`]) so the caller controls label
/// wording; this module only scores what it is handed.
#[derive(Debug, Clone)]
pub struct ScoreRequest {
    pub premise: String,
    pub hypotheses: Vec<String>,
}

/// A loaded ModernBERT-class zero-shot decision head. Cheap to share behind
/// an `Arc`; scoring takes `&self`.
pub struct DecideModel {
    model: ModernBertForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    /// Row of the classifier head that means "entailment", resolved from
    /// `id2label` at load time.
    entailment: usize,
    pad_token_id: u32,
    /// The label vocabulary, for diagnostics (`/_decide`, `/v1/models`).
    labels: Vec<String>,
}

impl DecideModel {
    /// The classifier's own label vocabulary, index-ordered.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Load the model from a local directory. **Blocking** — callers run
    /// this off the async executor (see [`DecideHandle`]).
    pub fn load(cfg: &DecideConfig) -> Result<Self> {
        let dir = &cfg.model_dir;
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");
        for (name, path) in [
            ("config.json", &config_path),
            ("tokenizer.json", &tokenizer_path),
            ("model.safetensors", &weights_path),
        ] {
            if !path.exists() {
                return Err(anyhow!(
                    "decide model dir {} is missing {name} (candle requires safetensors \
                     weights, not pytorch_model.bin)",
                    dir.display()
                ));
            }
        }

        let config_json = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read decide config {}", config_path.display()))?;
        let config: Config = serde_json::from_str(&config_json)
            .with_context(|| format!("parse decide config {}", config_path.display()))?;
        let (entailment, labels) = entailment_index(&config)
            .with_context(|| format!("decide config {}", config_path.display()))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
        // Padding is applied per batch in [`Self::score_blocking`], so that
        // one long pair cannot charge every short pair its length — the same
        // discipline as [`crate::neural`].
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS.min(config.max_position_embeddings.max(1)),
                ..Default::default()
            }))
            .map_err(|e| anyhow!("configure decide tokenizer truncation: {e}"))?;

        let device = Device::Cpu;
        #[cfg(target_os = "trueos")]
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(&weights_path)
                .with_context(|| format!("read decide weights {}", weights_path.display()))?,
            DType::F32,
            &device,
        )
        .with_context(|| format!("load decide weights {}", weights_path.display()))?;
        #[cfg(not(target_os = "trueos"))]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                DType::F32,
                &device,
            )
            .with_context(|| format!("map decide weights {}", weights_path.display()))?
        };
        let model = ModernBertForSequenceClassification::load(vb, &config)
            .map_err(|e| anyhow!("load ModernBERT classifier: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            entailment,
            pad_token_id: config.pad_token_id,
            labels,
        })
    }

    /// Score every request's hypotheses against its premise. Returns, per
    /// request, one probability per hypothesis (input order) summing to 1.
    /// **Blocking / CPU-bound** — call via `spawn_blocking`.
    ///
    /// All pairs across all requests are flattened and pushed through the
    /// model in length-homogeneous batches, so a request's cost is paid once
    /// for all of its labels.
    pub fn score_blocking(&self, requests: &[ScoreRequest]) -> Result<Vec<Vec<f32>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(requests.len()); // (start, len)
        for req in requests {
            if req.hypotheses.is_empty() {
                return Err(anyhow!(
                    "a score request with no hypotheses cannot be scored"
                ));
            }
            spans.push((pairs.len(), req.hypotheses.len()));
            for hypothesis in &req.hypotheses {
                pairs.push((req.premise.clone(), hypothesis.clone()));
            }
        }

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow!("tokenize decide pairs: {e}"))?;
        let lengths: Vec<usize> = encodings
            .iter()
            .map(|enc| enc.get_ids().len().min(MAX_TOKENS))
            .collect();

        // Entailment probability per pair, input order. With template
        // post-processing every encoding holds at least [CLS] and [SEP], so
        // a length of 0 cannot occur.
        let mut entailment_probs: Vec<f32> = vec![0.0; lengths.len()];
        for rows in group_by_padded_cost(&lengths, MAX_BATCH_ROWS, PADDED_TOKEN_BUDGET) {
            let seq_len = rows.iter().map(|&i| lengths[i]).max().unwrap_or(0);
            if seq_len == 0 {
                continue;
            }
            let probs = self.forward_padded(&encodings, &rows, &lengths, seq_len)?;
            for (row, prob) in rows.into_iter().zip(probs) {
                entailment_probs[row] = prob;
            }
        }

        // Renormalise each request's hypothesis scores to sum to 1.
        let mut out = Vec::with_capacity(requests.len());
        for (start, count) in spans {
            let scores = &entailment_probs[start..start + count];
            let total: f32 = scores.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(anyhow!(
                    "decide model returned zero entailment mass for a question's \
                     hypotheses; refusing to fabricate a uniform distribution"
                ));
            }
            out.push(scores.iter().map(|p| p / total).collect());
        }
        Ok(out)
    }

    /// Run one rectangular forward pass over `rows` and read the entailment
    /// column of the classifier's softmax. Returns one probability per row,
    /// in `rows` order.
    fn forward_padded(
        &self,
        encodings: &[Encoding],
        rows: &[usize],
        lengths: &[usize],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let batch = rows.len();
        let mut ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut mask: Vec<u32> = Vec::with_capacity(batch * seq_len);
        for &row in rows {
            let len = lengths[row];
            ids.extend_from_slice(&encodings[row].get_ids()[..len]);
            ids.resize(ids.len() + (seq_len - len), self.pad_token_id);
            mask.extend_from_slice(&encodings[row].get_attention_mask()[..len]);
            mask.resize(mask.len() + (seq_len - len), 0);
        }
        let input_ids = Tensor::from_vec(ids, (batch, seq_len), &self.device)
            .map_err(|e| anyhow!("build decide input_ids tensor: {e}"))?;
        let attention_mask = Tensor::from_vec(mask, (batch, seq_len), &self.device)
            .map_err(|e| anyhow!("build decide attention_mask tensor: {e}"))?;
        // The classifier's forward already softmaxes over its label rows.
        let probs = self
            .model
            .forward(&input_ids, &attention_mask)
            .map_err(|e| anyhow!("modernbert forward: {e}"))?;
        let entailment = probs
            .i((.., self.entailment))
            .map_err(|e| anyhow!("read entailment column: {e}"))?;
        entailment
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("read entailment probabilities: {e}"))
    }
}

/// Resolve the entailment row from `id2label`, and the label vocabulary in
/// index order. An NLI-shaped head is this module's whole contract — a
/// fixed-vocabulary classifier cannot score arbitrary labels, so anything
/// else is refused at load with the label set it found.
fn entailment_index(config: &Config) -> Result<(usize, Vec<String>)> {
    let classifier = config.classifier_config.as_ref().ok_or_else(|| {
        anyhow!("config.json has no id2label/label2id classifier block — not a classifier")
    })?;
    let mut labels: Vec<(usize, String)> = classifier
        .id2label
        .iter()
        .filter_map(|(idx, label)| idx.parse::<usize>().ok().map(|i| (i, label.clone())))
        .collect();
    labels.sort_by_key(|(idx, _)| *idx);
    if labels.is_empty() {
        return Err(anyhow!("config.json id2label is empty"));
    }
    let entailment = labels
        .iter()
        .find(|(_, label)| label.to_lowercase().contains("entailment"))
        .map(|(idx, _)| *idx)
        .ok_or_else(|| {
            let found: Vec<&str> = labels.iter().map(|(_, l)| l.as_str()).collect();
            anyhow!(
                "the decision head must be NLI-shaped with an \"entailment\" label; \
                 id2label is {found:?}"
            )
        })?;
    Ok((entailment, labels.into_iter().map(|(_, l)| l).collect()))
}

// ─────────────────────────────────────────────────────────────────────────────
// The lazy handle — the same shape as `embedder::NeuralHandle`
// ─────────────────────────────────────────────────────────────────────────────

type DecideCell = tokio::sync::OnceCell<Arc<DecideModel>>;

/// Process-scoped registry of lazily loaded decision models, so every
/// `AppState` (or test node) configured with the same directory shares one
/// loaded copy. Weak values: the registry coordinates sharing without
/// extending a model's lifetime.
fn shared_decide_cell(cfg: &DecideConfig) -> Arc<DecideCell> {
    static CELLS: OnceLock<Mutex<HashMap<DecideConfig, Weak<DecideCell>>>> = OnceLock::new();
    let mut cells = CELLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cell) = cells.get(cfg).and_then(Weak::upgrade) {
        return cell;
    }
    cells.retain(|_, cell| cell.strong_count() > 0);
    let cell = Arc::new(DecideCell::new());
    cells.insert(cfg.clone(), Arc::downgrade(&cell));
    cell
}

/// A lazily-loaded decision head. The model loads on the first score, off
/// the async executor; later calls reuse the shared `Arc`. Cheap to clone:
/// a config path and the shared cell.
#[derive(Clone)]
pub struct DecideHandle {
    cfg: DecideConfig,
    cell: Arc<DecideCell>,
}

impl DecideHandle {
    pub fn new(cfg: DecideConfig) -> Self {
        let cell = shared_decide_cell(&cfg);
        Self { cfg, cell }
    }

    /// The configured model directory, for diagnostics.
    pub fn model_dir(&self) -> &Path {
        &self.cfg.model_dir
    }

    /// Get-or-load the model. The first caller pays the blocking load;
    /// concurrent callers await the same init. A load failure is sticky for
    /// the process (the OnceCell holds the error), which is the honest
    /// behaviour for a mis-configured model directory: every request says
    /// so until the operator fixes the directory and restarts.
    async fn get(&self) -> Result<Arc<DecideModel>> {
        self.cell
            .get_or_try_init(|| async {
                let cfg = self.cfg.clone();
                let model = tokio::task::spawn_blocking(move || DecideModel::load(&cfg))
                    .await
                    .map_err(|e| anyhow!("decide model load task panicked: {e}"))??;
                tracing::info!(
                    model_dir = %self.cfg.model_dir.display(),
                    "local decision head loaded (ModernBERT-class, candle)"
                );
                Ok::<_, anyhow::Error>(Arc::new(model))
            })
            .await
            .cloned()
    }

    /// [`DecideModel::score_blocking`], off the async executor.
    pub async fn score(&self, requests: Vec<ScoreRequest>) -> Result<Vec<Vec<f32>>> {
        let model = self.get().await?;
        tokio::task::spawn_blocking(move || model.score_blocking(&requests))
            .await
            .map_err(|e| anyhow!("decide score task panicked: {e}"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir(tag: &str) -> std::path::PathBuf {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        crate::decide::testing::write_fixture(&dir)
            .unwrap_or_else(|e| panic!("write decide fixture {tag}: {e:#}"));
        dir
    }

    #[test]
    fn loads_and_scores_noul_and_choice_pairs_deterministically() {
        let model = DecideModel::load(&DecideConfig {
            model_dir: fixture_dir("roundtrip"),
        })
        .expect("load fixture");
        assert_eq!(
            model
                .labels()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["contradiction", "neutral", "entailment"],
            "fixture is NLI-shaped"
        );

        let requests = vec![
            ScoreRequest {
                premise: "urgent prize claim call now".into(),
                hypotheses: vec![hypothesis("spam"), negation_hypothesis("spam")],
            },
            ScoreRequest {
                premise: "hi it is me again about dinner".into(),
                hypotheses: vec![
                    hypothesis("billing"),
                    hypothesis("tech"),
                    hypothesis("spam"),
                ],
            },
        ];
        let first = model.score_blocking(&requests).expect("score");
        let second = model.score_blocking(&requests).expect("score again");
        assert_eq!(first.len(), 2, "one score vector per request");
        assert_eq!(first[0].len(), 2, "noul: positive + negation");
        assert_eq!(first[1].len(), 3, "choice: one hypothesis per option");
        for (i, scores) in first.iter().enumerate() {
            let sum: f32 = scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "request {i} scores sum to {sum}: {scores:?}"
            );
            assert!(
                scores.iter().all(|p| (0.0..=1.0).contains(p)),
                "probabilities in [0,1]: {scores:?}"
            );
        }
        // Same inputs, same outputs — CPU forward with fixed batching is
        // deterministic, and the tier's answers must be reproducible.
        assert_eq!(first, second, "scoring must be deterministic");
    }

    #[test]
    fn a_missing_model_file_is_named_at_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = DecideModel::load(&DecideConfig {
            model_dir: dir.path().to_path_buf(),
        })
        .err()
        .expect("load must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("config.json"), "{msg}");
    }

    #[test]
    fn a_head_without_an_entailment_label_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        crate::decide::testing::write_fixture(&dir).expect("write fixture");
        let config_path = dir.join("config.json");
        let mut config: serde_json::Value = std::fs::read_to_string(&config_path)
            .expect("read")
            .parse()
            .expect("parse");
        config["id2label"] = serde_json::json!({"0": "neg", "1": "pos"});
        config["label2id"] = serde_json::json!({"neg": "0", "pos": "1"});
        std::fs::write(&config_path, config.to_string()).expect("rewrite");
        let err = DecideModel::load(&DecideConfig { model_dir: dir })
            .err()
            .expect("load must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("entailment"), "{msg}");
        assert!(msg.contains("NLI-shaped"), "{msg}");
    }

    #[test]
    fn handles_sharing_by_config_and_stay_lazy() {
        let cfg = DecideConfig {
            model_dir: fixture_dir("sharing"),
        };
        let first = DecideHandle::new(cfg.clone());
        let second = DecideHandle::new(cfg);
        assert!(Arc::ptr_eq(&first.cell, &second.cell));
        assert!(first.cell.get().is_none(), "construction must remain lazy");
    }
}

/// Test-support: writes a tiny, deterministic, NLI-shaped ModernBERT
/// classifier into a directory. The fixture is untrained — its scores are
/// reproducible arithmetic, not judgement quality — so it exists to prove
/// the loading and scoring path end to end, never to say anything about
/// accuracy (that is `benchmarks/decisions-as-retrieval`'s job, against the
/// real trained model).
#[doc(hidden)]
pub mod testing {
    use super::*;
    use std::collections::BTreeMap;

    /// A tiny wordpiece vocabulary: specials first, then the words the tests
    /// speak. Anything else tokenises to `[UNK]`, which the model handles.
    const VOCAB: &[&str] = &[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "this",
        "example",
        "is",
        "not",
        "spam",
        "ham",
        "billing",
        "tech",
        "refund",
        "shipping",
        "money",
        "back",
        "urgent",
        "prize",
        "claim",
        "call",
        "now",
        "hi",
        "it",
        "me",
        "again",
        "about",
        "dinner",
        "message",
        "subscription",
        "cancel",
    ];

    struct TinyConfig {
        vocab_size: usize,
        hidden: usize,
        layers: usize,
        intermediate: usize,
        labels: usize,
    }

    /// Deterministic pseudo-random fill in (-0.5, 0.5): a plain LCG keeps the
    /// fixture byte-stable across runs and machines.
    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*seed >> 8) as f32 / 16_777_216.0) - 0.5
    }

    fn tensor(seed: &mut u32, shape: &[usize]) -> Result<Tensor> {
        let len: usize = shape.iter().product();
        let values: Vec<f32> = (0..len).map(|_| lcg(seed)).collect();
        Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
    }

    fn ones(len: usize) -> Result<Tensor> {
        Ok(Tensor::ones((len,), DType::F32, &Device::Cpu)?)
    }

    /// Write the fixture and return the [`DecideConfig`] naming it.
    pub fn write_fixture(dir: &Path) -> Result<DecideConfig> {
        let tiny = TinyConfig {
            vocab_size: VOCAB.len(),
            hidden: 32,
            layers: 2,
            intermediate: 64,
            labels: 3,
        };
        std::fs::create_dir_all(dir)?;

        // config.json — the classifier block rides at the top level
        // (candle's Config flattens it in).
        let id2label: BTreeMap<String, String> = [
            ("0", "contradiction"),
            ("1", "neutral"),
            ("2", "entailment"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let label2id: BTreeMap<String, String> = id2label
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();
        let config = serde_json::json!({
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
            "vocab_size": tiny.vocab_size,
            "hidden_size": tiny.hidden,
            "num_hidden_layers": tiny.layers,
            "num_attention_heads": 4,
            "intermediate_size": tiny.intermediate,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "global_attn_every_n_layers": 3,
            "global_rope_theta": 10_000.0,
            "local_attention": 128,
            "local_rope_theta": 10_000.0,
            "id2label": id2label,
            "label2id": label2id,
            "classifier_pooling": "cls",
        });
        std::fs::write(dir.join("config.json"), config.to_string())?;

        // tokenizer.json — a real wordpiece tokenizer with BERT template
        // post-processing, so pair encoding produces [CLS] a [SEP] b [SEP].
        // The vocab is fed through the standard one-token-per-line file (line
        // number = id) because that is the builder's vocab input that does
        // not depend on the hasher of a map we build ourselves.
        let vocab_path = dir.join("vocab.txt");
        std::fs::write(&vocab_path, VOCAB.join("\n") + "\n")?;
        let model = tokenizers::models::wordpiece::WordPieceBuilder::default()
            .files(vocab_path.to_string_lossy().into_owned())
            .unk_token("[UNK]".into())
            .continuing_subword_prefix("##".into())
            .build()
            .map_err(|e| anyhow!("build wordpiece: {e}"))?;
        std::fs::remove_file(&vocab_path).ok();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_normalizer(Some(tokenizers::normalizers::BertNormalizer::default()));
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::bert::BertPreTokenizer));
        let post = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .map_err(|e| anyhow!("single template: {e}"))?
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .map_err(|e| anyhow!("pair template: {e}"))?
            .special_tokens(vec![("[CLS]", 2u32), ("[SEP]", 3u32)])
            .build()
            .map_err(|e| anyhow!("template processor: {e}"))?;
        tokenizer.with_post_processor(Some(post));
        tokenizer
            .save(dir.join("tokenizer.json"), true)
            .map_err(|e| anyhow!("save tokenizer: {e}"))?;

        // model.safetensors — every tensor ModernBertForSequenceClassification
        // loads, under the names its loader asks for.
        let mut seed = 0x5eed_1234u32;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let h = tiny.hidden;
        tensors.insert(
            "model.embeddings.tok_embeddings.weight".into(),
            tensor(&mut seed, &[tiny.vocab_size, h])?,
        );
        tensors.insert("model.embeddings.norm.weight".into(), ones(h)?);
        for layer in 0..tiny.layers {
            let p = format!("model.layers.{layer}");
            tensors.insert(
                format!("{p}.attn.Wqkv.weight"),
                tensor(&mut seed, &[3 * h, h])?,
            );
            tensors.insert(format!("{p}.attn.Wo.weight"), tensor(&mut seed, &[h, h])?);
            tensors.insert(
                format!("{p}.mlp.Wi.weight"),
                tensor(&mut seed, &[2 * tiny.intermediate, h])?,
            );
            tensors.insert(
                format!("{p}.mlp.Wo.weight"),
                tensor(&mut seed, &[h, tiny.intermediate])?,
            );
            tensors.insert(format!("{p}.mlp_norm.weight"), ones(h)?);
        }
        tensors.insert("model.final_norm.weight".into(), ones(h)?);
        tensors.insert("head.dense.weight".into(), tensor(&mut seed, &[h, h])?);
        tensors.insert("head.norm.weight".into(), ones(h)?);
        tensors.insert(
            "classifier.weight".into(),
            tensor(&mut seed, &[tiny.labels, h])?,
        );
        tensors.insert("classifier.bias".into(), tensor(&mut seed, &[tiny.labels])?);
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))?;

        Ok(DecideConfig {
            model_dir: dir.to_path_buf(),
        })
    }
}
