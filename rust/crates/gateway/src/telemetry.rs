use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const LATENCY_BUCKETS_MS: &[u64] = &[
    1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000,
];

#[derive(Debug)]
struct Histogram {
    buckets: &'static [u64],
    counts: Vec<u64>,
    sum_ms: u64,
    count: u64,
}

impl Histogram {
    fn new(buckets: &'static [u64]) -> Self {
        Self {
            buckets,
            counts: vec![0; buckets.len()],
            sum_ms: 0,
            count: 0,
        }
    }

    fn observe(&mut self, value_ms: u64) {
        self.sum_ms = self.sum_ms.saturating_add(value_ms);
        self.count = self.count.saturating_add(1);
        for (index, bucket) in self.buckets.iter().enumerate() {
            if value_ms <= *bucket {
                self.counts[index] = self.counts[index].saturating_add(1);
            }
        }
    }

    fn render(&self, name: &str, help: &str, output: &mut String) {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} histogram");
        for (bucket, count) in self.buckets.iter().zip(self.counts.iter()) {
            let _ = writeln!(output, "{name}_bucket{{le=\"{bucket}\"}} {count}");
        }
        let _ = writeln!(output, "{name}_bucket{{le=\"+Inf\"}} {}", self.count);
        let _ = writeln!(output, "{name}_sum {}", self.sum_ms);
        let _ = writeln!(output, "{name}_count {}", self.count);
    }
}

/// Runtime telemetry registry.
pub struct MetricsRegistry {
    requests_total: AtomicU64,
    requests_active: AtomicU64,
    requests_failed_total: AtomicU64,
    requests_cancelled_total: AtomicU64,
    queue_rejected_total: AtomicU64,
    worker_up: AtomicBool,
    worker_restarts_total: AtomicU64,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    prompt_cache_hits_total: AtomicU64,
    prompt_cache_misses_total: AtomicU64,
    prompt_cache_cached_tokens_total: AtomicU64,
    prompt_cache_bytes: AtomicU64,
    active_batch_cache_bytes: AtomicU64,
    prompt_batch_size: AtomicU64,
    decode_batch_size: AtomicU64,
    ipc_messages_sent_total: AtomicU64,
    ipc_messages_received_total: AtomicU64,
    ipc_roundtrip_latency_ms: AtomicU64,
    worker_memory_bytes: AtomicU64,
    kv_cache_bytes: AtomicU64,
    decode_tokens_per_second: Mutex<f64>,
    prefill_tokens_per_second: Mutex<f64>,
    ttft_histogram: Mutex<Histogram>,
    request_latency_histogram: Mutex<Histogram>,
    // VLM-specific counters (Phase 8)
    vlm_requests_total: AtomicU64,
    vlm_image_count_total: AtomicU64,
    vlm_image_preprocess_latency_ms: AtomicU64,
    vlm_prompt_template_latency_ms: AtomicU64,
    vlm_load_errors_total: AtomicU64,
    labeled_counters: Mutex<HashMap<String, u64>>,
    labeled_gauges: Mutex<HashMap<String, u64>>,
}

impl MetricsRegistry {
    /// Creates a fresh metrics registry.
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_active: AtomicU64::new(0),
            requests_failed_total: AtomicU64::new(0),
            requests_cancelled_total: AtomicU64::new(0),
            queue_rejected_total: AtomicU64::new(0),
            worker_up: AtomicBool::new(false),
            worker_restarts_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            prompt_cache_hits_total: AtomicU64::new(0),
            prompt_cache_misses_total: AtomicU64::new(0),
            prompt_cache_cached_tokens_total: AtomicU64::new(0),
            prompt_cache_bytes: AtomicU64::new(0),
            active_batch_cache_bytes: AtomicU64::new(0),
            prompt_batch_size: AtomicU64::new(0),
            decode_batch_size: AtomicU64::new(0),
            ipc_messages_sent_total: AtomicU64::new(0),
            ipc_messages_received_total: AtomicU64::new(0),
            ipc_roundtrip_latency_ms: AtomicU64::new(0),
            worker_memory_bytes: AtomicU64::new(0),
            kv_cache_bytes: AtomicU64::new(0),
            decode_tokens_per_second: Mutex::new(0.0),
            prefill_tokens_per_second: Mutex::new(0.0),
            ttft_histogram: Mutex::new(Histogram::new(LATENCY_BUCKETS_MS)),
            request_latency_histogram: Mutex::new(Histogram::new(LATENCY_BUCKETS_MS)),
            vlm_requests_total: AtomicU64::new(0),
            vlm_image_count_total: AtomicU64::new(0),
            vlm_image_preprocess_latency_ms: AtomicU64::new(0),
            vlm_prompt_template_latency_ms: AtomicU64::new(0),
            vlm_load_errors_total: AtomicU64::new(0),
            labeled_counters: Mutex::new(HashMap::new()),
            labeled_gauges: Mutex::new(HashMap::new()),
        }
    }

    pub fn increment_requests_total(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_requests_active(&self) {
        self.requests_active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_requests_active(&self) {
        self.requests_active.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn increment_requests_failed_total(&self) {
        self.requests_failed_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_requests_cancelled_total(&self) {
        self.requests_cancelled_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_queue_rejected_total(&self) {
        self.queue_rejected_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_worker_up(&self, up: bool) {
        self.worker_up.store(up, Ordering::Relaxed);
    }

    #[expect(dead_code)]
    pub fn increment_worker_restarts_total(&self) {
        self.worker_restarts_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_prompt_tokens(&self, tokens: u64) {
        self.prompt_tokens_total
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn add_completion_tokens(&self, tokens: u64) {
        self.completion_tokens_total
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_prompt_cache_hit(&self, hit: bool) {
        if hit {
            self.prompt_cache_hits_total.fetch_add(1, Ordering::Relaxed);
        } else {
            self.prompt_cache_misses_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn add_prompt_cache_cached_tokens(&self, tokens: u64) {
        self.prompt_cache_cached_tokens_total
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn set_prompt_cache_bytes(&self, value: u64) {
        self.prompt_cache_bytes.store(value, Ordering::Relaxed);
    }

    pub fn set_active_batch_cache_bytes(&self, value: u64) {
        self.active_batch_cache_bytes
            .store(value, Ordering::Relaxed);
    }

    pub fn set_prompt_batch_size(&self, value: u64) {
        self.prompt_batch_size.store(value, Ordering::Relaxed);
    }

    pub fn set_decode_batch_size(&self, value: u64) {
        self.decode_batch_size.store(value, Ordering::Relaxed);
    }

    pub fn increment_ipc_messages_sent_total(&self) {
        self.ipc_messages_sent_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_ipc_messages_received_total(&self) {
        self.ipc_messages_received_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ipc_roundtrip_latency_ms(&self, value_ms: u64) {
        self.ipc_roundtrip_latency_ms
            .store(value_ms, Ordering::Relaxed);
    }

    pub fn set_worker_memory_bytes(&self, value: u64) {
        self.worker_memory_bytes.store(value, Ordering::Relaxed);
    }

    pub fn set_kv_cache_bytes(&self, value: u64) {
        self.kv_cache_bytes.store(value, Ordering::Relaxed);
    }

    pub fn record_decode_tokens_per_second(&self, value: f64) {
        if let Ok(mut guard) = self.decode_tokens_per_second.lock() {
            *guard = value;
        }
    }

    pub fn record_prefill_tokens_per_second(&self, value: f64) {
        if let Ok(mut guard) = self.prefill_tokens_per_second.lock() {
            *guard = value;
        }
    }

    pub fn record_ttft_ms(&self, value_ms: u64) {
        if let Ok(mut guard) = self.ttft_histogram.lock() {
            guard.observe(value_ms);
        }
    }

    pub fn record_request_latency_ms(&self, value_ms: u64) {
        if let Ok(mut guard) = self.request_latency_histogram.lock() {
            guard.observe(value_ms);
        }
    }

    // ── VLM metric methods (Phase 8) ──────────────────────────────────────

    /// Increments the VLM request counter.
    pub fn increment_vlm_requests_total(&self) {
        self.vlm_requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds image count for a VLM request.
    pub fn add_vlm_image_count(&self, count: u64) {
        self.vlm_image_count_total
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Records VLM image preprocessing latency in milliseconds.
    pub fn record_vlm_image_preprocess_latency_ms(&self, value_ms: u64) {
        self.vlm_image_preprocess_latency_ms
            .store(value_ms, Ordering::Relaxed);
    }

    /// Records VLM prompt/template construction latency in milliseconds.
    pub fn record_vlm_prompt_template_latency_ms(&self, value_ms: u64) {
        self.vlm_prompt_template_latency_ms
            .store(value_ms, Ordering::Relaxed);
    }

    /// Increments VLM load error counter.
    pub fn increment_vlm_load_errors_total(&self) {
        self.vlm_load_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_labeled_counter(
        &self,
        metric_name: &str,
        labels: &[(&str, &str)],
        value: u64,
    ) {
        if let Ok(mut guard) = self.labeled_counters.lock() {
            let key = metric_key(metric_name, labels);
            *guard.entry(key).or_insert(0) += value;
        }
    }

    pub fn set_labeled_gauge(&self, metric_name: &str, labels: &[(&str, &str)], value: u64) {
        if let Ok(mut guard) = self.labeled_gauges.lock() {
            let key = metric_key(metric_name, labels);
            guard.insert(key, value);
        }
    }

    /// Renders Prometheus exposition text.
    pub fn render_prometheus(&self, queue_depth: u64) -> String {
        let mut output = String::new();
        write_counter(
            &mut output,
            "mlx_requests_total",
            "Total observed requests.",
            self.requests_total.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_requests_active",
            "Current active requests.",
            self.requests_active.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_requests_failed_total",
            "Total failed requests.",
            self.requests_failed_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_requests_cancelled_total",
            "Total cancelled requests.",
            self.requests_cancelled_total.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_queue_depth",
            "Current queued requests.",
            queue_depth,
        );
        write_counter(
            &mut output,
            "mlx_queue_rejected_total",
            "Total requests rejected by queue limits.",
            self.queue_rejected_total.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_worker_up",
            "Whether worker is ready.",
            u64::from(self.worker_up.load(Ordering::Relaxed)),
        );
        write_counter(
            &mut output,
            "mlx_worker_restarts_total",
            "Reserved for worker restarts (not yet wired).",
            self.worker_restarts_total.load(Ordering::Relaxed),
        );
        if let Ok(guard) = self.ttft_histogram.lock() {
            guard.render(
                "mlx_ttft_ms",
                "Time to first token in milliseconds.",
                &mut output,
            );
        }
        if let Ok(guard) = self.request_latency_histogram.lock() {
            guard.render(
                "mlx_request_latency_ms",
                "Request latency in milliseconds.",
                &mut output,
            );
        }
        write_counter(
            &mut output,
            "mlx_prompt_tokens_total",
            "Total prompt tokens observed.",
            self.prompt_tokens_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_completion_tokens_total",
            "Total completion tokens observed.",
            self.completion_tokens_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_prompt_cache_hits_total",
            "Total prompt cache hits observed.",
            self.prompt_cache_hits_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_prompt_cache_misses_total",
            "Total prompt cache misses observed.",
            self.prompt_cache_misses_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_prompt_cache_cached_tokens_total",
            "Total prompt cache tokens reused.",
            self.prompt_cache_cached_tokens_total
                .load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_prompt_cache_bytes",
            "Latest prompt cache bytes observed.",
            self.prompt_cache_bytes.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_active_batch_cache_bytes",
            "Latest active batch cache bytes observed.",
            self.active_batch_cache_bytes.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_prompt_batch_size",
            "Latest prompt batch size observed.",
            self.prompt_batch_size.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_decode_batch_size",
            "Latest decode batch size observed.",
            self.decode_batch_size.load(Ordering::Relaxed),
        );
        write_gauge_f64(
            &mut output,
            "mlx_decode_tokens_per_second",
            "Latest decode throughput.",
            self.decode_tokens_per_second
                .lock()
                .map(|guard| *guard)
                .unwrap_or(0.0),
        );
        write_gauge_f64(
            &mut output,
            "mlx_prefill_tokens_per_second",
            "Latest prefill throughput.",
            self.prefill_tokens_per_second
                .lock()
                .map(|guard| *guard)
                .unwrap_or(0.0),
        );
        write_counter(
            &mut output,
            "mlx_ipc_messages_sent_total",
            "Total IPC messages sent.",
            self.ipc_messages_sent_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_ipc_messages_received_total",
            "Total IPC messages received.",
            self.ipc_messages_received_total.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_ipc_roundtrip_latency_ms",
            "Latest IPC roundtrip latency in milliseconds.",
            self.ipc_roundtrip_latency_ms.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_worker_memory_bytes",
            "Latest worker memory usage in bytes.",
            self.worker_memory_bytes.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_kv_cache_bytes",
            "Latest KV cache usage in bytes.",
            self.kv_cache_bytes.load(Ordering::Relaxed),
        );
        // VLM-specific metrics
        write_counter(
            &mut output,
            "mlx_vlm_requests_total",
            "Total VLM inference requests.",
            self.vlm_requests_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_vlm_image_count_total",
            "Total images processed by VLM.",
            self.vlm_image_count_total.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_vlm_image_preprocess_latency_ms",
            "Latest VLM image preprocessing latency in milliseconds.",
            self.vlm_image_preprocess_latency_ms.load(Ordering::Relaxed),
        );
        write_gauge(
            &mut output,
            "mlx_vlm_prompt_template_latency_ms",
            "Latest VLM prompt/template construction latency in milliseconds.",
            self.vlm_prompt_template_latency_ms.load(Ordering::Relaxed),
        );
        write_counter(
            &mut output,
            "mlx_vlm_load_errors_total",
            "Total VLM model load errors.",
            self.vlm_load_errors_total.load(Ordering::Relaxed),
        );
        render_labeled_metrics(
            &mut output,
            &self.labeled_counters,
            "counter",
            &[
                (
                    "mlx_requests_by_backend_stage_total",
                    "Total requests by backend and scheduler stage.",
                ),
                (
                    "mlx_cache_hits_by_backend_total",
                    "Total cache hits by backend and cache family.",
                ),
                (
                    "mlx_cache_misses_by_backend_total",
                    "Total cache misses by backend and cache family.",
                ),
                (
                    "mlx_cached_tokens_by_backend_total",
                    "Total cached tokens by backend.",
                ),
                (
                    "mlx_cache_evictions_by_backend_total",
                    "Total cache evictions by backend and cache family.",
                ),
                (
                    "mlx_cache_entries_by_backend",
                    "Latest cache entry counts by backend and cache family.",
                ),
                (
                    "mlx_worker_cancellations_by_backend_total",
                    "Total worker-side cancellations by backend.",
                ),
                (
                    "mlx_worker_errors_by_backend_total",
                    "Total worker-side errors by backend.",
                ),
                (
                    "mlx_vlm_image_count_by_backend_total",
                    "Total VLM images processed by backend.",
                ),
                (
                    "mlx_vlm_load_errors_by_backend_total",
                    "Total VLM load errors by backend.",
                ),
            ],
        );
        render_labeled_metrics(
            &mut output,
            &self.labeled_gauges,
            "gauge",
            &[
                (
                    "mlx_batch_size_by_backend",
                    "Latest batch sizes by backend and stage.",
                ),
                (
                    "mlx_latency_by_backend_ms",
                    "Latest latency measurements by backend.",
                ),
                (
                    "mlx_cache_bytes_by_backend",
                    "Latest cache bytes by backend and cache family.",
                ),
                (
                    "mlx_scheduler_tick_latency_by_backend_ms",
                    "Latest scheduler tick latency by backend.",
                ),
                (
                    "mlx_scheduler_requests_by_backend",
                    "Latest waiting and running request counts by backend.",
                ),
                (
                    "mlx_scheduled_tokens_by_backend",
                    "Latest scheduled token counts by backend and phase.",
                ),
                (
                    "mlx_arbitration_delay_by_backend_ms",
                    "Latest arbitration delay by backend.",
                ),
                (
                    "mlx_peak_memory_by_backend_bytes",
                    "Latest peak memory by backend.",
                ),
                ("mlx_apc_mode_by_backend", "Latest APC mode by backend."),
                (
                    "mlx_vlm_image_preprocess_latency_by_backend_ms",
                    "Latest VLM image preprocessing latency by backend.",
                ),
                (
                    "mlx_vlm_prompt_template_latency_by_backend_ms",
                    "Latest VLM prompt/template latency by backend.",
                ),
            ],
        );
        output
    }
}

fn metric_key(name: &str, labels: &[(&str, &str)]) -> String {
    let mut rendered = String::from(name);
    if !labels.is_empty() {
        rendered.push('{');
        for (index, (key, value)) in labels.iter().enumerate() {
            if index > 0 {
                rendered.push(',');
            }
            let _ = write!(rendered, r#"{key}="{value}""#);
        }
        rendered.push('}');
    }
    rendered
}

fn render_labeled_metrics(
    output: &mut String,
    store: &Mutex<HashMap<String, u64>>,
    metric_type: &str,
    families: &[(&str, &str)],
) {
    let Ok(guard) = store.lock() else {
        return;
    };
    for (name, help) in families {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} {metric_type}");
        for (metric_key, value) in guard.iter().filter(|(key, _)| key.starts_with(name)) {
            let _ = writeln!(output, "{metric_key} {value}");
        }
    }
}

fn write_counter(output: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} counter");
    let _ = writeln!(output, "{name} {value}");
}

fn write_gauge(output: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} gauge");
    let _ = writeln!(output, "{name} {value}");
}

fn write_gauge_f64(output: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} gauge");
    let _ = writeln!(output, "{name} {value}");
}

/// Request lifecycle tracking for structured logs and metrics.
pub struct RequestTracker {
    metrics: Arc<MetricsRegistry>,
    request_id: String,
    model: String,
    max_tokens: u32,
    stream: bool,
    queue_time_ms: u64,
    started_at: Instant,
    first_delta_at: Mutex<Option<Instant>>,
    finished: AtomicBool,
}

/// Request outcome for telemetry logging.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RequestLog {
    pub request_id: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub max_tokens: u32,
    pub stream: bool,
    pub queue_time_ms: u64,
    pub ttft_ms: Option<u64>,
    pub latency_ms: u64,
    pub completion_tokens: u64,
    pub finish_reason: Option<String>,
    pub cancelled: bool,
    pub error: Option<String>,
}

impl RequestTracker {
    /// Starts tracking a request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metrics: Arc<MetricsRegistry>,
        request_id: impl Into<String>,
        model: impl Into<String>,
        max_tokens: u32,
        stream: bool,
        queue_time_ms: u64,
        is_vlm: bool,
        image_count: u64,
    ) -> Self {
        metrics.increment_requests_total();
        metrics.increment_requests_active();
        if is_vlm {
            metrics.increment_vlm_requests_total();
            if image_count > 0 {
                metrics.add_vlm_image_count(image_count);
            }
        }
        Self {
            metrics,
            request_id: request_id.into(),
            model: model.into(),
            max_tokens,
            stream,
            queue_time_ms,
            started_at: Instant::now(),
            first_delta_at: Mutex::new(None),
            finished: AtomicBool::new(false),
        }
    }

    /// Marks first streamed delta arrival.
    pub fn record_first_delta(&self) {
        if let Ok(mut guard) = self.first_delta_at.lock() {
            if guard.is_none() {
                *guard = Some(Instant::now());
            }
        }
    }

    /// Completes request tracking and emits structured logs.
    pub fn finish(
        &self,
        prompt_tokens: u64,
        completion_tokens: u64,
        finish_reason: Option<String>,
        cancelled: bool,
        error: Option<String>,
    ) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }

        self.metrics.decrement_requests_active();
        if cancelled {
            self.metrics.increment_requests_cancelled_total();
        }
        if error.is_some() {
            self.metrics.increment_requests_failed_total();
        }
        self.metrics.add_prompt_tokens(prompt_tokens);
        self.metrics.add_completion_tokens(completion_tokens);

        let latency_ms = self.started_at.elapsed().as_millis() as u64;
        self.metrics.record_request_latency_ms(latency_ms);

        let ttft_ms = self
            .first_delta_at
            .lock()
            .ok()
            .and_then(|guard| *guard)
            .map(|first_delta| first_delta.duration_since(self.started_at).as_millis() as u64);
        if let Some(ttft_ms) = ttft_ms {
            self.metrics.record_ttft_ms(ttft_ms);
        }

        if let Some(ttft_ms) = ttft_ms {
            let prefill_ms = ttft_ms.max(1);
            self.metrics.record_prefill_tokens_per_second(
                prompt_tokens as f64 / (prefill_ms as f64 / 1000.0),
            );
        }

        let decode_ms = latency_ms.saturating_sub(ttft_ms.unwrap_or(0)).max(1);
        self.metrics.record_decode_tokens_per_second(
            completion_tokens as f64 / (decode_ms as f64 / 1000.0),
        );

        log_request(&RequestLog {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            prompt_tokens,
            max_tokens: self.max_tokens,
            stream: self.stream,
            queue_time_ms: self.queue_time_ms,
            ttft_ms,
            latency_ms,
            completion_tokens,
            finish_reason,
            cancelled,
            error,
        });
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Acquire) {
            self.metrics.decrement_requests_active();
        }
    }
}

/// Emits a structured request log line.
pub fn log_request(log: &RequestLog) {
    eprintln!("{}", render_request_log(log));
}

/// Renders a structured request log line.
pub fn render_request_log(log: &RequestLog) -> String {
    serde_json::to_string(log).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prometheus_includes_required_metric_names() {
        let metrics = MetricsRegistry::new();
        metrics.increment_requests_total();
        metrics.increment_requests_active();
        metrics.record_ttft_ms(10);
        metrics.record_request_latency_ms(20);

        let output = metrics.render_prometheus(3);

        assert!(output.contains("mlx_requests_total"));
        assert!(output.contains("mlx_requests_active"));
        assert!(output.contains("mlx_ttft_ms_bucket"));
        assert!(output.contains("mlx_request_latency_ms_bucket"));
        assert!(output.contains("mlx_ipc_messages_sent_total"));
        assert!(output.contains("mlx_kv_cache_bytes"));
        assert!(output.contains("mlx_prompt_cache_hits_total"));
        assert!(output.contains("mlx_active_batch_cache_bytes"));
    }

    #[test]
    fn render_request_log_includes_core_fields() {
        let log = RequestLog {
            request_id: "req-1".to_string(),
            model: "test-model".to_string(),
            prompt_tokens: 12,
            max_tokens: 42,
            stream: true,
            queue_time_ms: 8,
            ttft_ms: Some(16),
            latency_ms: 64,
            completion_tokens: 7,
            finish_reason: Some("stop".to_string()),
            cancelled: false,
            error: None,
        };

        let rendered = render_request_log(&log);

        assert!(rendered.contains("\"request_id\":\"req-1\""));
        assert!(rendered.contains("\"ttft_ms\":16"));
        assert!(rendered.contains("\"cancelled\":false"));
    }

    #[test]
    fn render_prometheus_includes_all_metric_families() {
        let metrics = MetricsRegistry::new();
        let output = metrics.render_prometheus(0);

        let required = [
            "mlx_requests_total 0",
            "mlx_requests_active 0",
            "mlx_requests_failed_total 0",
            "mlx_requests_cancelled_total 0",
            "mlx_queue_depth 0",
            "mlx_queue_rejected_total 0",
            "mlx_worker_up 0",
            "mlx_worker_restarts_total 0",
            "mlx_ttft_ms_bucket",
            "mlx_ttft_ms_sum",
            "mlx_ttft_ms_count",
            "mlx_request_latency_ms_bucket",
            "mlx_request_latency_ms_sum",
            "mlx_request_latency_ms_count",
            "mlx_prompt_tokens_total 0",
            "mlx_completion_tokens_total 0",
            "mlx_prompt_cache_hits_total 0",
            "mlx_prompt_cache_misses_total 0",
            "mlx_prompt_cache_cached_tokens_total 0",
            "mlx_prompt_cache_bytes 0",
            "mlx_active_batch_cache_bytes 0",
            "mlx_prompt_batch_size 0",
            "mlx_decode_batch_size 0",
            "mlx_cache_entries_by_backend",
            "mlx_cache_evictions_by_backend_total",
            "mlx_decode_tokens_per_second",
            "mlx_prefill_tokens_per_second",
            "mlx_ipc_messages_sent_total 0",
            "mlx_ipc_messages_received_total 0",
            "mlx_ipc_roundtrip_latency_ms 0",
            "mlx_worker_memory_bytes 0",
            "mlx_kv_cache_bytes 0",
            "mlx_scheduler_tick_latency_by_backend_ms",
            "mlx_arbitration_delay_by_backend_ms",
            "mlx_peak_memory_by_backend_bytes",
            "mlx_apc_mode_by_backend",
            "mlx_worker_cancellations_by_backend_total",
            "mlx_worker_errors_by_backend_total",
            // VLM Phase 8 metrics
            "mlx_vlm_requests_total 0",
            "mlx_vlm_image_count_total 0",
            "mlx_vlm_image_preprocess_latency_ms 0",
            "mlx_vlm_prompt_template_latency_ms 0",
            "mlx_vlm_load_errors_total 0",
            "mlx_vlm_image_count_by_backend_total",
            "mlx_vlm_load_errors_by_backend_total",
            "mlx_vlm_image_preprocess_latency_by_backend_ms",
            "mlx_vlm_prompt_template_latency_by_backend_ms",
        ];

        for metric in &required {
            assert!(output.contains(metric), "missing metric: {metric}");
        }
    }

    #[test]
    fn render_prometheus_includes_labeled_backend_metrics() {
        let metrics = MetricsRegistry::new();
        metrics.increment_labeled_counter(
            "mlx_requests_by_backend_stage_total",
            &[("backend", "text"), ("stage", "decoding")],
            2,
        );
        metrics.set_labeled_gauge(
            "mlx_batch_size_by_backend",
            &[("backend", "vlm"), ("stage", "prompt_processing")],
            4,
        );
        metrics.set_labeled_gauge(
            "mlx_cache_bytes_by_backend",
            &[("backend", "vlm"), ("family", "vision_feature")],
            128,
        );

        let output = metrics.render_prometheus(0);

        assert!(output.contains(
            "mlx_requests_by_backend_stage_total{backend=\"text\",stage=\"decoding\"} 2"
        ));
        assert!(output
            .contains("mlx_batch_size_by_backend{backend=\"vlm\",stage=\"prompt_processing\"} 4"));
        assert!(output
            .contains("mlx_cache_bytes_by_backend{backend=\"vlm\",family=\"vision_feature\"} 128"));
    }

    #[test]
    fn request_tracker_finish_idempotent() {
        let metrics = Arc::new(MetricsRegistry::new());
        let tracker = RequestTracker::new(
            metrics.clone(),
            "req-1",
            "test-model",
            16,
            false,
            0,
            false,
            0,
        );

        tracker.finish(10, 5, Some("stop".to_string()), false, None);
        tracker.finish(10, 5, Some("stop".to_string()), false, None);

        let output = metrics.render_prometheus(0);
        assert!(output.contains("mlx_requests_active 0"));
        assert!(output.contains("mlx_prompt_tokens_total 10"));
        assert!(output.contains("mlx_completion_tokens_total 5"));
    }

    #[test]
    fn request_tracker_drop_guard_adjusts_active() {
        let metrics = Arc::new(MetricsRegistry::new());
        {
            let _tracker = RequestTracker::new(
                metrics.clone(),
                "req-1",
                "test-model",
                16,
                false,
                0,
                false,
                0,
            );
        }

        let output = metrics.render_prometheus(0);
        assert!(output.contains("mlx_requests_active 0"));
    }

    #[test]
    fn render_request_log_includes_all_required_fields() {
        let log = RequestLog {
            request_id: "req-abc".to_string(),
            model: "test-model".to_string(),
            prompt_tokens: 12,
            max_tokens: 42,
            stream: false,
            queue_time_ms: 8,
            ttft_ms: None,
            latency_ms: 64,
            completion_tokens: 7,
            finish_reason: Some("stop".to_string()),
            cancelled: true,
            error: Some("timeout".to_string()),
        };

        let rendered = render_request_log(&log);

        assert!(rendered.contains("\"request_id\":\"req-abc\""));
        assert!(rendered.contains("\"cancelled\":true"));
        assert!(rendered.contains("\"error\":\"timeout\""));
        assert!(rendered.contains("\"latency_ms\":64"));
        assert!(rendered.contains("\"finish_reason\":\"stop\""));
        assert!(rendered.contains("\"ttft_ms\":null"));
    }
}
