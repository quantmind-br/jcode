//! Cross-provider activity ledger.
//!
//! Tracks two things per login/credential ("source key"):
//!   1. When jcode last successfully used it (for recency-sorted `/usage`).
//!   2. Locally accumulated API-key spend in USD (day / month / all-time),
//!      mirroring the dollar figures the TUI cost paths compute, since most
//!      providers do not expose per-key spend through their public APIs.
//!
//! Data persists to `~/.jcode/provider_activity.json` and is shared across
//! processes (server records last-used, TUI records spend, `/usage` reads
//! both), so queries re-read the file with a short TTL instead of trusting a
//! process-local cache.
//!
//! Source key conventions:
//!   - `claude:oauth:<label>` / `claude:api-key`
//!   - `openai:oauth:<label>` / `openai:api-key`
//!   - `openai-compatible:<profile-id>` (DeepSeek, Moonshot, NVIDIA NIM, ...)
//!   - `named-profile:<escaped-profile-name>` (configured `[providers.*]`
//!     profiles; the escaped name contains no credential data)
//!   - `openrouter`, `jcode`, `copilot`, `gemini`, `cursor`, `bedrock`,
//!     `antigravity`, `azure-openai`

use chrono::{Datelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Re-reads of the ledger are throttled to this interval for query paths.
const QUERY_RELOAD_TTL: Duration = Duration::from_secs(2);

/// Skip persisting a new last-used timestamp when the stored one is within
/// this many seconds, so busy sessions do not rewrite the file on every call.
const LAST_USED_WRITE_THROTTLE_SECS: u64 = 30;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderSpend {
    /// `YYYY-MM-DD` the `day_usd` bucket belongs to.
    #[serde(default)]
    pub day_date: String,
    #[serde(default)]
    pub day_usd: f64,
    /// `YYYY-MM` the `month_usd` bucket belongs to.
    #[serde(default)]
    pub month: String,
    #[serde(default)]
    pub month_usd: f64,
    #[serde(default)]
    pub all_time_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderActivityEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_unix_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend: Option<ProviderSpend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderActivityStore {
    #[serde(default)]
    pub entries: HashMap<String, ProviderActivityEntry>,
}

/// Durable billing/source identity for the active session route.
///
/// The session's `provider_key` and `route_api_method` are the authoritative
/// route metadata.  The display provider and process environment are not part
/// of this type's resolution path; callers may use the old label/env helper
/// only when this metadata is unavailable for a legacy session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSourceIdentity {
    pub source_key: String,
    pub is_anthropic: bool,
    pub is_openai: bool,
    pub is_metered: bool,
}

struct CachedStore {
    loaded_at: Instant,
    store: ProviderActivityStore,
}

static LEDGER: Mutex<Option<CachedStore>> = Mutex::new(None);

fn ledger_path() -> PathBuf {
    crate::storage::jcode_dir()
        .unwrap_or_else(|_| PathBuf::from(".").join(".jcode"))
        .join("provider_activity.json")
}

fn load_store() -> ProviderActivityStore {
    // Serde defaults keep the original ledger format readable. Do not merge
    // legacy display-label keys into new profile keys: labels are not unique
    // identities and such a merge could attribute spend to the wrong profile.
    crate::storage::read_json(&ledger_path()).unwrap_or_default()
}

fn save_store(store: &ProviderActivityStore) {
    let _ = crate::storage::write_json(&ledger_path(), store);
}

fn now_unix_secs() -> u64 {
    Utc::now().timestamp().max(0) as u64
}

fn roll_spend(spend: &mut ProviderSpend) {
    let now = Utc::now();
    let today = now.format("%Y-%m-%d").to_string();
    let month = format!("{}-{:02}", now.year(), now.month());
    if spend.day_date != today {
        spend.day_date = today;
        spend.day_usd = 0.0;
    }
    if spend.month != month {
        spend.month = month;
        spend.month_usd = 0.0;
    }
}

/// Run `mutate` against a freshly loaded copy of the ledger and persist it.
/// Returns without writing when `mutate` reports no change.
fn with_fresh_store(mutate: impl FnOnce(&mut ProviderActivityStore) -> bool) {
    let mut guard = match LEDGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    // Always merge against the on-disk state so concurrent writers (server
    // last-used vs TUI spend) do not clobber each other's entries.
    let mut store = load_store();
    if mutate(&mut store) {
        save_store(&store);
    }
    *guard = Some(CachedStore {
        loaded_at: Instant::now(),
        store,
    });
}

fn snapshot_entry(source_key: &str) -> Option<ProviderActivityEntry> {
    let mut guard = match LEDGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let needs_reload = guard
        .as_ref()
        .map(|cached| cached.loaded_at.elapsed() > QUERY_RELOAD_TTL)
        .unwrap_or(true);
    if needs_reload {
        *guard = Some(CachedStore {
            loaded_at: Instant::now(),
            store: load_store(),
        });
    }
    guard
        .as_ref()
        .and_then(|cached| cached.store.entries.get(source_key).cloned())
}

/// Record a successful use of a login/credential right now.
pub fn record_use(source_key: &str) {
    let source_key = source_key.trim();
    if source_key.is_empty() {
        return;
    }
    let now = now_unix_secs();
    let source_key = source_key.to_string();
    with_fresh_store(move |store| {
        let entry = store.entries.entry(source_key).or_default();
        let throttled = entry
            .last_used_unix_secs
            .map(|prev| now.saturating_sub(prev) < LAST_USED_WRITE_THROTTLE_SECS)
            .unwrap_or(false);
        if throttled {
            return false;
        }
        entry.last_used_unix_secs = Some(now);
        true
    });
}

/// Accumulate locally computed API-key spend (in USD) for a credential.
pub fn record_spend(source_key: &str, usd: f64) {
    let source_key = source_key.trim();
    if source_key.is_empty() || !usd.is_finite() || usd <= 0.0 {
        return;
    }
    let now = now_unix_secs();
    let source_key = source_key.to_string();
    with_fresh_store(move |store| {
        let entry = store.entries.entry(source_key).or_default();
        // Spend implies use; keep recency in the same write.
        entry.last_used_unix_secs = Some(now);
        let spend = entry.spend.get_or_insert_with(ProviderSpend::default);
        roll_spend(spend);
        spend.day_usd += usd;
        spend.month_usd += usd;
        spend.all_time_usd += usd;
        true
    });
}

pub fn last_used_unix_secs(source_key: &str) -> Option<u64> {
    snapshot_entry(source_key)?.last_used_unix_secs
}

/// Spend snapshot with day/month buckets rolled to the current date.
pub fn spend_snapshot(source_key: &str) -> Option<ProviderSpend> {
    let mut spend = snapshot_entry(source_key)?.spend?;
    roll_spend(&mut spend);
    Some(spend)
}

/// All ledger entries (source key -> activity), with spend buckets rolled.
/// Used by `/usage` to surface logins that have been used but have no
/// dedicated usage fetcher (Cursor, Bedrock, Azure, ...).
pub fn all_entries() -> Vec<(String, ProviderActivityEntry)> {
    let mut guard = match LEDGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let needs_reload = guard
        .as_ref()
        .map(|cached| cached.loaded_at.elapsed() > QUERY_RELOAD_TTL)
        .unwrap_or(true);
    if needs_reload {
        *guard = Some(CachedStore {
            loaded_at: Instant::now(),
            store: load_store(),
        });
    }
    let Some(cached) = guard.as_ref() else {
        return Vec::new();
    };
    let mut entries: Vec<(String, ProviderActivityEntry)> = cached
        .store
        .entries
        .iter()
        .map(|(key, entry)| {
            let mut entry = entry.clone();
            if let Some(spend) = entry.spend.as_mut() {
                roll_spend(spend);
            }
            (key.clone(), entry)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Human-facing display name for a ledger source key, e.g.
/// `openai-compatible:deepseek` -> `DeepSeek (API key)`,
/// `claude:oauth:claude-1` -> `Anthropic (Claude) [claude-1]`.
pub fn display_name_for_source_key(source_key: &str) -> String {
    if let Some(encoded_name) = source_key.strip_prefix("named-profile:") {
        return format!("{} (API key)", decode_source_component(encoded_name));
    }
    if let Some(profile_id) = source_key.strip_prefix("openai-compatible:") {
        let name = crate::provider_catalog::openai_compatible_profile_by_id(profile_id)
            .map(|profile| profile.display_name.to_string())
            .unwrap_or_else(|| profile_id.to_string());
        return format!("{} (API key)", name);
    }
    if let Some(label) = source_key.strip_prefix("claude:oauth:") {
        return format!("Anthropic (Claude) [{}]", label);
    }
    if let Some(label) = source_key.strip_prefix("openai:oauth:") {
        return format!("OpenAI (ChatGPT) [{}]", label);
    }
    match source_key {
        "claude:api-key" => "Anthropic API key".to_string(),
        "openai:api-key" => "OpenAI API key".to_string(),
        "openrouter" => "OpenRouter".to_string(),
        "jcode" => "Jcode subscription".to_string(),
        "copilot" => "GitHub Copilot".to_string(),
        "gemini" => "Google Gemini".to_string(),
        "cursor" => "Cursor".to_string(),
        "bedrock" => "AWS Bedrock".to_string(),
        "antigravity" => "Antigravity".to_string(),
        "azure-openai" => "Azure OpenAI".to_string(),
        other => {
            // Slug -> Title Case fallback.
            other
                .split('-')
                .filter(|part| !part.is_empty())
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        Some(first) => first.to_uppercase().to_string() + chars.as_str(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

fn encode_source_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-') {
            encoded.push(*byte as char);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{:02X}", byte));
        }
    }
    encoded
}

fn decode_source_component(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) =
                (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
        {
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn named_profile_source_key(profile_name: &str) -> Option<String> {
    let profile_name = profile_name.trim();
    (!profile_name.is_empty())
        .then(|| format!("named-profile:{}", encode_source_component(profile_name)))
}

fn runtime_source_key(runtime_provider: &str) -> Option<String> {
    let runtime = runtime_provider.trim().to_ascii_lowercase();
    if runtime.is_empty() {
        return None;
    }
    match runtime.as_str() {
        "jcode" | "openrouter" | "azure-openai" | "bedrock" | "copilot" | "gemini" | "cursor"
        | "antigravity" => Some(runtime),
        "claude" | "claude-oauth" => Some("claude:oauth:default".to_string()),
        "claude-api" | "anthropic-api" => Some("claude:api-key".to_string()),
        "openai" | "openai-oauth" => Some("openai:oauth:default".to_string()),
        "openai-api" => Some("openai:api-key".to_string()),
        "openai-compatible" => None,
        other => crate::provider_catalog::openai_compatible_profile_by_id(other)
            .map(|_| format!("openai-compatible:{other}")),
    }
}

fn configured_named_profile_source_key(profile_name: &str) -> Option<String> {
    let profile_name = profile_name.trim();
    if profile_name.is_empty() {
        return None;
    }
    crate::config::config()
        .providers
        .keys()
        .find(|name| name.eq_ignore_ascii_case(profile_name))
        .and_then(|name| named_profile_source_key(name))
}

fn route_component(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

fn route_provider_component(canonical_provider_model_identity: Option<&str>) -> Option<&str> {
    let identity = canonical_provider_model_identity?.trim();
    let (provider, _) = identity.split_once('/')?;
    route_component(provider)
}

fn named_profile_source_key_for_route_component(component: &str) -> Option<String> {
    let component = component.trim();
    let component = component
        .strip_prefix("openai-compatible:")
        .map(str::trim)
        .filter(|component| !component.is_empty())
        .unwrap_or(component);
    if component.is_empty() {
        return None;
    }
    configured_named_profile_source_key(component).or_else(|| {
        // A structured `openai-compatible:<id>` route can refer to a user
        // profile that is not present in this process's config snapshot (for
        // example, a remote session). Keep that identity distinct from native
        // OpenAI and public OpenRouter rather than guessing from its label.
        (crate::provider_catalog::openai_compatible_profile_by_id(component).is_none())
            .then(|| named_profile_source_key(component))
            .flatten()
    })
}

fn route_provider_key_is_builtin(provider_key: &str) -> bool {
    let provider_key = provider_key.trim();
    let lower = provider_key.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "jcode"
            | "jcode-subscription"
            | "openrouter"
            | "openai-compatible"
            | "copilot"
            | "gemini"
            | "code-assist-oauth"
            | "cursor"
            | "bedrock"
            | "antigravity"
            | "https"
            | "azure-openai"
    ) || jcode_provider_core::AuthRoute::parse(provider_key).is_some()
        || crate::provider_catalog::openai_compatible_profile_by_id(&lower).is_some()
        || lower
            .strip_prefix("openai-compatible:")
            .is_some_and(|profile_id| {
                crate::provider_catalog::openai_compatible_profile_by_id(profile_id).is_some()
            })
}

fn route_source_identity_from_key(source_key: String) -> RouteSourceIdentity {
    let is_anthropic = source_key == "claude:api-key" || source_key.starts_with("claude:oauth:");
    let is_openai = source_key == "openai:api-key" || source_key.starts_with("openai:oauth:");
    let is_metered = if let Some(profile_id) = source_key.strip_prefix("openai-compatible:") {
        crate::provider_catalog::openai_compatible_profile_by_id(profile_id)
            .map(|profile| profile.requires_api_key)
            .unwrap_or(true)
    } else if let Some(encoded_name) = source_key.strip_prefix("named-profile:") {
        let profile_name = decode_source_component(encoded_name);
        crate::config::config()
            .providers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(&profile_name))
            .map(|(_, profile)| {
                profile.requires_api_key.unwrap_or(!matches!(
                    profile.auth,
                    crate::config::NamedProviderAuth::None
                ))
            })
            .unwrap_or(true)
    } else {
        match source_key.as_str() {
            "claude:api-key" | "openai:api-key" | "openrouter" | "bedrock" | "azure-openai" => true,
            "jcode" | "copilot" | "gemini" | "cursor" | "antigravity" => false,
            key if key.starts_with("claude:oauth:") || key.starts_with("openai:oauth:") => false,
            // An explicit route unknown to this version is safer as a metered
            // route than as a subscription route.  This branch is only reached
            // after structured metadata identified a route.
            _ => true,
        }
    };
    RouteSourceIdentity {
        source_key,
        is_anthropic,
        is_openai,
        is_metered,
    }
}

/// Resolve the ledger/pricing source from durable session route metadata.
///
/// `canonical_provider_model_identity` is the canonical `provider/model`
/// session identity when available.  It is used as a tie-breaker for named
/// profiles because the OpenRouter-compatible transport is shared by public
/// OpenRouter, built-in compatible profiles, and user-defined profiles.
///
/// This function deliberately does not inspect `JCODE_NAMED_PROVIDER_PROFILE`,
/// `JCODE_RUNTIME_PROVIDER`, cache namespaces, or display labels.  A `None`
/// result means the caller is handling a legacy session without enough route
/// metadata and may use its documented compatibility fallback.
pub fn source_identity_for_route_metadata(
    provider_key: Option<&str>,
    route_api_method: Option<&str>,
    canonical_provider_model_identity: Option<&str>,
) -> Option<RouteSourceIdentity> {
    let provider_key = provider_key.and_then(route_component);
    let route_api_method = route_api_method.and_then(route_component);
    let identity_provider = route_provider_component(canonical_provider_model_identity);

    // A named profile must win over the shared OpenRouter transport. Check
    // every durable identity spelling before interpreting transport tokens,
    // but never override an explicit dual-auth route with a coincidentally
    // named config profile such as `[providers.openai]`.
    if route_api_method
        .and_then(jcode_provider_core::AuthRoute::parse)
        .is_none()
    {
        for component in [provider_key, identity_provider]
            .into_iter()
            .flatten()
            .flat_map(|value| {
                let profile_id = value
                    .strip_prefix("openai-compatible:")
                    .map(str::trim)
                    .filter(|profile_id| !profile_id.is_empty());
                [Some(value), profile_id]
            })
            .flatten()
        {
            if !route_provider_key_is_builtin(component)
                && let Some(source_key) = configured_named_profile_source_key(component)
            {
                return Some(route_source_identity_from_key(source_key));
            }
        }
    }

    // `openrouter` is also the transport slot for named profiles. When a
    // session has a non-built-in provider key, that durable key wins even if a
    // legacy route method only describes the shared transport.
    if let Some(provider_key) = provider_key
        && !route_provider_key_is_builtin(provider_key)
        && let Some(source_key) = named_profile_source_key_for_route_component(provider_key)
    {
        return Some(route_source_identity_from_key(source_key));
    }

    if let Some(identity_provider) = identity_provider
        && !route_provider_key_is_builtin(identity_provider)
        && let Some(source_key) = named_profile_source_key_for_route_component(identity_provider)
    {
        return Some(route_source_identity_from_key(source_key));
    }

    if let Some(method) = route_api_method
        && let Some(profile_id) = method
            .strip_prefix("openai-compatible:")
            .map(str::trim)
            .filter(|profile_id| !profile_id.is_empty())
    {
        if let Some(source_key) = named_profile_source_key_for_route_component(profile_id) {
            return Some(route_source_identity_from_key(source_key));
        }
        return Some(route_source_identity_from_key(format!(
            "openai-compatible:{profile_id}"
        )));
    }

    // A bare compatible route still needs the provider key to distinguish a
    // configured profile from the public OpenRouter slot.
    if route_api_method == Some("openai-compatible")
        && let Some(provider_key) = provider_key
    {
        if let Some(profile_id) = provider_key.strip_prefix("openai-compatible:") {
            if let Some(source_key) = named_profile_source_key_for_route_component(profile_id) {
                return Some(route_source_identity_from_key(source_key));
            }
            return Some(route_source_identity_from_key(format!(
                "openai-compatible:{}",
                profile_id.trim()
            )));
        }
        if let Some(source_key) = named_profile_source_key_for_route_component(provider_key) {
            return Some(route_source_identity_from_key(source_key));
        }
    }

    // Known OpenAI-compatible catalog profiles have their own pricing/ledger
    // bucket even though their requests use the shared OpenRouter-capable
    // transport. A provider-level switch can persist the bare profile id,
    // while picker selections persist `openai-compatible:<id>`.
    if route_api_method != Some("openrouter")
        && let Some(provider_key) = provider_key
    {
        let profile_id = provider_key
            .strip_prefix("openai-compatible:")
            .map(str::trim)
            .filter(|profile_id| !profile_id.is_empty())
            .unwrap_or(provider_key);
        if crate::provider_catalog::openai_compatible_profile_by_id(profile_id).is_some() {
            return Some(route_source_identity_from_key(format!(
                "openai-compatible:{profile_id}"
            )));
        }
    }

    let auth_route = route_api_method.and_then(jcode_provider_core::AuthRoute::parse);
    if let Some(route) = auth_route {
        let source_key = match route {
            jcode_provider_core::AuthRoute {
                provider: jcode_provider_core::DualAuthProvider::Anthropic,
                mode: jcode_provider_core::AuthMode::Oauth,
            } => "claude:oauth:default",
            jcode_provider_core::AuthRoute {
                provider: jcode_provider_core::DualAuthProvider::Anthropic,
                mode: jcode_provider_core::AuthMode::ApiKey,
            } => "claude:api-key",
            jcode_provider_core::AuthRoute {
                provider: jcode_provider_core::DualAuthProvider::OpenAI,
                mode: jcode_provider_core::AuthMode::Oauth,
            } => "openai:oauth:default",
            jcode_provider_core::AuthRoute {
                provider: jcode_provider_core::DualAuthProvider::OpenAI,
                mode: jcode_provider_core::AuthMode::ApiKey,
            } => "openai:api-key",
        };
        return Some(route_source_identity_from_key(source_key.to_string()));
    }

    let method = route_api_method.map(str::to_ascii_lowercase);
    let provider_key = provider_key.map(str::to_ascii_lowercase);
    let identity_provider = identity_provider.map(str::to_ascii_lowercase);
    let token = method
        .as_deref()
        .or(provider_key.as_deref())
        .or(identity_provider.as_deref())?;

    let source_key = match token {
        "jcode" | "jcode-subscription" => "jcode".to_string(),
        "openrouter" => "openrouter".to_string(),
        "copilot" => "copilot".to_string(),
        "gemini" | "code-assist-oauth" => "gemini".to_string(),
        "cursor" => "cursor".to_string(),
        "bedrock" => "bedrock".to_string(),
        "antigravity" | "https" => "antigravity".to_string(),
        "openai-compatible" => "openai-compatible:openai-compatible".to_string(),
        "openai" | "openai-api" | "openai-api-key" => "openai:api-key".to_string(),
        "openai-oauth" => "openai:oauth:default".to_string(),
        "claude" | "anthropic" | "claude-api" | "anthropic-api" => "claude:api-key".to_string(),
        "claude-oauth" => "claude:oauth:default".to_string(),
        value if value.starts_with("openai-compatible:") => {
            let profile_id = value.trim_start_matches("openai-compatible:").trim();
            if profile_id.is_empty() {
                return None;
            }
            format!("openai-compatible:{profile_id}")
        }
        _ => named_profile_source_key_for_route_component(token)?,
    };
    Some(route_source_identity_from_key(source_key))
}

/// Human-readable relative age such as `just now`, `5m ago`, `3h ago`, `2d ago`.
pub fn format_relative_age(unix_secs: u64) -> String {
    let secs = now_unix_secs().saturating_sub(unix_secs);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3_600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        let hours = secs / 3_600;
        let minutes = (secs % 3_600) / 60;
        if minutes > 0 {
            format!("{}h {}m ago", hours, minutes)
        } else {
            format!("{}h ago", hours)
        }
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

/// Map a human-facing provider label (e.g. `"DeepSeek"`, `"OpenRouter"`,
/// `"NVIDIA NIM"`) plus the optional `JCODE_RUNTIME_PROVIDER` key onto a
/// ledger source key. Used by spend recorders that only know display names.
pub fn source_key_for_provider_label(label: &str, runtime_provider: Option<&str>) -> String {
    let normalized = label.trim().to_ascii_lowercase();
    let runtime = runtime_provider
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());

    if let Ok(profile_name) = std::env::var("JCODE_NAMED_PROVIDER_PROFILE")
        && let Some(source_key) = named_profile_source_key(&profile_name)
    {
        return source_key;
    }

    if let Some(runtime) = runtime.as_deref()
        && let Some(source_key) = runtime_source_key(runtime)
    {
        return source_key;
    }

    if let Some(runtime) = runtime.as_deref()
        && let Some(source_key) = configured_named_profile_source_key(runtime)
    {
        return source_key;
    }

    if runtime.as_deref() == Some("openai-compatible")
        && let Ok(namespace) = std::env::var("JCODE_OPENROUTER_CACHE_NAMESPACE")
    {
        let namespace = namespace.trim();
        if !namespace.is_empty() {
            if crate::provider_catalog::openai_compatible_profile_by_id(namespace).is_some() {
                return format!("openai-compatible:{}", namespace.to_ascii_lowercase());
            }
            if let Some(source_key) = named_profile_source_key(namespace) {
                return source_key;
            }
        }
    }

    // OpenRouter first: the catalog also carries an `openrouter` compatible
    // profile, but the ledger treats the public aggregator as its own bucket.
    if normalized.contains("openrouter") {
        // The OpenRouter slot multiplexes direct profiles; prefer the runtime
        // provider key when it names one.
        if let Some(runtime) = runtime.as_deref()
            && runtime != "openrouter"
            && crate::provider_catalog::openai_compatible_profile_by_id(runtime).is_some()
        {
            return format!("openai-compatible:{}", runtime);
        }
        return "openrouter".to_string();
    }

    // Direct OpenAI-compatible profiles, matched by id or display name.
    for profile in crate::provider_catalog::openai_compatible_profiles() {
        if normalized == profile.id || normalized == profile.display_name.to_ascii_lowercase() {
            return format!("openai-compatible:{}", profile.id);
        }
    }

    if normalized.contains("azure") {
        return "azure-openai".to_string();
    }
    if normalized.contains("bedrock") {
        return "bedrock".to_string();
    }
    if normalized.contains("anthropic") || normalized.contains("claude") {
        return "claude:api-key".to_string();
    }
    if normalized.contains("openai") {
        return "openai:api-key".to_string();
    }
    if normalized.contains("copilot") {
        return "copilot".to_string();
    }
    if normalized.contains("gemini") {
        return "gemini".to_string();
    }
    if normalized.contains("cursor") {
        return "cursor".to_string();
    }

    // Fallback: slug of the display name so unknown providers still bucket
    // consistently.
    let slug: String = normalized
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        "unknown".to_string()
    } else {
        slug
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        crate::storage::lock_test_env()
    }

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let prev = std::env::var_os(key);
            crate::env::set_var(key, value);
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                crate::env::set_var(self.key, prev);
            } else {
                crate::env::remove_var(self.key);
            }
        }
    }

    fn clear_ledger_cache() {
        if let Ok(mut guard) = LEDGER.lock() {
            *guard = None;
        }
    }

    #[test]
    fn record_use_and_spend_roundtrip_under_jcode_home() {
        let _env_lock = lock_env();
        clear_ledger_cache();
        let temp = tempfile::tempdir().expect("tempdir");
        let _home = EnvVarGuard::set("JCODE_HOME", temp.path().as_os_str());

        record_use("claude:oauth:claude-1");
        record_spend("claude:api-key", 0.25);
        record_spend("claude:api-key", 0.50);

        let used = last_used_unix_secs("claude:oauth:claude-1").expect("last used recorded");
        assert!(now_unix_secs().saturating_sub(used) < 5);

        let spend = spend_snapshot("claude:api-key").expect("spend recorded");
        assert!((spend.day_usd - 0.75).abs() < 1e-9);
        assert!((spend.all_time_usd - 0.75).abs() < 1e-9);
        // Spend also bumps recency.
        assert!(last_used_unix_secs("claude:api-key").is_some());

        // Persisted to disk, not just memory.
        clear_ledger_cache();
        let spend = spend_snapshot("claude:api-key").expect("spend reloaded from disk");
        assert!((spend.all_time_usd - 0.75).abs() < 1e-9);
    }

    #[test]
    fn record_spend_ignores_invalid_amounts() {
        let _env_lock = lock_env();
        clear_ledger_cache();
        let temp = tempfile::tempdir().expect("tempdir");
        let _home = EnvVarGuard::set("JCODE_HOME", temp.path().as_os_str());

        record_spend("openai:api-key", 0.0);
        record_spend("openai:api-key", -1.0);
        record_spend("openai:api-key", f64::NAN);
        assert!(spend_snapshot("openai:api-key").is_none());
    }

    #[test]
    fn source_key_mapping_covers_known_providers() {
        let _env_lock = lock_env();
        let _profile = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "");
        assert_eq!(
            source_key_for_provider_label("DeepSeek", None),
            "openai-compatible:deepseek"
        );
        assert_eq!(
            source_key_for_provider_label("Moonshot AI", None),
            "openai-compatible:moonshotai"
        );
        assert_eq!(
            source_key_for_provider_label("OpenRouter", None),
            "openrouter"
        );
        assert_eq!(
            source_key_for_provider_label("OpenRouter", Some("deepseek")),
            "openai-compatible:deepseek"
        );
        assert_eq!(
            source_key_for_provider_label("Anthropic", None),
            "claude:api-key"
        );
        assert_eq!(
            source_key_for_provider_label("OpenAI", None),
            "openai:api-key"
        );
        assert_eq!(
            source_key_for_provider_label("Some Custom Endpoint", None),
            "some-custom-endpoint"
        );
    }

    #[test]
    fn route_metadata_source_identity_ignores_ambient_profile_and_display_name() {
        let _env_lock = lock_env();
        let _profile = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "openai-proxy");
        let _runtime = EnvVarGuard::set("JCODE_RUNTIME_PROVIDER", "openai");
        let _namespace = EnvVarGuard::set("JCODE_OPENROUTER_CACHE_NAMESPACE", "openrouter");

        let source = source_identity_for_route_metadata(
            Some("openai-proxy"),
            Some("openai-compatible:openai-proxy"),
            Some("openai-proxy/gpt-5.4"),
        )
        .expect("named route metadata should resolve");
        assert_eq!(source.source_key, "named-profile:openai-proxy");
        assert!(
            !source.is_openai,
            "a custom profile must not become native OpenAI"
        );
        assert!(source.is_metered);

        let source = source_identity_for_route_metadata(
            Some("openrouter"),
            Some("openrouter"),
            Some("openrouter/openai/gpt-5.4@OpenAI"),
        )
        .expect("OpenRouter route metadata should resolve");
        assert_eq!(source.source_key, "openrouter");
        assert!(source.is_metered);

        let source = source_identity_for_route_metadata(
            Some("openai-proxy"),
            Some("openrouter"),
            Some("openai-proxy/gpt-5.4"),
        )
        .expect("named profile on shared transport should resolve");
        assert_eq!(source.source_key, "named-profile:openai-proxy");
        assert!(!source.is_openai);
    }

    #[test]
    fn route_metadata_source_identity_tracks_a_to_b_transition() {
        let _env_lock = lock_env();
        let _profile = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "route-a");
        let _runtime = EnvVarGuard::set("JCODE_RUNTIME_PROVIDER", "route-a");

        let route_a = source_identity_for_route_metadata(
            Some("route-a"),
            Some("openai-compatible:route-a"),
            Some("route-a/shared-model"),
        )
        .expect("route A should resolve");
        let route_b = source_identity_for_route_metadata(
            Some("route-b"),
            Some("openai-compatible:route-b"),
            Some("route-b/shared-model"),
        )
        .expect("route B should resolve");

        assert_eq!(route_a.source_key, "named-profile:route-a");
        assert_eq!(route_b.source_key, "named-profile:route-b");
        assert_ne!(route_a.source_key, route_b.source_key);
    }

    #[test]
    fn named_profiles_with_same_display_label_get_distinct_source_keys() {
        let _env_lock = lock_env();
        let first = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "work gateway");
        assert_eq!(
            source_key_for_provider_label("Same Gateway", Some("openai-compatible")),
            "named-profile:work%20gateway"
        );
        drop(first);

        let _second = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "personal gateway");
        assert_eq!(
            source_key_for_provider_label("Same Gateway", Some("openai-compatible")),
            "named-profile:personal%20gateway"
        );
    }

    #[test]
    fn named_profiles_keep_spend_in_separate_ledger_entries() {
        let _env_lock = lock_env();
        clear_ledger_cache();
        let temp = tempfile::tempdir().expect("tempdir");
        let _home = EnvVarGuard::set("JCODE_HOME", temp.path().as_os_str());

        let _work = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "work");
        let work_key = source_key_for_provider_label("Same Gateway", Some("openai-compatible"));
        record_spend(&work_key, 1.25);

        let _personal = EnvVarGuard::set("JCODE_NAMED_PROVIDER_PROFILE", "personal");
        let personal_key = source_key_for_provider_label("Same Gateway", Some("openai-compatible"));
        record_spend(&personal_key, 2.50);

        assert_ne!(work_key, personal_key);
        assert_eq!(
            spend_snapshot(&work_key).expect("work spend").all_time_usd,
            1.25
        );
        assert_eq!(
            spend_snapshot(&personal_key)
                .expect("personal spend")
                .all_time_usd,
            2.50
        );
        let entries = all_entries();
        assert!(entries.iter().any(|(key, _)| key == &work_key));
        assert!(entries.iter().any(|(key, _)| key == &personal_key));
    }

    #[test]
    fn legacy_ledger_entry_still_parses_without_origin_metadata() {
        let value = serde_json::json!({
            "entries": {
                "openrouter": {
                    "last_used_unix_secs": 123,
                    "spend": {
                        "day_date": "2026-07-31",
                        "day_usd": 1.25,
                        "month": "2026-07",
                        "month_usd": 2.50,
                        "all_time_usd": 3.75
                    }
                }
            }
        });
        let store: ProviderActivityStore = serde_json::from_value(value).expect("legacy ledger");
        assert_eq!(store.entries["openrouter"].last_used_unix_secs, Some(123));
        assert_eq!(
            store.entries["openrouter"]
                .spend
                .as_ref()
                .map(|spend| spend.all_time_usd),
            Some(3.75)
        );
    }

    #[test]
    fn relative_age_formatting() {
        let now = now_unix_secs();
        assert_eq!(format_relative_age(now), "just now");
        assert_eq!(format_relative_age(now - 120), "2m ago");
        assert_eq!(format_relative_age(now - 3_600), "1h ago");
        assert_eq!(format_relative_age(now - 2 * 86_400), "2d ago");
    }
}
