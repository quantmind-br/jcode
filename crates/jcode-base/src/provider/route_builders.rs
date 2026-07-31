use super::pricing::{cheapness_for_route, openrouter_pricing_from_model_pricing};
use super::{ModelRoute, ModelRouteApiMethod, RouteCostConfidence, RouteCostSource};
use std::collections::{BTreeMap, BTreeSet};

pub fn is_listable_model_name(model: &str) -> bool {
    let trimmed = model.trim();
    !trimmed.is_empty()
        && !matches!(trimmed, "copilot models" | "openrouter models")
        && !model_name_is_likely_non_chat(trimmed)
}

/// Heuristic to keep obviously non-chat models (embeddings, speech, image,
/// rerankers, etc.) out of the chat model picker. OpenAI-compatible profiles
/// (e.g. NVIDIA NIM, FPT, Chutes, Groq) and Bedrock expose their *entire*
/// catalog, which otherwise floods the picker with hundreds of models that
/// can't be used for chat. The match is conservative and token-boundary aware
/// so it won't drop a legitimately named chat model.
pub fn model_name_is_likely_non_chat(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    // Strip any provider/path prefix so "01-ai/yi-large" -> matches on the full
    // string but boundary tokens still work across '/', '.', '-', '_' and ':'.
    let tokens: Vec<&str> = lower
        .split(|c: char| !(c.is_ascii_alphanumeric()))
        .filter(|t| !t.is_empty())
        .collect();

    // Substring markers that are unambiguous regardless of token boundaries.
    const SUBSTRING_MARKERS: &[&str] = &[
        "embed",
        "embedding",
        "rerank",
        "reranker",
        "whisper",
        "speech",
        "tts",
        "stt",
        "transcribe",
        "voxtral",
        "moderation",
        "guard",
        "upscale",
        "outpaint",
        "nemoretriever",
        "riva-translate",
        "gpt-audio",
        "style-transfer",
        "video-detector",
    ];
    if SUBSTRING_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }

    // Whole-token markers (avoid false positives like "vits" inside a word).
    const TOKEN_MARKERS: &[&str] = &[
        "vits",
        "pegasus",
        "asr",
        "ocr",
        "kie",
        "vad",
        "diffusion",
        "image",
        "vision",
        "bge",
        "gte",
        "e5",
        "nvclip",
        "orpheus",
        "lyria",
        "deplot",
        "parse",
        "gliner",
        "fuyu",
        "kosmos",
        "neva",
        "vila",
        "nvembed",
        "reward",
    ];
    if tokens.iter().any(|t| TOKEN_MARKERS.contains(t)) {
        // "vision" is sometimes part of a multimodal chat model, so only drop
        // it when paired with another non-chat signal. Vision-language chat
        // models keep words like "instruct"/"chat" which we treat as a keep.
        if tokens.contains(&"vision")
            && (tokens.contains(&"instruct") || tokens.contains(&"chat") || tokens.contains(&"vl"))
        {
            // multimodal chat: keep
        } else {
            return true;
        }
    }

    false
}

pub fn openrouter_catalog_model_id(model: &str) -> Option<String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Only the curated native ids may be rewritten.  Prefix heuristics such as
    // `gpt-*`/`claude-*` are valid for standalone provider detection, but are
    // not enough to prove that an opaque id from a named compatible profile is
    // an OpenAI/Anthropic model.  A slash-qualified value is already an
    // explicit OpenRouter catalog id and is preserved byte-for-byte.
    if crate::provider::bedrock::BedrockProvider::is_bedrock_model_id(trimmed) {
        return None;
    }
    if trimmed.contains('/') || trimmed.contains('@') {
        return Some(trimmed.to_string());
    }
    if jcode_provider_core::model_id::matches_known_model(trimmed, super::ALL_CLAUDE_MODELS) {
        return Some(format!("anthropic/{}", trimmed));
    }
    if jcode_provider_core::model_id::matches_known_model(trimmed, super::ALL_OPENAI_MODELS) {
        return Some(format!("openai/{}", trimmed));
    }
    None
}

#[cfg(test)]
mod identity_tests {
    use super::{
        is_listable_model_name, listable_model_names_from_routes, openrouter_catalog_model_id,
    };
    use crate::provider::ModelRoute;

    #[test]
    fn custom_gpt_and_claude_ids_are_not_rewritten_as_native_openrouter_models() {
        assert_eq!(
            openrouter_catalog_model_id("gpt-custom/v1"),
            Some("gpt-custom/v1".to_string())
        );
        assert_eq!(
            openrouter_catalog_model_id("claude-custom/model"),
            Some("claude-custom/model".to_string())
        );
        assert_eq!(openrouter_catalog_model_id("gpt-5.4-proxy"), None);
        assert_eq!(openrouter_catalog_model_id("claude-sonnet-custom"), None);
    }

    fn route(model: &str, provider: &str, api_method: &str) -> ModelRoute {
        ModelRoute {
            model: model.to_string(),
            provider: provider.to_string(),
            api_method: api_method.to_string(),
            available: true,
            detail: String::new(),
            cheapness: None,
        }
    }

    #[test]
    fn listable_models_qualify_homonymous_ids_and_keep_unique_bare() {
        let routes = vec![
            route("gpt-5.6-sol", "Provider A", "openai-compatible:prov-a"),
            route("only-here-9", "Provider C", "openai-compatible:prov-c"),
            route("gpt-5.6-sol", "Provider B", "openai-compatible:prov-b"),
        ];

        assert!(is_listable_model_name("gpt-5.6-sol"));
        assert!(is_listable_model_name("only-here-9"));

        let listed = listable_model_names_from_routes(&routes);
        assert_eq!(
            listed,
            vec![
                "prov-a/gpt-5.6-sol".to_string(),
                "only-here-9".to_string(),
                "prov-b/gpt-5.6-sol".to_string(),
            ]
        );
    }
}

/// Provider namespace for a route, used to build canonical `provider/model` ids.
pub fn route_provider_key(route: &ModelRoute) -> Option<String> {
    match route.api_method_kind() {
        ModelRouteApiMethod::JcodeSubscription => Some("jcode".to_string()),
        ModelRouteApiMethod::ClaudeOAuth | ModelRouteApiMethod::AnthropicApiKey => {
            Some("claude".to_string())
        }
        ModelRouteApiMethod::OpenAIOAuth | ModelRouteApiMethod::OpenAIApiKey => {
            Some("openai".to_string())
        }
        ModelRouteApiMethod::Other(method) if method == "chatgpt-web" => {
            Some("openai".to_string())
        }
        ModelRouteApiMethod::OpenRouter => Some("openrouter".to_string()),
        ModelRouteApiMethod::OpenAiCompatible {
            profile_id: Some(profile_id),
        } => Some(profile_id),
        ModelRouteApiMethod::OpenAiCompatible { profile_id: None } => {
            let provider = route.provider.trim();
            let named_profile = crate::config::config()
                .providers
                .keys()
                .find(|name| name.eq_ignore_ascii_case(provider));
            named_profile
                .cloned()
                .or_else(|| (!provider.is_empty()).then(|| "openai-compatible".to_string()))
        }
        ModelRouteApiMethod::Copilot => Some("copilot".to_string()),
        ModelRouteApiMethod::Cursor => Some("cursor".to_string()),
        ModelRouteApiMethod::Bedrock => Some("bedrock".to_string()),
        ModelRouteApiMethod::CodeAssistOAuth => Some("gemini".to_string()),
        ModelRouteApiMethod::AntigravityHttps => Some("antigravity".to_string()),
        ModelRouteApiMethod::RemoteCatalog | ModelRouteApiMethod::Current => None,
        ModelRouteApiMethod::Other(_) => {
            let provider = route.provider.trim();
            (!provider.is_empty()).then(|| provider.to_ascii_lowercase())
        }
    }
}

/// Canonical `provider/model` identity for a route, if one can be derived.
///
/// OpenRouter-native ids that already start with `openrouter/` are returned
/// unchanged so they are not double-wrapped.
pub fn canonical_route_identity(route: &ModelRoute) -> Option<String> {
    let provider = route_provider_key(route)?;
    if provider == "openrouter" && route.model.starts_with("openrouter/") {
        Some(route.model.clone())
    } else {
        Some(jcode_provider_core::format_provider_model(
            &provider,
            &route.model,
        ))
    }
}


pub fn listable_model_names_from_routes(routes: &[ModelRoute]) -> Vec<String> {
    let mut bare_identity_counts: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for route in routes {
        if !is_listable_model_name(&route.model) {
            continue;
        }
        if let Some(identity) = canonical_route_identity(route) {
            bare_identity_counts
                .entry(route.model.as_str())
                .or_default()
                .insert(identity);
        }
    }

    let mut models = Vec::new();
    let mut seen = BTreeSet::new();
    for route in routes {
        if !is_listable_model_name(&route.model) {
            continue;
        }
        let identity = match canonical_route_identity(route) {
            Some(identity) => identity,
            None => {
                if seen.insert(route.model.clone()) {
                    models.push(route.model.clone());
                }
                continue;
            }
        };
        let bare = route.model.as_str();
        let distinct = bare_identity_counts
            .get(bare)
            .map(|set| set.len())
            .unwrap_or(0);
        let emitted = if distinct <= 1 {
            route.model.clone()
        } else {
            identity
        };
        if seen.insert(emitted.clone()) {
            models.push(emitted);
        }
    }
    models
}

pub fn build_anthropic_oauth_route(
    model: &str,
    available: bool,
    detail: impl Into<String>,
) -> ModelRoute {
    ModelRoute {
        model: model.to_string(),
        provider: "Anthropic".to_string(),
        api_method: "claude-oauth".to_string(),
        available,
        detail: detail.into(),
        cheapness: cheapness_for_route(model, "Anthropic", "claude-oauth"),
    }
}

pub fn build_openai_oauth_route(
    model: &str,
    available: bool,
    detail: impl Into<String>,
) -> ModelRoute {
    build_openai_route(model, "openai-oauth", available, detail)
}

pub fn build_openai_api_key_route(
    model: &str,
    available: bool,
    detail: impl Into<String>,
) -> ModelRoute {
    build_openai_route(model, "openai-api-key", available, detail)
}

pub fn build_chatgpt_web_route() -> ModelRoute {
    ModelRoute {
        model: super::CHATGPT_WEB_MODEL.to_string(),
        provider: "OpenAI".to_string(),
        api_method: "chatgpt-web".to_string(),
        available: true,
        detail: "logged-in Firefox ChatGPT session".to_string(),
        cheapness: None,
    }
}

fn build_openai_route(
    model: &str,
    api_method: &str,
    available: bool,
    detail: impl Into<String>,
) -> ModelRoute {
    ModelRoute {
        model: model.to_string(),
        provider: "OpenAI".to_string(),
        api_method: api_method.to_string(),
        available,
        detail: detail.into(),
        cheapness: cheapness_for_route(model, "OpenAI", api_method),
    }
}

pub fn build_copilot_route(model: &str, available: bool, detail: impl Into<String>) -> ModelRoute {
    ModelRoute {
        model: model.to_string(),
        provider: "Copilot".to_string(),
        api_method: "copilot".to_string(),
        available,
        detail: detail.into(),
        cheapness: cheapness_for_route(model, "Copilot", "copilot"),
    }
}

pub fn build_openrouter_auto_route(
    model: &str,
    available: bool,
    auto_detail: impl Into<String>,
) -> ModelRoute {
    ModelRoute {
        model: model.to_string(),
        provider: "auto".to_string(),
        api_method: "openrouter".to_string(),
        available,
        detail: auto_detail.into(),
        cheapness: cheapness_for_route(model, "auto", "openrouter"),
    }
}

pub fn build_openrouter_endpoint_route(
    model: &str,
    endpoint: &crate::provider::openrouter::EndpointInfo,
    available: bool,
    age_suffix: Option<&str>,
) -> ModelRoute {
    let mut detail = endpoint.detail_string();
    if let Some(age_suffix) = age_suffix.map(str::trim).filter(|value| !value.is_empty()) {
        if !detail.is_empty() {
            detail = format!("{}, {}", detail, age_suffix);
        } else {
            detail = age_suffix.to_string();
        }
    }

    ModelRoute {
        model: model.to_string(),
        provider: endpoint.provider_name.clone(),
        api_method: "openrouter".to_string(),
        available,
        detail,
        cheapness: openrouter_pricing_from_model_pricing(
            &endpoint.pricing,
            RouteCostSource::OpenRouterEndpoint,
            RouteCostConfidence::High,
            Some(format!(
                "OpenRouter endpoint pricing for {}",
                endpoint.provider_name
            )),
        ),
    }
}

pub fn build_openrouter_fallback_provider_route(
    display_model: &str,
    catalog_model: &str,
    provider: &str,
) -> ModelRoute {
    ModelRoute {
        model: display_model.to_string(),
        provider: provider.to_string(),
        api_method: "openrouter".to_string(),
        available: true,
        detail: String::new(),
        cheapness: cheapness_for_route(catalog_model, provider, "openrouter"),
    }
}

#[cfg(test)]
mod listable_tests {
    use super::{is_listable_model_name, model_name_is_likely_non_chat};

    #[test]
    fn keeps_real_chat_models() {
        for model in [
            "claude-opus-4-8",
            "gpt-5.5",
            "gemini-3.1-pro",
            "MiniMax-M2.5",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "Llama-3.3-70B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "01-ai/yi-large",
            "anthropic/claude-opus-4.8",
            "deepseek-chat",
            "kimi-k2",
        ] {
            assert!(
                is_listable_model_name(model),
                "expected chat model to be listable: {model}"
            );
        }
    }

    #[test]
    fn drops_non_chat_models() {
        for model in [
            "amazon.titan-embed-text-v2:0",
            "cohere.embed-v4:0",
            "Vietnamese_Embedding",
            "text-embedding-3-large",
            "FPT.AI-whisper-large-v3-turbo",
            "FPT.AI-whisper-medium",
            "FPT.AI-VITs",
            "FPT.AI-KIE-v1.7",
            "twelvelabs.pegasus-1-2-v1:0",
            "whisper-large-v3",
            "voxtral-small",
            "BAAI/bge-reranker-v2-m3",
            "llama-guard-3-8b",
            "omni-moderation-latest",
            "baai/bge-m3",
            "multilingual-e5-large",
            "gte-qwen2-7b",
            "nvidia/nvclip",
            "canopylabs/orpheus-v1-english",
            "google/lyria-3-pro-preview",
            "google/deplot",
            "nvidia/nemoretriever-parse",
            "nvidia/riva-translate-4b-instruct",
            "us.stability.stable-fast-upscale-v1:0",
            "us.stability.stable-outpaint-v1:0",
            "adept/fuyu-8b",
            "microsoft/kosmos-2",
            "nvidia/gliner-pii",
            "openai/gpt-audio",
            "openai/gpt-audio-mini",
            "us.stability.stable-style-transfer-v1:0",
            "nvidia/nemotron-4-340b-reward",
            "nvidia/ai-synthetic-video-detector",
        ] {
            assert!(
                model_name_is_likely_non_chat(model),
                "expected non-chat model to be filtered: {model}"
            );
            assert!(!is_listable_model_name(model));
        }
    }

    #[test]
    fn empty_and_sentinels_filtered() {
        assert!(!is_listable_model_name(""));
        assert!(!is_listable_model_name("   "));
        assert!(!is_listable_model_name("copilot models"));
        assert!(!is_listable_model_name("openrouter models"));
    }
}
