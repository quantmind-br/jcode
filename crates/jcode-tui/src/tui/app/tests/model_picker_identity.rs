use crate::message::Message;
use crate::provider::{EventStream, ModelRoute, Provider, RouteSelection};
use crate::tool::Registry;
use crate::tui::app::App;
use crate::tui::PickerKind;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// A provider that exposes the same short model id through two different sources:
/// the built-in OpenAI OAuth route and a named OpenAI-compatible profile called
/// "Solarpanel". This is the core identity collision the Phase 2 UI must make
/// impossible to confuse.
#[derive(Clone)]
struct DualSourceProvider {
    routes: Vec<ModelRoute>,
    effort: Arc<StdMutex<Option<String>>>,
    set_route_calls: Arc<StdMutex<Vec<String>>>,
    set_model_calls: Arc<StdMutex<Vec<String>>>,
}

impl DualSourceProvider {
    fn new(
        routes: Vec<ModelRoute>,
        set_route_calls: Arc<StdMutex<Vec<String>>>,
        set_model_calls: Arc<StdMutex<Vec<String>>>,
    ) -> Self {
        Self {
            routes,
            effort: Arc::new(StdMutex::new(Some("none".to_string()))),
            set_route_calls,
            set_model_calls,
        }
    }
}

#[async_trait]
impl Provider for DualSourceProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[crate::message::ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        unimplemented!("DualSourceProvider")
    }

    fn name(&self) -> &str {
        "solarpanel"
    }

    fn display_name(&self) -> String {
        "Solarpanel".to_string()
    }

    fn model(&self) -> String {
        "gpt-5.6-sol".to_string()
    }

    fn reasoning_effort(&self) -> Option<String> {
        self.effort.lock().unwrap().clone()
    }

    fn set_reasoning_effort(&self, effort: &str) -> Result<()> {
        *self.effort.lock().unwrap() = Some(effort.to_string());
        Ok(())
    }

    fn available_efforts(&self) -> Vec<&'static str> {
        vec!["none", "low", "medium", "high", "max"]
    }

    fn model_routes(&self) -> Vec<ModelRoute> {
        self.routes.clone()
    }

    fn set_model(&self, model: &str) -> Result<()> {
        self.set_model_calls
            .lock()
            .unwrap()
            .push(model.to_string());
        Ok(())
    }

    fn set_route_selection(&self, selection: &RouteSelection) -> Result<()> {
        self.set_route_calls
            .lock()
            .unwrap()
            .push(selection.routed_model_spec());
        Ok(())
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

enum RouteSet {
    Dual,
    OpenAiOnly,
}

fn build_routes(kind: RouteSet) -> Vec<ModelRoute> {
    match kind {
        RouteSet::Dual => vec![
            ModelRoute {
                model: "gpt-5.6-sol".to_string(),
                provider: "OpenAI".to_string(),
                api_method: "openai-oauth".to_string(),
                available: true,
                detail: String::new(),
                cheapness: None,
            },
            ModelRoute {
                model: "gpt-5.6-sol".to_string(),
                provider: "Solarpanel".to_string(),
                api_method: "openai-compatible:solarpanel".to_string(),
                available: true,
                detail: String::new(),
                cheapness: None,
            },
        ],
        RouteSet::OpenAiOnly => vec![ModelRoute {
            model: "gpt-5.6-sol".to_string(),
            provider: "OpenAI".to_string(),
            api_method: "openai-oauth".to_string(),
            available: true,
            detail: String::new(),
            cheapness: None,
        }],
    }
}

fn create_dual_source_test_app(
    kind: RouteSet,
) -> (
    App,
    Arc<StdMutex<Vec<String>>>,
    Arc<StdMutex<Vec<String>>>,
) {
    ensure_test_jcode_home_if_unset();
    clear_persisted_test_ui_state();
    crate::tui::ui::clear_test_render_state_for_tests();

    let route_calls = Arc::new(StdMutex::new(Vec::new()));
    let model_calls = Arc::new(StdMutex::new(Vec::new()));
    let provider: Arc<dyn Provider> = Arc::new(DualSourceProvider::new(
        build_routes(kind),
        route_calls.clone(),
        model_calls.clone(),
    ));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let registry = rt.block_on(Registry::new(provider.clone()));
    let mut app = App::new_for_test_harness(provider, registry);
    app.queue_mode = false;
    app.diff_mode = crate::config::DiffDisplayMode::Inline;
    (app, route_calls, model_calls)
}

#[test]
fn model_picker_groups_homonymous_models_by_provider_identity() {
    let (mut app, _route_calls, _model_calls) = create_dual_source_test_app(RouteSet::Dual);

    app.open_model_picker();
    wait_for_model_picker_load(&mut app);

    let picker = app
        .inline_interactive_state
        .as_ref()
        .expect("model picker should be open");
    assert_eq!(picker.kind, PickerKind::Model);

    let gpt_entries: Vec<_> = picker
        .entries
        .iter()
        .filter(|entry| {
            let base = entry
                .effort
                .as_deref()
                .and_then(|effort| entry.name.strip_suffix(&format!(" ({effort})")))
                .unwrap_or(&entry.name);
            base == "gpt-5.6-sol"
        })
        .collect();
    assert!(
        gpt_entries.len() >= 2,
        "two providers exposing the same short id must produce distinct rows, got {:?}",
        picker
            .entries
            .iter()
            .map(|e| (
                e.name.clone(),
                e.active_option().map(|r| r.provider.clone())
            ))
            .collect::<Vec<_>>()
    );

    let providers: Vec<String> = gpt_entries
        .iter()
        .map(|entry| {
            entry
                .active_option()
                .expect("entry must have a route")
                .provider
                .clone()
        })
        .collect();
    assert!(providers.contains(&"OpenAI".to_string()));
    assert!(providers.contains(&"Solarpanel".to_string()));

    let current = gpt_entries
        .iter()
        .find(|entry| entry.is_current)
        .expect("exactly the current provider/model row should be highlighted");
    assert_eq!(
        current.active_option().unwrap().provider,
        "Solarpanel",
        "current selection must point to the Solarpanel source, not silently fall back to OpenAI"
    );

    let non_current = gpt_entries
        .iter()
        .find(|entry| !entry.is_current)
        .expect("the other provider row should be present but not marked current");
    assert_eq!(non_current.active_option().unwrap().provider, "OpenAI");
}

#[test]
fn typed_model_bare_name_shows_qualified_candidates_when_ambiguous() {
    let (mut app, _route_calls, _model_calls) = create_dual_source_test_app(RouteSet::Dual);

    let handled =
        crate::tui::app::model_context::handle_model_command(&mut app, "/model gpt-5.6-sol");
    assert!(handled, "/model command should be consumed");

    let last = app
        .display_messages
        .last()
        .expect("ambiguous /model should post an error message");
    assert_eq!(last.role, "error");
    let content = &last.content;
    assert!(
        content.contains("gpt-5.6-sol"),
        "error should name the ambiguous model: {content}"
    );
    assert!(
        content.contains("openai/gpt-5.6-sol") && content.contains("solarpanel/gpt-5.6-sol"),
        "error should list both canonical candidates: {content}"
    );
    assert!(
        content.contains("/model <provider>/gpt-5.6-sol")
            || content.contains("/model <provider>:gpt-5.6-sol"),
        "error should advise a qualified form: {content}"
    );
    assert!(
        content.contains("/model") && content.contains("picker"),
        "error should preserve the normal picker affordance: {content}"
    );
    assert_eq!(
        app.status_notice(),
        Some("Ambiguous model".to_string()),
        "status notice should report the ambiguity"
    );
}

#[test]
fn typed_model_bare_name_uses_canonical_set_model_when_unambiguous() {
    let (mut app, route_calls, model_calls) =
        create_dual_source_test_app(RouteSet::OpenAiOnly);

    let handled =
        crate::tui::app::model_context::handle_model_command(&mut app, "/model gpt-5.6-sol");
    assert!(handled);

    assert!(
        route_calls.lock().unwrap().is_empty(),
        "a bare model switch must not invoke credential-specific set_route_selection"
    );
    let calls = model_calls.lock().unwrap();
    assert_eq!(
        calls.as_slice(),
        ["openai/gpt-5.6-sol"],
        "a single-source bare model should switch via the canonical provider/model spec: {calls:?}"
    );
}

#[test]
fn typed_model_qualified_spec_passes_through_to_provider() {
    let (mut app, _route_calls, model_calls) = create_dual_source_test_app(RouteSet::Dual);

    let handled = crate::tui::app::model_context::handle_model_command(
        &mut app,
        "/model openai:gpt-5.6-sol",
    );
    assert!(handled);

    let calls = model_calls.lock().unwrap();
    assert!(
        calls.iter().any(|c| c == "openai:gpt-5.6-sol"),
        "qualified spec should be passed through to provider.set_model: {calls:?}"
    );
}

#[test]
fn typed_model_bare_name_persists_canonical_session_identity() {
    let (mut app, _route_calls, _model_calls) =
        create_dual_source_test_app(RouteSet::OpenAiOnly);

    let handled =
        crate::tui::app::model_context::handle_model_command(&mut app, "/model gpt-5.6-sol");
    assert!(handled);

    assert_eq!(app.session.model.as_deref(), Some("openai/gpt-5.6-sol"));
    assert_eq!(app.session.provider_key.as_deref(), Some("openai"));
    assert_eq!(app.session.route_api_method, None);
    assert_eq!(app.session.model_identity_format, Some(1));
}

#[test]
fn typed_canonical_cross_provider_switch_clears_stale_route_api_method() {
    let (mut app, _route_calls, _model_calls) =
        create_dual_source_test_app(RouteSet::Dual);

    app.session.model = Some("solarpanel/gpt-5.6-sol".to_string());
    app.session.provider_key = Some("solarpanel".to_string());
    app.session.route_api_method = Some("openai-compatible:solarpanel".to_string());
    app.session.model_identity_format = Some(1);

    let handled = crate::tui::app::model_context::handle_model_command(
        &mut app,
        "/model openai/gpt-5.6-sol",
    );
    assert!(handled);

    assert_eq!(app.session.model.as_deref(), Some("openai/gpt-5.6-sol"));
    assert_eq!(app.session.provider_key.as_deref(), Some("openai"));
    assert_eq!(app.session.route_api_method, None);
    assert_eq!(app.session.model_identity_format, Some(1));
}

#[derive(Clone)]
struct DisplayCollisionProvider {
    routes: Vec<ModelRoute>,
}

#[async_trait]
impl Provider for DisplayCollisionProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[crate::message::ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        unimplemented!("DisplayCollisionProvider")
    }

    fn name(&self) -> &str {
        "openrouter"
    }

    fn display_name(&self) -> String {
        "OpenAI".to_string()
    }

    fn model(&self) -> String {
        "gpt-5.4".to_string()
    }

    fn reasoning_effort(&self) -> Option<String> {
        Some("none".to_string())
    }

    fn available_efforts(&self) -> Vec<&'static str> {
        vec!["none", "low", "medium", "high", "max"]
    }

    fn model_routes(&self) -> Vec<ModelRoute> {
        self.routes.clone()
    }

    fn set_model(&self, _model: &str) -> Result<()> {
        Ok(())
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

fn create_display_collision_test_app() -> App {
    ensure_test_jcode_home_if_unset();
    clear_persisted_test_ui_state();
    crate::tui::ui::clear_test_render_state_for_tests();

    let routes = vec![
        ModelRoute {
            model: "gpt-5.4".to_string(),
            provider: "OpenAI".to_string(),
            api_method: "openai-oauth".to_string(),
            available: true,
            detail: String::new(),
            cheapness: None,
        },
        ModelRoute {
            model: "gpt-5.4".to_string(),
            provider: "OpenAI".to_string(),
            api_method: "openrouter".to_string(),
            available: true,
            detail: String::new(),
            cheapness: None,
        },
    ];
    let provider: Arc<dyn Provider> = Arc::new(DisplayCollisionProvider { routes });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let registry = rt.block_on(Registry::new(provider.clone()));
    let mut app = App::new_for_test_harness(provider, registry);
    app.queue_mode = false;
    app.diff_mode = crate::config::DiffDisplayMode::Inline;
    app
}

#[test]
fn display_label_collision_resolves_to_distinct_canonical_providers() {
    let mut app = create_display_collision_test_app();

    let handled =
        crate::tui::app::model_context::handle_model_command(&mut app, "/model gpt-5.4");
    assert!(handled);

    let last = app
        .display_messages
        .last()
        .expect("ambiguous /model should post an error message");
    assert_eq!(last.role, "error");
    let content = &last.content;
    assert!(
        content.contains("openai/gpt-5.4"),
        "error should list the native OpenAI canonical candidate: {content}"
    );
    assert!(
        content.contains("openrouter/openai/gpt-5.4@OpenAI"),
        "error should list the OpenRouter canonical candidate: {content}"
    );
}

#[test]
fn current_marker_uses_canonical_identity_not_display_label() {
    let mut app = create_display_collision_test_app();

    app.session.model = Some("openrouter/openai/gpt-5.4@OpenAI".to_string());
    app.session.provider_key = Some("openrouter".to_string());
    app.session.route_api_method = Some("openrouter".to_string());
    app.session.model_identity_format = Some(1);

    app.open_model_picker();
    wait_for_model_picker_load(&mut app);

    let picker = app
        .inline_interactive_state
        .as_ref()
        .expect("model picker should be open");

    let openrouter_entry = picker
        .entries
        .iter()
        .find(|entry| {
            entry
                .active_option()
                .map(|r| r.api_method == "openrouter")
                .unwrap_or(false)
        })
        .expect("OpenRouter route should be a distinct entry");
    assert!(
        openrouter_entry.is_current,
        "the OpenRouter row must be highlighted by canonical identity, not by display label collision: {:?}",
        picker.entries.iter().map(|e| (e.name.clone(), e.active_option().map(|r| r.api_method.clone()), e.is_current)).collect::<Vec<_>>()
    );

    let openai_entry = picker
        .entries
        .iter()
        .find(|entry| {
            entry
                .active_option()
                .map(|r| r.api_method == "openai-oauth")
                .unwrap_or(false)
        })
        .expect("OpenAI route should be a distinct entry");
    assert!(
        !openai_entry.is_current,
        "the native OpenAI row must not be highlighted when the OpenRouter identity is active"
    );
}

#[test]
fn explicit_picker_selection_retains_api_method_pin() {
    let (mut app, _route_calls, _model_calls) =
        create_dual_source_test_app(RouteSet::OpenAiOnly);

    let route = ModelRoute {
        model: "gpt-5.6-sol".to_string(),
        provider: "OpenAI".to_string(),
        api_method: "openai-oauth".to_string(),
        available: true,
        detail: String::new(),
        cheapness: None,
    };
    let selection = RouteSelection::from_model_route(&route);
    app.provider.set_route_selection(&selection).unwrap();

    let active_model = app.finalize_model_switch_from_selection(&selection);
    assert_eq!(active_model, "gpt-5.6-sol");
    assert_eq!(app.session.model.as_deref(), Some("openai/gpt-5.6-sol"));
    assert_eq!(app.session.provider_key.as_deref(), Some("openai-oauth"));
    assert_eq!(app.session.route_api_method.as_deref(), Some("openai-oauth"));
    assert_eq!(app.session.model_identity_format, Some(1));
}

/// A minimal configurable provider for state tests that only need a stable
/// canonical name/model and never make API calls.
#[derive(Clone)]
struct NamedProvider {
    provider_name: &'static str,
    provider_model: &'static str,
}

#[async_trait]
impl Provider for NamedProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[crate::message::ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        unimplemented!("NamedProvider")
    }

    fn name(&self) -> &str {
        // Machine-facing transport name. Named OpenAI-compatible profiles keep
        // this as "openrouter" while display_name carries the profile id.
        if self.provider_name.starts_with("profile:") {
            "openrouter"
        } else {
            self.provider_name
        }
    }

    fn display_name(&self) -> String {
        if let Some(profile) = self.provider_name.strip_prefix("profile:") {
            profile.to_string()
        } else {
            self.provider_name.to_string()
        }
    }

    fn model(&self) -> String {
        self.provider_model.to_string()
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

fn create_named_test_app(provider_name: &'static str, provider_model: &'static str) -> App {
    ensure_test_jcode_home_if_unset();
    clear_persisted_test_ui_state();
    crate::tui::ui::clear_test_render_state_for_tests();

    let provider: Arc<dyn Provider> = Arc::new(NamedProvider {
        provider_name,
        provider_model,
    });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let registry = rt.block_on(Registry::new(provider.clone()));
    let mut app = App::new_for_test_harness(provider, registry);
    app.queue_mode = false;
    app.diff_mode = crate::config::DiffDisplayMode::Inline;
    app
}

#[test]
fn canonical_identity_separates_same_model_by_provider() {
    let mut openai_app = create_named_test_app("openai", "gpt-5.4");
    openai_app.session.provider_key = Some("openai".to_string());
    openai_app.session.route_api_method = Some("openai-api".to_string());
    openai_app.session.model_identity_format = Some(1);
    let openai_identity =
        crate::tui::app::model_context::model_route_metadata::canonical_session_provider_model_identity(
            &openai_app,
        );
    assert!(
        openai_identity.starts_with("openai/"),
        "openai identity should be namespaced: {openai_identity}"
    );

    let mut claude_app = create_named_test_app("claude", "gpt-5.4");
    claude_app.session.provider_key = Some("claude".to_string());
    claude_app.session.route_api_method = Some("claude-api".to_string());
    claude_app.session.model_identity_format = Some(1);
    let claude_identity =
        crate::tui::app::model_context::model_route_metadata::canonical_session_provider_model_identity(
            &claude_app,
        );
    assert!(
        claude_identity.starts_with("claude/"),
        "claude identity should be namespaced: {claude_identity}"
    );

    assert_ne!(
        openai_identity, claude_identity,
        "same model id from different providers must produce distinct canonical identities"
    );
}

#[test]
fn pricing_memo_key_separates_same_model_by_provider_identity() {
    let mut openai_app = create_named_test_app("openai", "gpt-5.4");
    openai_app.session.provider_key = Some("openai".to_string());
    openai_app.session.route_api_method = Some("openai-api".to_string());
    openai_app.session.model_identity_format = Some(1);
    let openai_key = format!(
        "{}|{}",
        crate::tui::app::model_context::model_route_metadata::canonical_session_provider_model_identity(
            &openai_app,
        ),
        "standard"
    );

    let mut claude_app = create_named_test_app("claude", "gpt-5.4");
    claude_app.session.provider_key = Some("claude".to_string());
    claude_app.session.route_api_method = Some("claude-api".to_string());
    claude_app.session.model_identity_format = Some(1);
    let claude_key = format!(
        "{}|{}",
        crate::tui::app::model_context::model_route_metadata::canonical_session_provider_model_identity(
            &claude_app,
        ),
        "standard"
    );

    assert_ne!(
        openai_key, claude_key,
        "per-call pricing memo key must include provider namespace so the same model id does not share state across origins"
    );
}

#[test]
fn route_metadata_source_identity_tracks_named_profile_switch_without_env_swap() {
    let _env_lock = crate::storage::lock_test_env();
    let previous_profile = std::env::var_os("JCODE_NAMED_PROVIDER_PROFILE");
    let previous_runtime = std::env::var_os("JCODE_RUNTIME_PROVIDER");
    crate::env::set_var("JCODE_NAMED_PROVIDER_PROFILE", "openai-proxy");
    crate::env::set_var("JCODE_RUNTIME_PROVIDER", "openai");

    let mut app = create_named_test_app("openrouter", "shared-model");
    app.session.model = Some("profile-a/shared-model".to_string());
    app.session.provider_key = Some("profile-a".to_string());
    app.session.route_api_method = Some("openai-compatible:profile-a".to_string());
    app.session.model_identity_format = Some(1);
    let source_a = app
        .active_route_source_identity()
        .expect("route A source identity");

    // Keep the environment unchanged: only durable route metadata changes.
    app.session.model = Some("openai-proxy/shared-model".to_string());
    app.session.provider_key = Some("openai-proxy".to_string());
    app.session.route_api_method = Some("openai-compatible:openai-proxy".to_string());
    let source_b = app
        .active_route_source_identity()
        .expect("route B source identity");

    assert_eq!(source_a.source_key, "named-profile:profile-a");
    assert_eq!(source_b.source_key, "named-profile:openai-proxy");
    assert!(!source_b.is_openai);
    assert_ne!(source_a.source_key, source_b.source_key);

    if let Some(value) = previous_profile {
        crate::env::set_var("JCODE_NAMED_PROVIDER_PROFILE", value);
    } else {
        crate::env::remove_var("JCODE_NAMED_PROVIDER_PROFILE");
    }
    if let Some(value) = previous_runtime {
        crate::env::set_var("JCODE_RUNTIME_PROVIDER", value);
    } else {
        crate::env::remove_var("JCODE_RUNTIME_PROVIDER");
    }
}

#[test]
fn tui_startup_and_clear_stamp_named_profile_session_identity() {
    // App::new / /clear must stamp canonical provider/model identity for named
    // OpenAI-compatible profiles even though provider.name() is "openrouter".
    let _env_lock = crate::storage::lock_test_env();
    let previous_runtime = std::env::var_os("JCODE_RUNTIME_PROVIDER");
    crate::env::set_var("JCODE_RUNTIME_PROVIDER", "gemini");

    let mut app = create_named_test_app("profile:prov-a", "gpt-5.6-sol");
    assert_eq!(app.provider.name(), "openrouter");
    assert_eq!(app.provider.display_name(), "prov-a");
    assert_eq!(app.session.model.as_deref(), Some("prov-a/gpt-5.6-sol"));
    assert_eq!(app.session.provider_key.as_deref(), Some("prov-a"));
    assert_eq!(app.session.model_identity_format, Some(1));

    // Pollute session fields then clear; /clear must re-stamp from live provider.
    app.session.model = Some("openrouter/gpt-5.6-sol".to_string());
    app.session.provider_key = Some("openrouter".to_string());
    app.session.model_identity_format = Some(1);
    assert!(super::commands::handle_session_command(&mut app, "/clear"));
    assert_eq!(app.session.model.as_deref(), Some("prov-a/gpt-5.6-sol"));
    assert_eq!(app.session.provider_key.as_deref(), Some("prov-a"));
    assert_eq!(app.session.model_identity_format, Some(1));

    if let Some(value) = previous_runtime {
        crate::env::set_var("JCODE_RUNTIME_PROVIDER", value);
    } else {
        crate::env::remove_var("JCODE_RUNTIME_PROVIDER");
    }
}

#[test]
fn fill_missing_does_not_promote_legacy_openrouter_identity_marker() {
    // Resume path: older sessions may have model+provider_key but no
    // model_identity_format. Promoting only the marker would reinterpret
    // historical OpenRouter ids (openrouter/custom-model).
    let _env_lock = crate::storage::lock_test_env();
    let mut session = crate::session::Session::create(None, None);
    session.model = Some("openrouter/custom-model".to_string());
    session.provider_key = Some("openrouter".to_string());
    session.route_api_method = Some("openrouter".to_string());
    session.model_identity_format = None;

    let provider: Arc<dyn Provider> = Arc::new(NamedProvider {
        provider_name: "openrouter",
        provider_model: "anthropic/claude-sonnet-4",
    });
    crate::tui::app::model_context::model_route_metadata::fill_missing_session_route_metadata_from_provider(
        &mut session,
        provider.as_ref(),
    );

    assert_eq!(session.model.as_deref(), Some("openrouter/custom-model"));
    assert_eq!(session.provider_key.as_deref(), Some("openrouter"));
    assert_eq!(session.model_identity_format, None);

    // Historical restore request must still double-wrap under openrouter:
    let request = crate::provider::MultiProvider::model_switch_request_for_session_route(
        session.model.as_deref().expect("model"),
        session.provider_key.as_deref(),
        session.route_api_method.as_deref(),
    );
    assert_eq!(request, "openrouter:openrouter/custom-model");
}

#[test]
fn fill_missing_promotes_format_only_after_recanonicalizing_existing_model() {
    // When provider_key is missing (so fill has real work), promoting to format
    // 1 must re-canonicalize any pre-existing model under the old marker first.
    let _env_lock = crate::storage::lock_test_env();
    let mut session = crate::session::Session::create(None, None);
    session.model = Some("openrouter/custom-model".to_string());
    session.provider_key = None;
    session.route_api_method = Some("openrouter".to_string());
    session.model_identity_format = None;

    let provider: Arc<dyn Provider> = Arc::new(NamedProvider {
        provider_name: "openrouter",
        provider_model: "openrouter/gpt-5.4",
    });
    crate::tui::app::model_context::model_route_metadata::fill_missing_session_route_metadata_from_provider(
        &mut session,
        provider.as_ref(),
    );

    assert_eq!(session.provider_key.as_deref(), Some("openrouter"));
    assert_eq!(session.model_identity_format, Some(1));
    // Under format-none, openrouter/custom-model is re-canonicalized to the
    // historical outer-namespace form before the marker becomes 1.
    assert_eq!(
        session.model.as_deref(),
        Some("openrouter/openrouter/custom-model")
    );
}

#[test]
fn fill_missing_fresh_openrouter_model_stays_format_one_without_double_wrap() {
    // Missing model gets filled from the live provider as a format-1 identity.
    // Re-running that fresh value under format-none would double-wrap
    // one-segment OpenRouter ids.
    let _env_lock = crate::storage::lock_test_env();
    let mut session = crate::session::Session::create(None, None);
    session.model = None;
    session.provider_key = Some("openrouter".to_string());
    session.route_api_method = Some("openrouter".to_string());
    session.model_identity_format = None;

    let provider: Arc<dyn Provider> = Arc::new(NamedProvider {
        provider_name: "openrouter",
        provider_model: "custom-model",
    });
    crate::tui::app::model_context::model_route_metadata::fill_missing_session_route_metadata_from_provider(
        &mut session,
        provider.as_ref(),
    );

    assert_eq!(session.model.as_deref(), Some("openrouter/custom-model"));
    assert_eq!(session.provider_key.as_deref(), Some("openrouter"));
    assert_eq!(session.model_identity_format, Some(1));
    assert_ne!(
        session.model.as_deref(),
        Some("openrouter/openrouter/custom-model")
    );
}

#[test]
fn tui_restore_request_honors_format_one_openrouter_identity() {
    // Mirrors tui_lifecycle_runtime restore_session model_request construction.
    let request =
        crate::provider::MultiProvider::model_switch_request_for_session_route_with_identity_format(
            "openrouter/custom-model",
            Some("openrouter"),
            Some("openrouter"),
            Some(1),
        );
    assert_eq!(request, "openrouter:custom-model");

    let legacy =
        crate::provider::MultiProvider::model_switch_request_for_session_route_with_identity_format(
            "openrouter/custom-model",
            Some("openrouter"),
            Some("openrouter"),
            None,
        );
    assert_eq!(legacy, "openrouter:openrouter/custom-model");
}
