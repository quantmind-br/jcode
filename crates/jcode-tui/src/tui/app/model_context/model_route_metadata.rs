use super::*;

/// Canonical provider/model identity for the active session, using the persisted
/// session identity when available and falling back to the live provider.
/// This is the stable key for pricing, usage, and render memoization.
pub(crate) fn canonical_session_provider_model_identity(app: &App) -> String {
    let live_model = app.provider.model();
    let model = app
        .session
        .model
        .as_deref()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or(&live_model);
    crate::provider::MultiProvider::canonical_session_model_with_identity_format(
        model,
        app.session.provider_key.as_deref(),
        app.session.route_api_method.as_deref(),
        app.session.model_identity_format,
    )
}

/// Atomically persist canonical session route metadata and notify the user
/// if persistence fails.
pub(crate) fn apply_session_route_metadata(
    app: &mut App,
    model: String,
    provider_key: Option<String>,
    route_api_method: Option<String>,
    model_identity_format: u8,
) {
    app.session.model = Some(model);
    app.session.provider_key = provider_key;
    app.session.route_api_method = route_api_method;
    app.session.model_identity_format = Some(model_identity_format);
    if let Err(e) = app.session.save() {
        app.push_display_message(DisplayMessage::error(format!(
            "Failed to persist session: {e}"
        )));
    }
}

/// Apply session route metadata for a remote model switch after the server
/// confirms it. Preserves an explicit picker API method if available, and
/// otherwise keeps the old route_api_method only when the canonical identity
/// is unchanged.
pub(crate) fn apply_remote_model_switch_metadata(
    app: &mut App,
    model: &str,
    provider_name: Option<String>,
) {
    let old = crate::provider::MultiProvider::canonical_session_model_with_identity_format(
        app.session.model.as_deref().unwrap_or(""),
        app.session.provider_key.as_deref(),
        app.session.route_api_method.as_deref(),
        app.session.model_identity_format,
    );
    if let Some(selection) = app.remote_model_switch_route_selection.take() {
        let meta =
            crate::provider::MultiProvider::session_route_metadata_from_selection(&selection);
        apply_session_route_metadata(
            app,
            meta.model,
            meta.provider_key,
            meta.route_api_method,
            meta.model_identity_format,
        );
        return;
    }
    let active_provider = provider_name
        .or_else(|| app.remote_provider_name.clone())
        .unwrap_or_else(|| "remote".to_string());
    let meta = crate::provider::MultiProvider::session_route_metadata_from_model_switch(
        model,
        &active_provider,
        app.session.provider_key.as_deref(),
    );
    let preserve_method = !old.is_empty() && old == meta.model;
    apply_session_route_metadata(
        app,
        meta.model,
        meta.provider_key,
        if preserve_method {
            app.session.route_api_method.clone()
        } else {
            meta.route_api_method
        },
        meta.model_identity_format,
    );
}
