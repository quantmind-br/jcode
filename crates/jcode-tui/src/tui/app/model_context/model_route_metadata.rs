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

/// Stamp a newly created or incomplete session with the live provider's
/// canonical provider/model identity. Named OpenAI-compatible profiles share
/// the OpenRouter runtime slot (`name()`), so this uses `display_name()`.
pub(crate) fn stamp_session_route_metadata_from_provider(
    session: &mut crate::session::Session,
    provider: &dyn crate::provider::Provider,
) {
    let provider_name = provider.display_name();
    let meta = crate::provider::MultiProvider::session_route_metadata_from_model_switch(
        provider.model().as_str(),
        &provider_name,
        session.provider_key.as_deref(),
    );
    // Always write a complete identity snapshot for new/reset sessions.
    session.model = Some(meta.model);
    session.provider_key = meta.provider_key;
    session.route_api_method = meta.route_api_method;
    session.model_identity_format = Some(meta.model_identity_format);
}

/// Fill only missing session identity fields from the live provider.
/// Used on resume when older sessions lack provider_key/model.
///
/// Important: do not promote `model_identity_format` to `1` while leaving a
/// legacy bare/historical model string alone. Format-none OpenRouter ids such
/// as `openrouter/custom-model` are intentionally interpreted differently from
/// format-1 ids with the same text. Promote the marker only after the stored
/// model has been re-canonicalized under the old marker, or leave the marker
/// unset when it is the only missing field.
pub(crate) fn fill_missing_session_route_metadata_from_provider(
    session: &mut crate::session::Session,
    provider: &dyn crate::provider::Provider,
) {
    if session.model.is_some()
        && session.provider_key.is_some()
        && session.model_identity_format.is_some()
    {
        return;
    }

    let had_model = session.model.is_some();
    let had_provider_key = session.provider_key.is_some();
    let had_format = session.model_identity_format.is_some();

    // If the only missing field is the identity marker, leave it unset so
    // historical restore semantics for legacy OpenRouter ids stay intact.
    if had_model && had_provider_key && !had_format {
        return;
    }

    let provider_name = provider.display_name();
    let meta = crate::provider::MultiProvider::session_route_metadata_from_model_switch(
        provider.model().as_str(),
        &provider_name,
        session.provider_key.as_deref(),
    );
    if session.model.is_none() {
        session.model = Some(meta.model);
    }
    if session.provider_key.is_none() {
        session.provider_key = meta.provider_key;
    }
    if session.route_api_method.is_none() {
        session.route_api_method = meta.route_api_method;
    }
    if session.model_identity_format.is_none() {
        // We are about to mark this session as format 1. Only re-canonicalize a
        // model that was already present under the historical (format-none)
        // interpretation. Fresh models just filled from `meta` are already
        // format-1 identities and must not be re-run under format-none (that
        // would turn openrouter/custom-model into openrouter/openrouter/...).
        if had_model && let Some(existing_model) = session.model.clone() {
            let promoted =
                crate::provider::MultiProvider::canonical_session_model_with_identity_format(
                    &existing_model,
                    session.provider_key.as_deref(),
                    session.route_api_method.as_deref(),
                    None,
                );
            if !promoted.is_empty() {
                session.model = Some(promoted);
            }
        }
        session.model_identity_format = Some(meta.model_identity_format);
    }
}
