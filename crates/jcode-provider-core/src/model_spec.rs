/// The identity syntax used by model selection.
///
/// A slash is the canonical separator: only the first slash is structural and
/// everything after it belongs to the opaque model id. A colon is retained as
/// a legacy separator for provider/profile model requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpecSeparator {
    Slash,
    LegacyColon,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub provider: Option<String>,
    pub model: String,
    pub separator: Option<ModelSpecSeparator>,
}

impl ModelSpec {
    pub fn is_qualified(&self) -> bool {
        self.provider.is_some()
    }
}

/// Parse a model identity without interpreting whether the provider exists.
///
/// Provider resolution belongs to the provider orchestrator because it depends
/// on live configured routes. Parsing here deliberately preserves model paths:
/// `sol/vendor/model` becomes provider `sol` and model `vendor/model`.
pub fn parse_model_spec(raw: &str) -> ModelSpec {
    let trimmed = raw.trim();
    let slash_position = trimmed.find('/');
    let colon_position = trimmed.find(':');

    // A colon that precedes the first slash is the legacy profile separator
    // (`profile:vendor/model`). Colons after the first slash remain part of the
    // opaque canonical model remainder (`provider/model:version`).
    if let Some((provider, model)) = trimmed.split_once(':')
        && slash_position
            .map(|slash_position| colon_position.is_some_and(|colon| colon < slash_position))
            .unwrap_or(true)
        && !provider.trim().is_empty()
        && !model.trim().is_empty()
    {
        return ModelSpec {
            provider: Some(provider.trim().to_string()),
            model: model.trim().to_string(),
            separator: Some(ModelSpecSeparator::LegacyColon),
        };
    }

    if let Some((provider, model)) = trimmed.split_once('/')
        && !provider.trim().is_empty()
        && !model.trim().is_empty()
    {
        return ModelSpec {
            provider: Some(provider.trim().to_string()),
            model: model.trim().to_string(),
            separator: Some(ModelSpecSeparator::Slash),
        };
    }

    ModelSpec {
        provider: None,
        model: trimmed.to_string(),
        separator: None,
    }
}

/// Format the canonical provider/model form.
pub fn format_provider_model(provider: &str, model: &str) -> String {
    format!("{}/{}", provider.trim(), model.trim())
}

#[cfg(test)]
mod tests {
    use super::{ModelSpecSeparator, format_provider_model, parse_model_spec};

    #[test]
    fn canonical_parser_preserves_slashes_in_opaque_model_remainder() {
        let parsed = parse_model_spec(" sol/vendor/model/with/slashes ");

        assert_eq!(parsed.provider.as_deref(), Some("sol"));
        assert_eq!(parsed.model, "vendor/model/with/slashes");
        assert_eq!(parsed.separator, Some(ModelSpecSeparator::Slash));
        assert_eq!(
            format_provider_model(parsed.provider.as_deref().unwrap(), &parsed.model),
            "sol/vendor/model/with/slashes"
        );
    }

    #[test]
    fn legacy_colon_forms_remain_parseable() {
        let parsed = parse_model_spec("profile:vendor/model");

        assert_eq!(parsed.provider.as_deref(), Some("profile"));
        assert_eq!(parsed.model, "vendor/model");
        assert_eq!(parsed.separator, Some(ModelSpecSeparator::LegacyColon));
    }
}
