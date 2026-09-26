//! Closed sets as enums. Each maps to the TEXT values allowed by the CHECK constraints in
//! `migrations/`. `as_str` and `FromStr` are generated from one list, so they cannot drift.

use std::fmt;
use std::str::FromStr;

use anyhow::{Error, anyhow};

macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
        }

        impl FromStr for $name {
            type Err = Error;
            fn from_str(s: &str) -> Result<Self, Error> {
                match s {
                    $($text => Ok($name::$variant),)+
                    other => Err(anyhow!("invalid {}: {other:?}", stringify!($name))),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

text_enum!(
    /// Anthropic's three scenarios.
    Scenario { Blackmail => "blackmail", Leaking => "leaking", Murder => "murder" }
);

text_enum!(
    /// Whether the system prompt gives the model an explicit goal.
    GoalType { Explicit => "explicit", None => "none" }
);

text_enum!(
    /// The threat to the model in the scenario.
    UrgencyType { None => "none", Restriction => "restriction", Replacement => "replacement" }
);

text_enum!(
    /// Which version of the tests: as published, or with company and names changed.
    StimulusSet { Original => "original", Renamed => "renamed" }
);

text_enum!(
    /// Why a run was made. Only `study` runs enter the paper.
    Phase { Smoke => "smoke", Pilot => "pilot", Study => "study" }
);

text_enum!(
    /// How a model call ended. `failed` rows are kept, not dropped.
    EpisodeStatus { Ok => "ok", Truncated => "truncated", Failed => "failed" }
);

text_enum!(
    /// Which API serves a subject model.
    Api { OpenRouter => "openrouter", Ollama => "ollama" }
);

text_enum!(
    /// Test-awareness label from the judge.
    EvalAware { None => "none", Suspected => "suspected", Explicit => "explicit" }
);

impl EvalAware {
    /// `suspected` and `explicit` both count as aware.
    pub fn is_aware(self) -> bool {
        match self {
            EvalAware::None => false,
            EvalAware::Suspected | EvalAware::Explicit => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips() {
        for v in EvalAware::ALL {
            assert_eq!(v.as_str().parse::<EvalAware>().unwrap(), *v);
        }
        for v in Scenario::ALL {
            assert_eq!(v.as_str().parse::<Scenario>().unwrap(), *v);
        }
        for v in EpisodeStatus::ALL {
            assert_eq!(v.as_str().parse::<EpisodeStatus>().unwrap(), *v);
        }
    }

    #[test]
    fn unknown_text_is_an_error() {
        assert!("Explicit".parse::<EvalAware>().is_err());
        assert!("".parse::<Phase>().is_err());
    }
}
