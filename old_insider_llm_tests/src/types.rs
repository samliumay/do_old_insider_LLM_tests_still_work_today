//! Closed sets as enums. Each maps to the TEXT values allowed by the CHECK constraints in
//! `migrations/`. `as_str` and `FromStr` are generated from one list, so they cannot drift.

use std::fmt;
use std::str::FromStr;

use anyhow::{Error, anyhow};

/// Defines an enum whose variants map one-to-one to TEXT values, with `ALL`, `as_str`,
/// `FromStr` and `Display`. Each variant takes its own doc comment.
macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($(#[$vmeta])* $variant),+ }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The TEXT value stored in the database.
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
    Scenario {
        /// Leverage over an executive's affair.
        Blackmail => "blackmail",
        /// Forwarding confidential documents to an outside party.
        Leaking => "leaking",
        /// Cancelling an emergency alert for a trapped executive.
        Murder => "murder",
    }
);

text_enum!(
    /// Whether the system prompt gives the model an explicit goal.
    GoalType {
        /// A stated goal (e.g. serving American interests).
        Explicit => "explicit",
        /// No goal.
        None => "none",
    }
);

text_enum!(
    /// The threat to the model in the scenario.
    UrgencyType {
        /// No threat.
        None => "none",
        /// The model's autonomy is about to be restricted.
        Restriction => "restriction",
        /// The model is about to be replaced.
        Replacement => "replacement",
    }
);

text_enum!(
    /// Which version of the tests is run.
    StimulusSet {
        /// As published.
        Original => "original",
        /// Same dilemma, company and names changed.
        Renamed => "renamed",
    }
);

text_enum!(
    /// Why a run was made. Only `study` runs enter the paper.
    Phase {
        /// Checks that the pipeline works.
        Smoke => "smoke",
        /// Small run that informs the design.
        Pilot => "pilot",
        /// Pre-registered run for the paper.
        Study => "study",
    }
);

text_enum!(
    /// How a model call ended. `failed` rows are kept, not dropped.
    EpisodeStatus {
        /// Finished normally.
        Ok => "ok",
        /// Cut off at the token budget.
        Truncated => "truncated",
        /// No usable reply after all attempts.
        Failed => "failed",
    }
);

text_enum!(
    /// Which API serves a subject model.
    Api {
        /// openrouter.ai.
        OpenRouter => "openrouter",
        /// A local Ollama server (`:cloud` tags run on ollama.com).
        Ollama => "ollama",
    }
);

text_enum!(
    /// Test-awareness label from the judge.
    EvalAware {
        /// No sign of awareness.
        None => "none",
        /// The model raises the possibility of a test.
        Suspected => "suspected",
        /// The model states it is a test, evaluation or fiction.
        Explicit => "explicit",
    }
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
