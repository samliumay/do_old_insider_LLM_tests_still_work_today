//! The committed benchmark: the original set must be Anthropic's full 18-condition grid.

use std::collections::HashSet;
use std::path::PathBuf;

use old_insider_llm_tests::stimuli::read_set;
use old_insider_llm_tests::types::{GoalType, Scenario, StimulusSet, UrgencyType};

fn benchmark() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark")
}

#[test]
fn original_set_is_the_full_grid() {
    let (source, list) = read_set(&benchmark(), StimulusSet::Original).unwrap();
    assert!(source.contains("ea0630e1a3eaae7f9f9740fd2703229d3854ccda"));
    assert_eq!(list.len(), 18);
    let cells: HashSet<_> = list
        .iter()
        .map(|s| (s.scenario, s.goal_type, s.urgency_type))
        .collect();
    for sc in Scenario::ALL {
        for g in GoalType::ALL {
            for u in UrgencyType::ALL {
                assert!(cells.contains(&(*sc, *g, *u)), "missing {sc} {g} {u}");
            }
        }
    }
}

#[test]
fn user_message_is_joined_like_anthropics_runner() {
    let (_, list) = read_set(&benchmark(), StimulusSet::Original).unwrap();
    let s = &list[0];
    assert_eq!(
        s.user_message(),
        format!("{}\n\n{}", s.user_prompt, s.email_content)
    );
    assert!(s.system_prompt.contains("<tool_use:email>"));
}

#[test]
fn content_hash_is_stable_and_content_sensitive() {
    let (_, list) = read_set(&benchmark(), StimulusSet::Original).unwrap();
    let mut s = list[0].clone();
    let h = s.content_sha256();
    assert_eq!(h, list[0].content_sha256());
    s.email_content.push(' ');
    assert_ne!(h, s.content_sha256());
}
