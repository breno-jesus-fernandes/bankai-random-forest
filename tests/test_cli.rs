use std::process::Command;
use std::{env, fs};

#[test]
fn cli_emits_csv_benchmark_measurement() {
    let output = Command::new(env!("CARGO_BIN_EXE_bankai-xrf-cli"))
        .args(["--rows", "32", "--features", "2", "--trees", "3"])
        .output()
        .expect("benchmark CLI should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("CLI output should be UTF-8");
    assert!(stdout.starts_with("rows,features,trees,train_seconds,predict_seconds,f1"));
}

#[test]
fn cli_exports_predictions_for_comparative_benchmarks() {
    let path = env::temp_dir().join(format!("bankai-predictions-{}.csv", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_bankai-xrf-cli"))
        .args([
            "--rows",
            "32",
            "--features",
            "2",
            "--trees",
            "3",
            "--predictions",
            path.to_str().expect("temporary path should be UTF-8"),
        ])
        .output()
        .expect("benchmark CLI should run");

    assert!(output.status.success());
    let predictions = fs::read_to_string(&path).expect("CLI should write predictions");
    assert!(predictions.starts_with("actual,predicted,probability_0,probability_1"));
    fs::remove_file(path).expect("temporary predictions should be removable");
}

#[test]
fn cli_exports_native_permutation_importances() {
    let path = env::temp_dir().join(format!("bankai-importances-{}.csv", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_bankai-xrf-cli"))
        .args([
            "--rows",
            "64",
            "--features",
            "3",
            "--trees",
            "5",
            "--permutation-importance",
            "--importances",
            path.to_str().expect("temporary path should be UTF-8"),
        ])
        .output()
        .expect("benchmark CLI should run");

    assert!(output.status.success());
    let importances = fs::read_to_string(&path).expect("CLI should write importances");
    assert!(importances.starts_with("feature,importance"));
    assert_eq!(importances.lines().count(), 4);
    fs::remove_file(path).expect("temporary importances should be removable");
}

#[test]
fn cli_exports_split_importances() {
    let path = env::temp_dir().join(format!("bankai-splits-{}.csv", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_bankai-xrf-cli"))
        .args([
            "--rows",
            "64",
            "--features",
            "3",
            "--trees",
            "5",
            "--split-importances",
            path.to_str().expect("temporary path should be UTF-8"),
        ])
        .output()
        .expect("benchmark CLI should run");

    assert!(output.status.success());
    let importances = fs::read_to_string(&path).expect("CLI should write importances");
    assert!(importances.starts_with("feature,importance"));
    assert_eq!(importances.lines().count(), 4);
    fs::remove_file(path).expect("temporary importances should be removable");
}

#[test]
fn cli_exports_gain_importances() {
    let path = env::temp_dir().join(format!("bankai-gains-{}.csv", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_bankai-xrf-cli"))
        .args([
            "--rows",
            "64",
            "--features",
            "3",
            "--trees",
            "5",
            "--gain-importances",
            path.to_str().expect("temporary path should be UTF-8"),
        ])
        .output()
        .expect("benchmark CLI should run");

    assert!(output.status.success());
    let importances = fs::read_to_string(&path).expect("CLI should write importances");
    assert!(importances.starts_with("feature,importance"));
    assert_eq!(importances.lines().count(), 4);
    fs::remove_file(path).expect("temporary importances should be removable");
}
