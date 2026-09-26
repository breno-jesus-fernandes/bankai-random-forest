use std::process::Command;

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
