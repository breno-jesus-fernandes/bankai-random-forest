use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use bankai_random_forest::dense::{Criterion, DenseInput};
use xrf::Forest;

struct Arguments {
    rows: usize,
    features: usize,
    trees: usize,
    markdown: bool,
    predictions: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments(env::args().skip(1))?;
    let (values, labels) = generate_dataset(arguments.rows, arguments.features);
    let weights = vec![1.0; arguments.rows];
    let training = DenseInput::training(
        values.clone(),
        arguments.rows,
        arguments.features,
        labels.clone(),
        weights,
        2,
        0.0,
        Criterion::Gini,
        2,
        1,
        0.0,
    )?;

    let train_started = Instant::now();
    let forest = Forest::new_with_settings(
        &training,
        arguments.trees,
        (arguments.features as f64).sqrt().max(1.0) as usize,
        true,
        false,
        false,
        42,
        512,
        usize::MAX,
        true,
        None,
    );
    let train_seconds = train_started.elapsed().as_secs_f64();

    let prediction = DenseInput::prediction(values, arguments.rows, arguments.features, 2)?;
    let predict_started = Instant::now();
    let probabilities: Vec<_> = forest
        .predict(&prediction)
        .predictions()
        .map(|(_, votes)| votes.probabilities())
        .collect();
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    let predicted: Vec<_> = probabilities
        .iter()
        .map(|probabilities| if probabilities[1] > probabilities[0] { 1 } else { 0 })
        .collect();
    let f1 = binary_f1(&labels, &predicted);

    if let Some(path) = arguments.predictions {
        let mut contents = String::from("actual,predicted,probability_0,probability_1\n");
        for ((actual, predicted), probabilities) in labels.iter().zip(&predicted).zip(&probabilities)
        {
            contents.push_str(&format!(
                "{actual},{predicted},{:.12},{:.12}\n",
                probabilities[0], probabilities[1]
            ));
        }
        std::fs::write(path, contents).map_err(|error| error.to_string())?;
    }

    if arguments.markdown {
        println!("| rows | features | trees | train_seconds | predict_seconds | f1 |");
        println!("| ---: | ---: | ---: | ---: | ---: | ---: |");
        println!(
            "| {} | {} | {} | {:.6} | {:.6} | {:.6} |",
            arguments.rows, arguments.features, arguments.trees, train_seconds, predict_seconds, f1
        );
    } else {
        println!("rows,features,trees,train_seconds,predict_seconds,f1");
        println!(
            "{},{},{},{:.6},{:.6},{:.6}",
            arguments.rows, arguments.features, arguments.trees, train_seconds, predict_seconds, f1
        );
    }
    Ok(())
}

fn parse_arguments(arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let mut parsed = Arguments {
        rows: 10_000,
        features: 20,
        trees: 100,
        markdown: false,
        predictions: None,
    };
    let mut arguments = arguments;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--rows" => parsed.rows = parse_positive(arguments.next(), "--rows")?,
            "--features" => parsed.features = parse_positive(arguments.next(), "--features")?,
            "--trees" => parsed.trees = parse_positive(arguments.next(), "--trees")?,
            "--markdown" => parsed.markdown = true,
            "--predictions" => {
                parsed.predictions = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--predictions requires a path".to_string())?,
                ))
            }
            "--help" | "-h" => {
                return Err(
                    "usage: bankai-xrf-cli [--rows N] [--features N] [--trees N] [--markdown] [--predictions PATH]"
                        .to_string(),
                )
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(parsed)
}

fn parse_positive(value: Option<String>, name: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<usize>()
        .map_err(|_| format!("{name} requires a positive integer"))
        .and_then(|value| {
            if value == 0 {
                Err(format!("{name} requires a positive integer"))
            } else {
                Ok(value)
            }
        })
}

fn generate_dataset(rows: usize, features: usize) -> (Vec<f64>, Vec<usize>) {
    let mut state = 42_u64;
    let mut values = Vec::with_capacity(rows * features);
    let mut labels = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut score = 0.0;
        for feature in 0..features {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let value = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            if feature < 3 {
                score += value;
            }
            values.push(value);
        }
        labels.push(usize::from(score > 0.0));
    }
    (values, labels)
}

fn binary_f1(actual: &[usize], predicted: &[usize]) -> f64 {
    let (mut true_positive, mut false_positive, mut false_negative) = (0_usize, 0_usize, 0_usize);
    for (&actual, &predicted) in actual.iter().zip(predicted) {
        match (actual, predicted) {
            (1, 1) => true_positive += 1,
            (0, 1) => false_positive += 1,
            (1, 0) => false_negative += 1,
            _ => {}
        }
    }
    let denominator = 2 * true_positive + false_positive + false_negative;
    if denominator == 0 { 0.0 } else { 2.0 * true_positive as f64 / denominator as f64 }
}
