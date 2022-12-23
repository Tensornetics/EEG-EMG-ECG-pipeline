extern crate mysql;
extern crate clap;
extern crate tensorflow;
extern crate gnuplot;

use mysql as my;
use clap::{Arg, App};
use tensorflow as tf;
use gnuplot::{Figure, Caption, Color};
use std::env;

fn main() {
    let matches = App::new("Data Processing")
        .version("1.0")
        .author("Your Name <your@email.com>")
        .about("Accepts ECG, EMG, and EEG data sets and stores them to an active database, then uses an inference engine to model, forecast, and predict")
        .arg(Arg::with_name("ecg")
            .short("e")
            .long("ecg")
            .value_name("ECG")
            .help("ECG data set")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("emg")
            .short("m")
            .long("emg")
            .value_name("EMG")
            .help("EMG data set")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("eeg")
            .short("g")
            .long("eeg")
            .value_name("EEG")
            .help("EEG data set")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("prediction")
            .short("p")
            .long("prediction")
            .value_name("PREDICTION")
            .help("Prediction value")
            .takes_value(true)
            .required(true))
        .get_matches();

    let ecg: f64 = matches.value_of("ecg").unwrap().parse().unwrap();
    let emg: f64 = matches.value_of("emg").unwrap().parse().unwrap();
    let eeg: f64 = matches.value_of("eeg").unwrap().parse().unwrap();
    let prediction: f64 = matches.value_of("prediction").unwrap().parse().unwrap();

    let url = "mysql://root:password@localhost:3306";
    let pool = match my::Pool::new(url) {
        Ok(pool) => pool,
        Err(err) => panic!("Unable to connect to database: {}",