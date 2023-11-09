use std::{fs::File, usize, process::exit, io::Write};

use ini::{Properties, Ini};
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};
use ndarray::{s, Array2};

// Load the ini file
fn ini_config(sec_choice: &str) -> Properties {
    let config = Ini::load_from_file("training-config.ini").expect("Failed to parse config data");

    match sec_choice {
        "main" => {
            let section = config.section(Some("main")).expect("Section not found");
            section.clone()
        }
        "svm" => {
            let section = config.section(Some("svm")).expect("Section not found");
            section.clone()
        } 
        _ => {
            let section = config.section(Some("main")).expect("Section not found");
            section.clone()
        }
    }
}

// Save the model
fn save_model(model: Result<Svm<f64, bool>>) -> core::result::Result<(), Box<dyn std::error::Error>> {
    let conf = ini_config("main");

    let save_location = conf.get("save_location").expect("Key not found");

    let serialized = serde_json::to_string(&model.unwrap())?;
    let mut file = File::create(save_location)?;
    file.write_all(serialized.as_bytes())?;
    println!("Model saved: {:?}", save_location);
    Ok(())
}
fn main() -> Result<()> {
    // Load ini variables
    let conf_main = ini_config("main");
    let conf_svm = ini_config("svm");

    let training_data = conf_main.get("training_data").unwrap();
    let feature_count = conf_main.get("feature_count").unwrap().parse::<usize>().unwrap();
    let data_headers = conf_main.get("data_headers").unwrap().parse::<bool>().unwrap();
    let target_float = conf_main.get("target_float").unwrap().parse::<f64>().unwrap();
    let model_kernal = conf_svm.get("model_kernal").unwrap();
    let eps = conf_svm.get("eps").unwrap().parse::<f64>().unwrap();
    let shrinking = conf_svm.get("shrinking").unwrap().parse::<bool>().unwrap();
    let c_pos = conf_svm.get("c_pos").unwrap().parse::<f64>().unwrap();
    let c_neg = conf_svm.get("c_neg").unwrap().parse::<f64>().unwrap();
    let nu_weight = conf_svm.get("nu_weight").unwrap().parse::<f64>().unwrap();
    let gaussian_kernal = conf_svm.get("gaussian_kernal").unwrap().parse::<f64>().unwrap();

    // Load training data in an array
    let training_data = File::open(training_data).unwrap();
    let training_data = linfa_datasets::array_from_csv(training_data, data_headers , b',').unwrap();

    // Verify feature count
    if feature_count != training_data.ncols()-1 {
        println!("Feature count does not match data provided.");
        exit(1);
    }

    // Seperate features and targets
    let mut features = Array2::zeros((0, feature_count));
    let targets = training_data.column(feature_count).into_owned();

    for i in training_data.outer_iter() {
        features.push_row(i.slice(s![..feature_count; 1])).err();
    }

    // Put features and targets into a DatasetBase
    let training_dataset = Dataset::new(features, targets).map_targets(|x| *x > target_float);

    // train the model using the configs
    // TOODO: figure out what and how to use the rest of this.
    let model = match model_kernal {
        "gaussian" => Svm::<_, bool>::params()
            .eps(eps)
            .shrinking(shrinking)
            // .with_kernel_params(kernel)
            // .with_platt_params(platt)
            .pos_neg_weights(c_pos, c_neg)
            .nu_weight(nu_weight)
            .gaussian_kernel(gaussian_kernal)
            .fit(&training_dataset),
        // "polynomial" => Svm::<_, bool>::params()
        //     .eps(eps)
        //     .shrinking(shrinking)
        //     // .with_kernel_params(kernel)
        //     // .with_platt_params(platt)
        //     .pos_neg_weights(c_pos, c_neg)
        //     .nu_weight(nu_weight)
        //     .polynomial_kernel(constant, degree)
        //     .fit(&training_dataset),
        "linear" => Svm::<_, bool>::params()
            .eps(eps)
            .shrinking(shrinking)
            // .with_kernel_params(kernel)
            // .with_platt_params(platt)
            .pos_neg_weights(c_pos, c_neg)
            .nu_weight(nu_weight)
            .linear_kernel()
            .fit(&training_dataset),
        _ => {
            println!("Unknown model kernal");
            exit(2);
        },
    };

    // Save the model
    save_model(model).err();

    Ok(())
}