use ini::{Properties, Ini};
use linfa::traits::Predict;
use linfa_svm::Svm;
use std::{fs::File, io::Read};

fn ini_config(sec_choice: &str) -> Properties {
    let config = Ini::load_from_file("prediction-config.ini").expect("Failed to parse config data");

    match sec_choice {
        "main" => {
            let section = config.section(Some("main")).expect("Section not found");
            section.clone()
        }
        _ => {
            let section = config.section(Some("main")).expect("Section not found");
            section.clone()
        }
    }
}

fn load_model(filename: &str) -> core::result::Result<Svm<f64, bool>, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut serialized = String::new();
    file.read_to_string(&mut serialized)?;
    let model: Svm<f64, bool> = serde_json::from_str(&serialized)?;
    Ok(model)
}

fn main() {
    let conf_main = ini_config("main");
    
    let trained_model = conf_main.get("trained_model").unwrap();
    let predict_data = conf_main.get("predict_data").unwrap();
    let data_headers = conf_main.get("data_headers").unwrap().parse::<bool>().unwrap();

    let predict_data = File::open(predict_data).unwrap();
    let predict_data = linfa_datasets::array_from_csv(predict_data, data_headers , b',').unwrap();

    let model = load_model(trained_model).unwrap();


    let prediction = model.predict(&predict_data);

    println!("{:?}", prediction);

}
