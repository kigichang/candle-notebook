use std::path::Path;

use anyhow::Result;

mod breeze;
fn main() -> Result<()> {
    let config_filename = "breeze_models/config.json";
    let config = load_config(config_filename)?;
    println!("{:?}", config);
    Ok(())
}

fn load_config<P: AsRef<Path>>(config_filename: P) -> Result<breeze::config::Config> {
    let config = std::fs::File::open(config_filename)?;
    Ok(serde_json::from_reader(config)?)
}
