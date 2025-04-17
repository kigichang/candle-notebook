use anyhow::Result;
use std::collections::HashMap;

#[test]
fn conv_pth_to_safetensor() -> Result<()> {
    let pth_vec = candle_core::pickle::read_all("fix-bert-base-chinese.pth")?;
    for item in &pth_vec {
        println!("{:?}", item.0);
    }
    let mut tensor_map = HashMap::new();

    for item in pth_vec {
        println!("{:?}:{:?}", item.0, item.1);
        tensor_map.insert(item.0, item.1);
    }

    candle_core::safetensors::save(&tensor_map, "fix-bert-base-chinese.safetensors")?;
    Ok(())
}
