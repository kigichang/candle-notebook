use anyhow::Result;
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::bert::{BertForMaskedLM, Config};

fn main() -> Result<()> {
    let data = unsafe { candle_core::safetensors::MmapedSafetensors::new("model.safetensors")? };

    let config = std::fs::read_to_string("config.json")?;
    let config: Config = serde_json::from_str(&config)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &candle_core::Device::Cpu);
    BertForMaskedLM::load(vs, &config)?;

    println!("{}", varmap.all_vars().len());
    data.tensors().iter().for_each(|(name, _)| {
        //println!("{:?} {:?}", name, tensor.shape());
        let new_name = if name.ends_with("beta") {
            name.replace("beta", "bias")
        } else if name.ends_with("gamma") {
            name.replace("gamma", "weight")
        } else {
            name.to_string()
        };

        let tensor = data.load(&name, &candle_core::Device::Cpu).unwrap();

        let _ = varmap.set_one(&new_name, tensor);
    });

    varmap.save("new-model.safetensors")?;

    let vs = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["new-model.safetensors"],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?
    };

    let bert = BertForMaskedLM::load(vs, &config)?;

    let data =
        unsafe { candle_core::safetensors::MmapedSafetensors::new("new-model.safetensors")? };

    data.tensors().iter().for_each(|(name, _)| {
        if name.ends_with("beta") || name.ends_with("gamma") {
            panic!("should not have beta or gamma in the name");
        }
    });
    Ok(())
}
