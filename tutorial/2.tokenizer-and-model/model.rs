use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{embedding, linear, Linear, VarBuilder, VarMap};

const IMAGE_DIM: usize = 4;
const HIDDEN_DIM: usize = 3;
const LABELS: usize = 2;

fn print_linear(name: &str, ln: &Linear) -> Result<()> {
    println!("{name}.weight: {:?}", ln.weight().to_vec2::<f32>()?);
    if let Some(bias) = ln.bias() {
        println!("ln.bias: {:?}", bias.to_vec1::<f32>()?);
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let default_model_name = "candle-ex2-2.safetensors";

    {
        // 產生一個空的 VarMap，並使用 VarBuilder 來建立變數，最後使用 save 方法將模型儲存到檔案中。

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
        let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
        let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
        varmap.save(default_model_name)?;

        print_linear("ln1", &ln1)?;
        print_linear("ln2", &ln2)?;
        println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
    }

    {
        // 使用 VarMap 載入預訓練的模型，並使用 VarBuilder 讀取模型內的變數。

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        varmap.load(default_model_name)?;

        let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
        let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
        let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
        print_linear("ln1", &ln1)?;
        print_linear("ln2", &ln2)?;
        println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
    }

    {
        // 直接使用 VarBuilder 來讀取預訓練的模型，並使用 VarBuilder 讀取模型內的變數。

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[default_model_name], DType::F32, &device)?
        };

        let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
        let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
        let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
        print_linear("ln1", &ln1)?;
        print_linear("ln2", &ln2)?;
        println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
    }

    Ok(())
}
