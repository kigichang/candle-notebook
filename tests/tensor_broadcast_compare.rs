use candle_core::{Device, Result, Tensor};

#[test]
fn broadcast_eq() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_eq(&t2)?;
    let result_v = candle_notebook::eq_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn broadcast_ne() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_ne(&t2)?;
    let result_v = candle_notebook::ne_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn broadcast_le() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_le(&t2)?;
    let result_v = candle_notebook::le_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn broadcast_lt() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_lt(&t2)?;
    let result_v = candle_notebook::lt_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn broadcast_ge() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_ge(&t2)?;
    let result_v = candle_notebook::ge_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);
    Ok(())
}

#[test]
fn broadcast_gt() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;
    let v2 = t2.broadcast_as((2, 3, 4))?.to_vec3::<f32>()?;

    let result = t1.broadcast_gt(&t2)?;
    let result_v = candle_notebook::gt_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);
    Ok(())
}
