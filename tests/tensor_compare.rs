use candle_core::{Device, Result, Tensor};
use candle_notebook::scalar_to_tensor;

#[test]
fn eq() -> Result<()> {
    // vector
    let t1 = Tensor::rand(-1.0f32, 1.0, 4, &Device::Cpu)?;
    let v1 = t1.to_vec1::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, 4, &Device::Cpu)?;
    let v2 = t2.to_vec1::<f32>()?;

    let result = t1.eq(&t2)?;
    let result_v = candle_notebook::eq_vec1(&v1, &v2);
    assert_eq!(result.to_vec1::<u8>()?, result_v);

    let scalar = v1[0];
    let result = t1.eq(scalar)?;
    assert_eq!(result.to_vec1::<u8>()?, vec![1, 0, 0, 0]);

    let t3 = scalar_to_tensor(v1[0], 4, t1.dtype(), t1.device())?;
    let result_t3 = t1.eq(&t3)?;
    assert_eq!(result_t3.to_vec1::<u8>()?, result.to_vec1::<u8>()?);

    // n-dimension tensor
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.eq(&t2)?;
    let result_v = candle_notebook::eq_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);
    let scalar = v1[0][0][0];
    let result = t1.eq(scalar)?;
    let result_scalar = {
        let mut v = [[[0u8; 4]; 3]; 2];
        v[0][0][0] = 1;
        Vec::from(v)
    };
    assert_eq!(result.to_vec3::<u8>()?, result_scalar);
    let scalar =
        candle_notebook::scalar_to_tensor(v1[0][0][0], (2, 3, 4), t1.dtype(), t1.device())?;
    let result_scalar = t1.eq(&scalar)?;
    assert_eq!(result_scalar.to_vec3::<u8>()?, result.to_vec3::<u8>()?);
    Ok(())
}

#[test]
fn ne() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.ne(&t2)?;
    let result_v = candle_notebook::ne_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn le() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.le(&t2)?;
    let result_v = candle_notebook::le_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn lt() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.lt(&t2)?;
    let result_v = candle_notebook::lt_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);

    Ok(())
}

#[test]
fn ge() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.ge(&t2)?;
    let result_v = candle_notebook::ge_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);
    Ok(())
}

#[test]
fn gt() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let result = t1.gt(&t2)?;
    let result_v = candle_notebook::gt_vec3(&v1, &v2);
    assert_eq!(result.to_vec3::<u8>()?, result_v);
    Ok(())
}
