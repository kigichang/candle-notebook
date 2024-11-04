use candle_core::{Device, Result, Tensor};

/// broadcast_maximum 和 broadcast_minimum 都是先用 broadcast_as 將右運算元轉成左運算元的形狀後，
/// 再使用 maximum 和 minimum 進行比較。

#[test]
fn broadcast_max() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;

    let max = t1.broadcast_maximum(&t2)?;
    assert_eq!(max.dims(), vec![2, 3, 4]);

    let t2_as_t1 = t2.broadcast_as(t1.shape())?;
    let max_as = t1.maximum(&t2_as_t1)?;
    assert_eq!(max.to_vec3::<f32>()?, max_as.to_vec3::<f32>()?);

    Ok(())
}

#[test]
fn broadcast_min() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (3, 4), &Device::Cpu)?;

    let min = t1.broadcast_minimum(&t2)?;
    assert_eq!(min.dims(), vec![2, 3, 4]);

    let t2_as_t1 = t2.broadcast_as(t1.shape())?;
    let min_as = t1.minimum(&t2_as_t1)?;
    assert_eq!(min.to_vec3::<f32>()?, min_as.to_vec3::<f32>()?);

    Ok(())
}
