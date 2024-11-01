use candle_core::{Device, Result, Tensor};

#[test]
fn squeeze() -> Result<()> {
    // The PyTorch semantics are to return the same tensor if the target dimension
    // does not have a size of 1.

    let t = Tensor::new(&[[1.0f32, 2., 3.]], &Device::Cpu)?;
    assert_eq!(t.dims(), vec![1, 3]);
    let s = t.squeeze(0)?;
    assert_eq!(s.dims(), vec![3]);
    let s = t.squeeze(1)?;
    assert_eq!(t.to_vec2::<f32>()?, s.to_vec2::<f32>()?);

    let t = Tensor::rand(-1.0f32, 1.0, (3, 4, 5), &Device::Cpu)?;

    let max0 = t.max(0)?;
    let max0_keepdim = t.max_keepdim(0)?.squeeze(0)?; // [1, 4, 5] -> [4, 5]
    assert_eq!(max0.dims(), vec![4, 5]);
    assert_eq!(max0_keepdim.dims(), vec![4, 5]);

    let max1 = t.max(1)?;
    let max1_keepdim = t.max_keepdim(1)?.squeeze(1)?; // [3, 1, 5] -> [3, 5]
    assert_eq!(max1.dims(), vec![3, 5]);
    assert_eq!(max1_keepdim.dims(), vec![3, 5]);

    let max2 = t.max(2)?;
    let max2_keepdim = t.max_keepdim(2)?.squeeze(2)?; // [3, 4, 1] -> [3, 4]
    assert_eq!(max2.dims(), vec![3, 4]);
    assert_eq!(max2_keepdim.dims(), vec![3, 4]);

    Ok(())
}

#[test]
fn unsqueeze() -> Result<()> {
    let t = Tensor::new(&[[1.0f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let mut dims = t.dims().to_vec();
    dims.insert(0, 1);
    let s0 = t.unsqueeze(0)?;
    assert_eq!(s0.dims(), dims);

    let s1 = t.unsqueeze(1)?;
    let mut dims = t.dims().to_vec();
    dims.insert(1, 1);
    assert_eq!(s1.dims(), dims);

    let max0_keepdim = t.max_keepdim(0)?;
    let max0 = t.max(0)?.unsqueeze(0)?;
    assert_eq!(max0.dims(), max0_keepdim.dims());

    let max1_keepdim = t.max_keepdim(1)?;
    let max1 = t.max(1)?.unsqueeze(1)?;
    assert_eq!(max1.dims(), max1_keepdim.dims());

    Ok(())
}
