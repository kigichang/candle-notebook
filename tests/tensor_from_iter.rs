use candle_core::{Device, Result, Tensor};

#[test]
fn from_iter() -> Result<()> {
    // vector only
    let t = Tensor::from_iter(0..4u32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    let t = Tensor::from_iter(vec![0u32, 1, 2, 3], &Device::Cpu);
    assert_eq!(t?.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    let t = Tensor::from_iter([0u32, 1, 2, 3], &Device::Cpu)?;
    assert_eq!(t.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    Ok(())
}
