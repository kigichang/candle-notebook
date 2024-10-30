use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::from_iter 建立向量
#[test]
fn from_iter() -> Result<()> {
    // from_iter 只能產生向量

    let t = Tensor::from_iter(0..4u32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    let t = Tensor::from_iter(vec![0u32, 1, 2, 3], &Device::Cpu);
    assert_eq!(t?.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    let t = Tensor::from_iter([0u32, 1, 2, 3], &Device::Cpu)?;
    assert_eq!(t.to_vec1::<u32>()?, vec![0, 1, 2, 3]);

    Ok(())
}
