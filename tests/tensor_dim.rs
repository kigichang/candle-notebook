use candle_core::{DType, Device, Result, Tensor, D};

#[test]
fn rank() -> Result<()> {
    // scalar
    let t = Tensor::ones((), DType::F32, &Device::Cpu)?;
    assert_eq!(t.rank(), 0);

    // vector
    let t = Tensor::ones(2, DType::F32, &Device::Cpu)?;
    assert_eq!(t.rank(), 1);

    // 3x5 矩陣
    let t = Tensor::ones((3, 5), DType::F32, &Device::Cpu)?;
    assert_eq!(t.rank(), 2);

    // 2x3x4 矩陣
    let t = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
    assert_eq!(t.rank(), 3);

    Ok(())
}

#[test]
fn dims() -> Result<()> {
    // scalar
    let t = Tensor::ones((), DType::F32, &Device::Cpu)?;
    assert!(t.dim(0).is_err());

    assert_eq!(t.dims(), &[0usize; 0]);
    assert_eq!(t.dims0()?, ());
    assert!(t.dims1().is_err());
    assert!(t.dims2().is_err());
    assert!(t.dims3().is_err());
    assert!(t.dims4().is_err());
    assert!(t.dims5().is_err());

    // vector
    let t = Tensor::ones(2, DType::F32, &Device::Cpu)?;
    assert_eq!(t.dim(0)?, 2);
    assert!(t.dim(1).is_err());

    assert_eq!(t.dims(), &[2]);
    assert!(t.dims0().is_err());
    assert_eq!(t.dims1()?, 2);
    assert!(t.dims2().is_err());
    assert!(t.dims3().is_err());
    assert!(t.dims4().is_err());
    assert!(t.dims5().is_err());

    // 3x5 矩陣
    let t = Tensor::ones((3, 5), DType::F32, &Device::Cpu)?;
    assert_eq!(t.dim(0)?, 3);
    assert_eq!(t.dim(D::Minus2)?, t.dim(0)?);
    assert_eq!(t.dim(1)?, 5);
    assert_eq!(t.dim(D::Minus1)?, t.dim(1)?);

    assert!(t.dim(2).is_err());

    assert_eq!(t.dims(), &[3, 5]);
    assert!(t.dims0().is_err());
    assert!(t.dims1().is_err());
    assert_eq!(t.dims2()?, (3, 5));
    assert!(t.dims3().is_err());
    assert!(t.dims4().is_err());
    assert!(t.dims5().is_err());

    // 2x3x4 矩陣
    let t = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
    assert_eq!(t.dim(0)?, 2);
    assert_eq!(t.dim(D::Minus(t.rank()))?, t.dim(0)?);
    assert_eq!(t.dim(1)?, 3);
    assert_eq!(t.dim(D::Minus2)?, t.dim(1)?);
    assert_eq!(t.dim(2)?, 4);
    assert_eq!(t.dim(D::Minus1)?, t.dim(2)?);

    assert_eq!(t.dims(), &[2, 3, 4]);
    assert!(t.dims0().is_err());
    assert!(t.dims1().is_err());
    assert!(t.dims2().is_err());
    assert_eq!(t.dims3()?, (2, 3, 4));
    assert!(t.dims4().is_err());
    assert!(t.dims5().is_err());

    Ok(())
}

#[test]
fn reshape() -> Result<()> {
    // vector with 24 elements
    let t = Tensor::arange(0u32, 24u32, &Device::Cpu)?;

    // convert to 2x3x4 矩陣
    let t = t.reshape((2, 3, 4))?;

    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.dims3()?, (2, 3, 4));
    assert_eq!(
        t.to_vec3::<u32>()?,
        vec![
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
            vec![
                vec![12, 13, 14, 15],
                vec![16, 17, 18, 19],
                vec![20, 21, 22, 23]
            ]
        ]
    );

    Ok(())
}
