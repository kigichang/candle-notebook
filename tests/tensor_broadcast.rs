use candle_core::{DType, Device, Result, Shape, Tensor};

/// 檢查兩個張量維度是否相容，並回傳相容的維度。
#[test]
fn shape_compatible() -> Result<()> {
    // scalar is compatible with any shape
    let s1: Shape = ().into(); // scalar demension
    let s2: Shape = (2, 3, 4).into();
    assert_eq!(
        s1.broadcast_shape_binary_op(&s2, "test")?.dims3()?,
        (2, 3, 4)
    );

    // vector deminsion is compatible with any shape with last dimension equal to vector length
    let s1: Shape = 4.into(); // vector demension
    let s2: Shape = (2, 3, 4).into();
    assert_eq!(
        s1.broadcast_shape_binary_op(&s2, "test")?.dims3()?,
        (2, 3, 4)
    );
    let s2: Shape = (3, 3).into();
    assert!(s1.broadcast_shape_binary_op(&s2, "test").is_err());

    // 2D demension with first dimension equal to 1 is like vector demension.
    let s1: Shape = (1, 4).into(); // 1xN demension
    let s2: Shape = (2, 3, 4).into();
    assert_eq!(
        s1.broadcast_shape_binary_op(&s2, "test")?.dims3()?,
        (2, 3, 4)
    );
    let s2: Shape = (3, 3).into();
    assert!(s1.broadcast_shape_binary_op(&s2, "test").is_err());

    // 2D demenion is compatible with any shape with last two dimensions equal to the 2D shape.
    let s1: Shape = (3, 4).into(); // MxN demension
    let s2: Shape = (2, 3, 4).into();
    assert_eq!(
        s1.broadcast_shape_binary_op(&s2, "test")?.dims3()?,
        (2, 3, 4)
    );
    let s2: Shape = (3, 3).into();
    assert!(s1.broadcast_shape_binary_op(&s2, "test").is_err());
    let s2: Shape = (2, 3, 3).into();
    assert!(s1.broadcast_shape_binary_op(&s2, "test").is_err());
    let s2: Shape = (2, 2, 4).into();
    assert!(s1.broadcast_shape_binary_op(&s2, "test").is_err());

    Ok(())
}

/// 由既有的張量，產生相容新維度的張量。
#[test]
fn broadcast_as() -> Result<()> {
    // 純量
    let t = Tensor::new(2.0f32, &Device::Cpu)?;
    // 純量轉向量
    let z = t.broadcast_as(4)?;
    assert_eq!(z.to_vec1::<f32>()?, vec![2.0, 2.0, 2.0, 2.0]);
    // 純量轉 2D 張量
    let z = t.broadcast_as((3, 4))?;
    assert_eq!(
        z.to_vec2::<f32>()?,
        Tensor::full(2.0f32, (3, 4), &Device::Cpu)?.to_vec2::<f32>()?
    );
    // 純量轉 3D 張量
    let z = t.broadcast_as((2, 3, 4))?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        Tensor::full(2.0f32, (2, 3, 4), &Device::Cpu)?.to_vec3::<f32>()?
    );

    // 向量
    let t = Tensor::new(&[1.0f32, 2.], &Device::Cpu)?;

    // vector to matrix
    let z = t.broadcast_as((2, 2))?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);

    // vector to 3D tensor
    let z = t.broadcast_as((2, 2, 2))?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![1.0, 2.0]],
            vec![vec![1.0, 2.0], vec![1.0, 2.0]]
        ]
    );

    // one-row matrix (1x2 matrix)
    let t = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?;

    let z = t.broadcast_as((2, 2))?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![1.0, 2.0]]);
    let z = t.broadcast_as((2, 2, 2))?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![1.0, 2.0]],
            vec![vec![1.0, 2.0], vec![1.0, 2.0]]
        ]
    );

    // 2x2 matrix
    let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
    let z = t.broadcast_as((4, 2));
    assert!(z.is_err());

    let z = t.broadcast_as((2, 2, 2))?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        ]
    );

    Ok(())
}

/// 進行不同維度的張量加法。
#[test]
fn broadcast_add() -> Result<()> {
    let a = Tensor::arange(0u32, 24, &Device::Cpu)?
        .reshape((2, 3, 4))?
        .to_dtype(DType::F32)?;

    // broadcast_add a scalar
    let b = Tensor::new(2.0f32, &Device::Cpu)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        Tensor::from_iter((0u32..24).map(|v| v + 2), &Device::Cpu)?
            .reshape((2, 3, 4))?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
    );

    // 交換律
    let c1 = b.broadcast_add(&a);
    assert_eq!(c.to_vec3::<f32>()?, c1?.to_vec3::<f32>()?);

    // broadcast_add a vector
    let b = Tensor::arange(0u32, 4, &Device::Cpu)?.to_dtype(DType::F32)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        &[
            [
                [0.0, 2.0, 4.0, 6.0],
                [4.0, 6.0, 8.0, 10.0],
                [8.0, 10.0, 12.0, 14.0]
            ],
            [
                [12.0, 14.0, 16.0, 18.0],
                [16.0, 18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0, 26.0]
            ]
        ]
    );
    // broadcast_add a one-row matrix.
    let b = Tensor::arange(0u32, 4, &Device::Cpu)?
        .reshape((1, 4))?
        .to_dtype(DType::F32)?;

    let c1 = a.broadcast_add(&b);
    assert_eq!(c.to_vec3::<f32>()?, c1?.to_vec3::<f32>()?);

    // broadcast_add a vector with wrong shape.
    let b = Tensor::arange(0u32, 2, &Device::Cpu)?.to_dtype(DType::F32)?;
    let c = a.broadcast_add(&b);
    assert!(c.is_err());

    // broadcast_add a 2x2 matrix with correct tail shape.
    let b = Tensor::arange(0u32, 12, &Device::Cpu)?
        .reshape((3, 4))?
        .to_dtype(DType::F32)?;

    let c = a.broadcast_add(&b)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        &[
            [
                [0.0, 2.0, 4.0, 6.0],
                [8.0, 10.0, 12.0, 14.0],
                [16.0, 18.0, 20.0, 22.0]
            ],
            [
                [12.0, 14.0, 16.0, 18.0],
                [20.0, 22.0, 24.0, 26.0],
                [28.0, 30.0, 32.0, 34.0]
            ]
        ]
    );

    // broadcast_add with wrong tail shape.
    let b = Tensor::arange(0u32, 8, &Device::Cpu)?
        .reshape((2, 4))?
        .to_dtype(DType::F32)?;
    let c = a.broadcast_add(&b);
    assert!(c.is_err());

    // broadcast_add with wrong shape.
    let b = Tensor::arange(0u32, 12, &Device::Cpu)?
        .reshape((4, 3))?
        .to_dtype(DType::F32)?;
    let c = a.broadcast_add(&b);
    assert!(c.is_err());

    Ok(())
}

/// 進行不同維度的張量減法。
#[test]
fn broadcast_sub() -> Result<()> {
    let a = Tensor::new(1.0f32, &Device::Cpu)?;
    let b = Tensor::arange(0u32, 24, &Device::Cpu)?
        .reshape((2, 3, 4))?
        .to_dtype(DType::F32)?;

    let c = a.broadcast_sub(&b)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        Tensor::from_iter((0u32..24).map(|v| 1.0 - v as f32), &Device::Cpu)?
            .reshape((2, 3, 4))?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
    );

    let a = Tensor::ones(4, DType::F32, &Device::Cpu)?;
    let c = b.broadcast_sub(&a)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        &[
            [
                [-1.0, 0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0]
            ],
            [
                [11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0]
            ]
        ]
    );

    let a = Tensor::ones((1, 4), DType::F32, &Device::Cpu)?;
    let c1 = b.broadcast_sub(&a);
    assert_eq!(c.to_vec3::<f32>()?, c1?.to_vec3::<f32>()?);

    let a = Tensor::ones((3, 4), DType::F32, &Device::Cpu)?;
    let c = b.broadcast_sub(&a)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        &[
            [
                [-1.0, 0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0]
            ],
            [
                [11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0]
            ]
        ]
    );

    Ok(())
}
