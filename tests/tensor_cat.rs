use candle_core::{Device, Result, Tensor, D};

/// 使用 Tensor::cat 建立張量。
/// 操作時，每個張量的維度必須相同，
/// 且只能指定的維度可以允許不同大小，其餘的維度大小必須相同。
/// 結果的張量維度與輸入張量相同。

/// Tensor::cat 多個向量。
#[test]
fn cat_vector() -> Result<()> {
    let t1 = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
    let t2 = Tensor::new(&[4u32, 5, 6], &Device::Cpu)?;
    let t3 = Tensor::new(&[7u32, 8, 9], &Device::Cpu)?;

    let t = Tensor::cat(&[&t1, &t2, &t3], 0)?;
    assert_eq!(t.dims(), vec![3 + 3 + 3]);
    assert_eq!(t.to_vec1::<u32>()?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let t = Tensor::cat(&[&t1, &t2, &t3], 1); // DimOutOfRange
    assert!(t.is_err());
    Ok(())
}

/// Tensor::cat 多個張量。
#[test]
fn cat_n_rank_tensor() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;
    let t3 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v3 = t3.to_vec3::<f32>()?;

    let t = Tensor::cat(&[&t1, &t2, &t3], D::Minus1)?;
    assert_eq!(t.dims(), vec![2, 3, 4 + 4 + 4]);
    let mut v = vec![];
    for i in 0..2 {
        let mut cat1 = vec![];
        for j in 0..3 {
            let cat2 = [
                v1[i][j].as_slice(),
                v2[i][j].as_slice(),
                v3[i][j].as_slice(),
            ]
            .concat();
            cat1.push(cat2);
        }
        v.push(cat1);
    }
    assert_eq!(t.to_vec3::<f32>()?, v);

    let t = Tensor::cat(&[&t1, &t2, &t3], 1)?;
    assert_eq!(t.dims(), vec![2, 3 + 3 + 3, 4]);
    let mut v = vec![];
    for i in 0..2 {
        v.push([v1[i].as_slice(), v2[i].as_slice(), v3[i].as_slice()].concat());
    }
    assert_eq!(t.to_vec3::<f32>()?, v);

    let t = Tensor::cat(&[&t1, &t2, &t3], 0)?;
    assert_eq!(t.dims(), vec![2 + 2 + 2, 3, 4]);
    let v = [v1.as_slice(), v2.as_slice(), v3.as_slice()].concat();
    assert_eq!(t.to_vec3::<f32>()?, v);

    Ok(())
}

/// cat 操作時，只能指定的維度可以允許不同大小，其餘的維度大小必須相同。
#[test]
fn cat_rank() -> Result<()> {
    // 不同長度的向量
    let t1 = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
    let t2 = Tensor::new(&[4u32, 5], &Device::Cpu)?;
    let t = Tensor::cat(&[&t1, &t2], 0)?;
    assert_eq!(t.dims(), vec![3 + 2]);
    assert_eq!(t.to_vec1::<u32>()?, vec![1, 2, 3, 4, 5]);

    // 最後一個維度不同
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 1), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t = Tensor::cat(&[&t1, &t2], D::Minus1)?;
    assert_eq!(t.dims(), vec![2, 3, 1 + 4]);
    let t = Tensor::cat(&[&t1, &t2], 1); // ShapeMismatchCat
    assert!(t.is_err());

    // 第二個維度不同
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 2, 4), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t = Tensor::cat(&[&t1, &t2], D::Minus1); // ShapeMismatchCat
    assert!(t.is_err());
    let t = Tensor::cat(&[&t1, &t2], 1)?;
    assert_eq!(t.dims(), vec![2, 2 + 3, 4]);
    let t = Tensor::cat(&[&t1, &t2], 0); // ShapeMismatchCat
    assert!(t.is_err());

    // 只有最後一個維度大小相同
    let t1 = Tensor::rand(-1.0f32, 1.0, (1, 2, 4), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t = Tensor::cat(&[&t1, &t2], D::Minus1);
    assert!(t.is_err());
    let t = Tensor::cat(&[&t1, &t2], 1);
    assert!(t.is_err());
    let t = Tensor::cat(&[&t1, &t2], 0);
    assert!(t.is_err());

    // 第一個維度大小不同
    let t1 = Tensor::rand(-1.0f32, 1.0, (1, 3, 4), &Device::Cpu)?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let t = Tensor::cat(&[&t1, &t2], D::Minus1);
    assert!(t.is_err());
    let t = Tensor::cat(&[&t1, &t2], 1);
    assert!(t.is_err());
    let t = Tensor::cat(&[&t1, &t2], 0)?;
    assert_eq!(t.dims(), vec![1 + 2, 3, 4]);

    Ok(())
}
