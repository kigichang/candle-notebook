use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::stack 建立張量。
/// 操作時，每個張量的維度必須相同，且每個維度的大小必須相同。
/// 由源碼來看，Tensor::stack 等同每個張量先做 t.unsqueeze，然後再做 Tensor::cat
/// 結果的張量維度比輸入張量多一維。

#[test]
fn stack_vector() -> Result<()> {
    let t1 = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
    let v1 = t1.to_vec1::<u32>()?;
    let t2 = Tensor::new(&[4u32, 5, 6], &Device::Cpu)?;
    let v2 = t2.to_vec1::<u32>()?;
    let t3 = Tensor::new(&[7u32, 8, 9], &Device::Cpu)?;
    let v3 = t3.to_vec1::<u32>()?;

    let t = Tensor::stack(&[&t1, &t2, &t3], 0)?;
    assert_eq!(t.dims(), vec![1 + 1 + 1, 3]);
    let v = vec![v1.clone(), v2.clone(), v3.clone()];
    assert_eq!(t.to_vec2::<u32>()?, v);
    let v = Tensor::cat(
        &[
            &t1.unsqueeze(0)?, // 1x3
            &t2.unsqueeze(0)?, // 1x3
            &t3.unsqueeze(0)?, // 1x3
        ],
        0,
    )?;
    assert_eq!(t.to_vec2::<u32>()?, v.to_vec2::<u32>()?);

    let t = Tensor::stack(&[&t1, &t2, &t3], 1)?;
    assert_eq!(t.dims(), vec![3, 1 + 1 + 1]);
    let mut v = vec![];
    for i in 0..3 {
        v.push(vec![v1[i], v2[i], v3[i]]);
    }
    assert_eq!(t.to_vec2::<u32>()?, v);
    let v = Tensor::cat(
        &[
            &t1.unsqueeze(1)?, // 3x1
            &t2.unsqueeze(1)?, // 3x1
            &t3.unsqueeze(1)?, // 3x1
        ],
        1,
    )?;
    assert_eq!(t.to_vec2::<u32>()?, v.to_vec2::<u32>()?);

    let t = Tensor::stack(&[&t1, &t2, &t3], 2);
    assert_eq!(t.is_err(), true);

    Ok(())
}

#[test]
fn stack_n_rank_tensor() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3), &Device::Cpu)?;
    let v1 = t1.to_vec2::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3), &Device::Cpu)?;
    let v2 = t2.to_vec2::<f32>()?;
    let t3 = Tensor::rand(-1.0f32, 1.0, (2, 3), &Device::Cpu)?;
    let v3 = t3.to_vec2::<f32>()?;

    let t = Tensor::stack(&[&t1, &t2, &t3], 0)?;
    assert_eq!(t.dims(), vec![1 + 1 + 1, 2, 3]);
    let v = vec![v1.clone(), v2.clone(), v3.clone()];
    assert_eq!(t.to_vec3::<f32>()?, v);
    let v = Tensor::cat(
        &[
            &t1.unsqueeze(0)?, // 1x2x3
            &t2.unsqueeze(0)?, // 1x2x3
            &t3.unsqueeze(0)?, // 1x2x3
        ],
        0,
    )?;
    assert_eq!(t.to_vec3::<f32>()?, v.to_vec3::<f32>()?);

    let t = Tensor::stack(&[&t1, &t2, &t3], 1)?;
    assert_eq!(t.dims(), vec![2, 1 + 1 + 1, 3]);
    let mut v = vec![];
    for i in 0..2 {
        v.push(vec![v1[i].clone(), v2[i].clone(), v3[i].clone()]);
    }
    assert_eq!(t.to_vec3::<f32>()?, v);
    let v = Tensor::cat(
        &[
            &t1.unsqueeze(1)?, // 2x1x3
            &t2.unsqueeze(1)?, // 2x1x3
            &t3.unsqueeze(1)?, // 2x1x3
        ],
        1,
    )?;
    assert_eq!(t.to_vec3::<f32>()?, v.to_vec3::<f32>()?);

    let t = Tensor::stack(&[&t1, &t2, &t3], 2)?;
    let mut v = vec![];
    for i in 0..2 {
        let mut s1 = vec![];
        for j in 0..3 {
            s1.push(vec![v1[i][j], v2[i][j], v3[i][j]]);
        }
        v.push(s1);
    }
    assert_eq!(t.to_vec3::<f32>()?, v);
    let v = Tensor::cat(
        &[
            &t1.unsqueeze(2)?, // 2x3x1
            &t2.unsqueeze(2)?, // 2x3x1
            &t3.unsqueeze(2)?, // 2x3x1
        ],
        2,
    )?;
    assert_eq!(t.to_vec3::<f32>()?, v.to_vec3::<f32>()?);

    Ok(())
}
