use candle_core::{Device, Result, Tensor, D};

#[test]
fn sum() -> Result<()> {
    let t = Tensor::arange(0u32, 24, &Device::Cpu)?;
    assert_eq!(t.rank(), 1);
    let sum = t.sum(D::Minus1)?;
    assert_eq!(sum.rank(), 0);
    assert_eq!(sum.to_scalar::<u32>()?, (0u32..24).sum::<u32>());

    let v = &[
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ];
    // ixjxk = 2x3x4
    let t = Tensor::new(v, &Device::Cpu)?;

    let sum0 = t.sum(0)?;
    assert_eq!(sum0.dims(), vec![3, 4]);
    // S(j, k) = sum(i) A(i, j, k)
    let mut v0 = vec![];
    for j in 0..3 {
        for k in 0..4 {
            let mut sum = 0u32;
            for i in 0..2 {
                sum += v[i][j][k];
            }
            v0.push(sum);
        }
    }
    let v0 = Tensor::from_vec(v0, (3, 4), &Device::Cpu)?;
    assert_eq!(sum0.to_vec2::<u32>()?, v0.to_vec2::<u32>()?);
    let sum0_keepdim = t.sum_keepdim(0)?;
    assert_eq!(sum0_keepdim.dims(), vec![1, 3, 4]);
    let v0 = v0.reshape((1, 3, 4))?;
    assert_eq!(sum0_keepdim.to_vec3::<u32>()?, v0.to_vec3::<u32>()?);

    let sum1 = t.sum(1)?;
    assert_eq!(sum1.dims(), vec![2, 4]);
    // S(i, k) = sum(j) A(i, j, k)
    let mut v1 = vec![];
    for i in 0..2 {
        for k in 0..4 {
            let mut sum = 0u32;
            for j in 0..3 {
                sum += v[i][j][k];
            }
            v1.push(sum);
        }
    }
    let v1 = Tensor::from_vec(v1, (2, 4), &Device::Cpu)?;
    assert_eq!(sum1.to_vec2::<u32>()?, v1.to_vec2::<u32>()?);
    let sum1_keepdim = t.sum_keepdim(1)?;
    assert_eq!(sum1_keepdim.dims(), vec![2, 1, 4]);
    let v1 = v1.reshape((2, 1, 4))?;
    assert_eq!(sum1_keepdim.to_vec3::<u32>()?, v1.to_vec3::<u32>()?);

    let sum2 = t.sum(2)?;
    assert_eq!(sum2.dims(), vec![2, 3]);
    // S(i, j) = sum(k) A(i, j, k)
    let mut v2 = vec![];
    for i in 0..2 {
        for j in 0..3 {
            let mut sum = 0u32;
            for k in 0..4 {
                sum += v[i][j][k];
            }
            v2.push(sum);
        }
    }
    let v2 = Tensor::from_vec(v2, (2, 3), &Device::Cpu)?;
    assert_eq!(sum2.to_vec2::<u32>()?, v2.to_vec2::<u32>()?);
    let sum2_keepdim = t.sum_keepdim(2)?;
    assert_eq!(sum2_keepdim.dims(), vec![2, 3, 1]);
    let v2 = v2.reshape((2, 3, 1))?;
    assert_eq!(sum2_keepdim.to_vec3::<u32>()?, v2.to_vec3::<u32>()?);

    let sum_all = t.sum_all()?;
    assert_eq!(sum_all.rank(), 0);
    let sum_all_v = v
        .map(|v1| v1.map(|v2| v2.iter().sum::<u32>()).iter().sum::<u32>())
        .iter()
        .sum::<u32>();

    assert_eq!(sum_all.to_scalar::<u32>()?, sum_all_v);
    Ok(())
}
