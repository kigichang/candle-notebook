use candle_core::{Device, Result, Tensor};

#[test]
fn mean() -> Result<()> {
    let v = &[
        [[0f32, 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
        [
            [12., 13., 14., 15.],
            [16., 17., 18., 19.],
            [20., 21., 22., 23.],
        ],
    ];
    // ixjxk = 2x3x4
    let t = Tensor::new(v, &Device::Cpu)?;

    let mean0 = t.mean(0)?;
    assert_eq!(mean0.dims(), vec![3, 4]);
    let mut v0 = vec![];
    for j in 0..3 {
        for k in 0..4 {
            let mut sum = 0f32;
            for i in 0..2 {
                sum += v[i][j][k];
            }
            v0.push(sum / 2.0);
        }
    }
    let v0 = Tensor::from_vec(v0, (3, 4), &Device::Cpu)?;
    assert_eq!(mean0.to_vec2::<f32>()?, v0.to_vec2::<f32>()?);
    let mean0_keepdim = t.mean_keepdim(0)?;
    assert_eq!(mean0_keepdim.dims(), vec![1, 3, 4]);
    let v0 = v0.reshape((1, 3, 4))?;
    assert_eq!(mean0_keepdim.to_vec3::<f32>()?, v0.to_vec3::<f32>()?);

    let mean1 = t.mean(1)?;
    assert_eq!(mean1.dims(), vec![2, 4]);
    let mut v1 = vec![];
    for i in 0..2 {
        for k in 0..4 {
            let mut sum = 0f32;
            for j in 0..3 {
                sum += v[i][j][k];
            }
            v1.push(sum / 3.);
        }
    }
    let v1 = Tensor::from_vec(v1, (2, 4), &Device::Cpu)?;
    assert_eq!(mean1.to_vec2::<f32>()?, v1.to_vec2::<f32>()?);
    let mean1_keepdim = t.mean_keepdim(1)?;
    assert_eq!(mean1_keepdim.dims(), vec![2, 1, 4]);
    let v1 = v1.reshape((2, 1, 4))?;
    assert_eq!(mean1_keepdim.to_vec3::<f32>()?, v1.to_vec3::<f32>()?);

    let mean2 = t.mean(2)?;
    assert_eq!(mean2.dims(), vec![2, 3]);
    let mut v2 = vec![];
    for i in 0..2 {
        for j in 0..3 {
            let mut sum = 0f32;
            for k in 0..4 {
                sum += v[i][j][k];
            }
            v2.push(sum / 4.);
        }
    }
    let v2 = Tensor::from_vec(v2, (2, 3), &Device::Cpu)?;
    assert_eq!(mean2.to_vec2::<f32>()?, v2.to_vec2::<f32>()?);
    let mean2_keepdim = t.mean_keepdim(2)?;
    assert_eq!(mean2_keepdim.dims(), vec![2, 3, 1]);
    let v2 = v2.reshape((2, 3, 1))?;
    assert_eq!(mean2_keepdim.to_vec3::<f32>()?, v2.to_vec3::<f32>()?);

    let mean_all = t.mean_all()?;
    assert_eq!(mean_all.rank(), 0);
    let sum_all_v = v
        .map(|v1| v1.map(|v2| v2.iter().sum::<f32>()).iter().sum::<f32>())
        .iter()
        .sum::<f32>();
    assert_eq!(mean_all.to_scalar::<f32>()?, sum_all_v / 24.0);

    Ok(())
}
