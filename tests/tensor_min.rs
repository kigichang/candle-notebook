use core::f32;

use candle_core::{Device, Result, Tensor};

#[test]
fn min() -> Result<()> {
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

    let min0 = t.min(0)?;
    assert_eq!(min0.dims(), vec![3, 4]);
    let mut v0 = vec![];
    for j in 0..3 {
        for k in 0..4 {
            let mut min = f32::MAX;
            for i in 0..2 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                }
            }
            v0.push(min);
        }
    }
    let v0 = Tensor::from_vec(v0, (3, 4), &Device::Cpu)?;
    assert_eq!(min0.to_vec2::<f32>()?, v0.to_vec2::<f32>()?);
    let min0_keepdim = t.min_keepdim(0)?;
    assert_eq!(min0_keepdim.dims(), vec![1, 3, 4]);
    let v0 = v0.unsqueeze(0)?; // (3, 4) -> (1, 3, 4)
    assert_eq!(min0_keepdim.to_vec3::<f32>()?, v0.to_vec3::<f32>()?);

    let min1 = t.min(1)?;
    assert_eq!(min1.dims(), vec![2, 4]);
    let mut v1 = vec![];
    for i in 0..2 {
        for k in 0..4 {
            let mut min = f32::MAX;
            for j in 0..3 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                }
            }
            v1.push(min);
        }
    }
    let v1 = Tensor::from_vec(v1, (2, 4), &Device::Cpu)?;
    assert_eq!(min1.to_vec2::<f32>()?, v1.to_vec2::<f32>()?);
    let min1_keepdim = t.min_keepdim(1)?;
    assert_eq!(min1_keepdim.dims(), vec![2, 1, 4]);
    let v1 = v1.unsqueeze(1)?; // (2, 4) -> (2, 1, 4)
    assert_eq!(min1_keepdim.to_vec3::<f32>()?, v1.to_vec3::<f32>()?);

    let min2 = t.min(2)?;
    assert_eq!(min2.dims(), vec![2, 3]);
    let mut v2 = vec![];
    for i in 0..2 {
        for j in 0..3 {
            let mut min = f32::MAX;
            for k in 0..4 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                }
            }
            v2.push(min);
        }
    }
    let v2 = Tensor::from_vec(v2, (2, 3), &Device::Cpu)?;
    assert_eq!(min2.to_vec2::<f32>()?, v2.to_vec2::<f32>()?);
    let min2_keepdim = t.min_keepdim(2)?;
    assert_eq!(min2_keepdim.dims(), vec![2, 3, 1]);
    let v2 = v2.unsqueeze(2)?; // (2, 3) -> (2, 3, 1)
    assert_eq!(min2_keepdim.to_vec3::<f32>()?, v2.to_vec3::<f32>()?);

    let min_all = t.min_all()?;
    assert_eq!(min_all.to_scalar::<f32>()?, 0.0f32);
    Ok(())
}
