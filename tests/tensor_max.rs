use candle_core::{Device, Result, Tensor};

#[test]
fn max() -> Result<()> {
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

    let max0 = t.max(0)?;
    assert_eq!(max0.dims(), vec![3, 4]);
    let mut v0 = vec![];
    for j in 0..3 {
        for k in 0..4 {
            let mut max = f32::MIN;
            for i in 0..2 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                }
            }
            v0.push(max);
        }
    }
    let v0 = Tensor::from_vec(v0, (3, 4), &Device::Cpu)?;
    assert_eq!(max0.to_vec2::<f32>()?, v0.to_vec2::<f32>()?);
    let max0_keepdim = t.max_keepdim(0)?;
    assert_eq!(max0_keepdim.dims(), vec![1, 3, 4]);
    let v0 = v0.unsqueeze(0)?; // (3, 4) -> (1, 3, 4)
    assert_eq!(max0_keepdim.to_vec3::<f32>()?, v0.to_vec3::<f32>()?);

    let max1 = t.max(1)?;
    assert_eq!(max1.dims(), vec![2, 4]);
    let mut v1 = vec![];
    for i in 0..2 {
        for k in 0..4 {
            let mut max = f32::MIN;
            for j in 0..3 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                }
            }
            v1.push(max);
        }
    }
    let v1 = Tensor::from_vec(v1, (2, 4), &Device::Cpu)?;
    assert_eq!(max1.to_vec2::<f32>()?, v1.to_vec2::<f32>()?);
    let max1_keepdim = t.max_keepdim(1)?;
    assert_eq!(max1_keepdim.dims(), vec![2, 1, 4]);
    let v1 = v1.unsqueeze(1)?; // (2, 4) -> (2, 1, 4)
    assert_eq!(max1_keepdim.to_vec3::<f32>()?, v1.to_vec3::<f32>()?);

    let max2 = t.max(2)?;
    assert_eq!(max2.dims(), vec![2, 3]);
    let mut v2 = vec![];
    for i in 0..2 {
        for j in 0..3 {
            let mut max = f32::MIN;
            for k in 0..4 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                }
            }
            v2.push(max);
        }
    }
    let v2 = Tensor::from_vec(v2, (2, 3), &Device::Cpu)?;
    assert_eq!(max2.to_vec2::<f32>()?, v2.to_vec2::<f32>()?);
    let max2_keepdim = t.max_keepdim(2)?;
    assert_eq!(max2_keepdim.dims(), vec![2, 3, 1]);
    let v2 = v2.unsqueeze(2)?; // (2, 3) -> (2, 3, 1)
    assert_eq!(max2_keepdim.to_vec3::<f32>()?, v2.to_vec3::<f32>()?);

    let max_all = t.max_all()?;
    assert_eq!(max_all.to_scalar::<f32>()?, 23.);
    Ok(())
}
