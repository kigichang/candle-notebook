use candle_core::{Device, Result, Tensor, D};

#[test]
fn transpose() -> Result<()> {
    let t = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape((2, 3, 4))?;

    let tensor_values = t.to_vec3::<u32>()?;
    let t0 = t.transpose(0, 1)?; // (2,3,4) -> (3,2,4)
    assert_eq!(t0.dims(), vec![3, 2, 4]);

    let mut v0 = vec![];
    for j in 0..3 {
        let mut v1 = vec![];
        for i in 0..2 {
            v1.push(tensor_values[i][j].clone());
        }
        v0.push(v1);
    }
    assert_eq!(t0.to_vec3::<u32>()?, v0);

    let t1 = t.transpose(0, 2)?; // (2,3,4) -> (4,3,2)
    assert_eq!(t1.dims(), vec![4, 3, 2]);
    let mut v2 = vec![];
    for k in 0..4 {
        let mut v1 = vec![];
        for j in 0..3 {
            let mut v0 = vec![];
            for i in 0..2 {
                v0.push(tensor_values[i][j][k]);
            }
            v1.push(v0);
        }
        v2.push(v1);
    }
    assert_eq!(t1.to_vec3::<u32>()?, v2);

    let t2 = t.transpose(1, 2)?; // (2,3,4) -> (2,4,3)
    assert_eq!(t2.dims(), vec![2, 4, 3]);

    let mut v0 = vec![];
    for i in 0..2 {
        let mut v1 = vec![];
        for k in 0..4 {
            let mut v2 = vec![];
            for j in 0..3 {
                v2.push(tensor_values[i][j][k]);
            }
            v1.push(v2);
        }
        v0.push(v1);
    }
    assert_eq!(t2.to_vec3::<u32>()?, v0);

    Ok(())
}

#[test]
fn t() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, 4, &Device::Cpu)?;
    assert!(t.t().is_err());

    // 2D 張量轉置
    let t = Tensor::rand(-1.0f32, 1.0, (1, 4), &Device::Cpu)?;
    assert_eq!(t.t()?.dims(), &[4, 1]);

    assert_eq!(
        t.transpose(0, 1)?.to_vec2::<f32>()?,
        t.t()?.to_vec2::<f32>()?
    );

    let t = Tensor::rand(-1.0f32, 1.0f32, (2, 3, 4), &Device::Cpu)?;
    assert_eq!(t.t()?.dims(), &[2, 4, 3]);
    assert_eq!(t.t()?.dims(), t.transpose(D::Minus2, D::Minus1)?.dims());
    Ok(())
}
