use candle_core::{Device, Result, Tensor};

#[test]
fn argmax_vector() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, 4, &Device::Cpu)?;
    let v = t.to_vec1::<f32>()?;

    let argmax = t.argmax(0)?;
    // 找出最大值的索引
    let idx = v
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &x)| if x > v[acc] { i } else { acc });

    assert_eq!(argmax.to_vec0::<u32>()?, idx as u32);
    Ok(())
}

#[test]
fn argmax() -> Result<()> {
    let t = Tensor::rand(-0.1f32, 1.0f32, (3, 4, 5), &Device::Cpu)?;
    let v = t.to_vec3::<f32>()?;

    let argmax0 = t.argmax(0)?;
    let mut v0 = vec![];
    for j in 0..4 {
        for k in 0..5 {
            let mut max = v[0][j][k];
            let mut idx = 0;
            for i in 1..3 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                    idx = i;
                }
            }
            v0.push(idx as u32);
        }
    }
    let v0 = Tensor::from_vec(v0, (4, 5), &Device::Cpu)?;
    assert_eq!(argmax0.to_vec2::<u32>()?, v0.to_vec2::<u32>()?);
    let argmax0_keepdim = t.argmax_keepdim(0)?;
    assert_eq!(argmax0_keepdim.dims(), vec![1, 4, 5]);
    let v0 = v0.unsqueeze(0)?; // (4, 5) -> (1, 4, 5)
    assert_eq!(argmax0_keepdim.to_vec3::<u32>()?, v0.to_vec3::<u32>()?);

    let argmax1 = t.argmax(1)?;
    let mut v1 = vec![];
    for i in 0..3 {
        for k in 0..5 {
            let mut max = v[i][0][k];
            let mut idx = 0;
            for j in 1..4 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                    idx = j;
                }
            }
            v1.push(idx as u32);
        }
    }
    let v1 = Tensor::from_vec(v1, (3, 5), &Device::Cpu)?;
    assert_eq!(argmax1.to_vec2::<u32>()?, v1.to_vec2::<u32>()?);
    let argmax1_keepdim = t.argmax_keepdim(1)?;
    assert_eq!(argmax1_keepdim.dims(), vec![3, 1, 5]);
    let v1 = v1.unsqueeze(1)?; // (3, 5) -> (3, 1, 5)
    assert_eq!(argmax1_keepdim.to_vec3::<u32>()?, v1.to_vec3::<u32>()?);

    let argmax2 = t.argmax(2)?;
    let mut v2 = vec![];
    for i in 0..3 {
        for j in 0..4 {
            let mut max = v[i][j][0];
            let mut idx = 0;
            for k in 1..5 {
                if max < v[i][j][k] {
                    max = v[i][j][k];
                    idx = k;
                }
            }
            v2.push(idx as u32);
        }
    }
    let v2 = Tensor::from_vec(v2, (3, 4), &Device::Cpu)?;
    assert_eq!(argmax2.to_vec2::<u32>()?, v2.to_vec2::<u32>()?);
    let argmax2_keepdim = t.argmax_keepdim(2)?;
    assert_eq!(argmax2_keepdim.dims(), vec![3, 4, 1]);
    let v2 = v2.unsqueeze(2)?;
    assert_eq!(argmax2_keepdim.to_vec3::<u32>()?, v2.to_vec3::<u32>()?);

    Ok(())
}

#[test]
fn argmin_vector() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, 4, &Device::Cpu)?;
    let v = t.to_vec1::<f32>()?;

    let argmin = t.argmin(0)?;
    // 找出最小值的索引
    let idx = v
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &x)| if x < v[acc] { i } else { acc });

    assert_eq!(argmin.to_vec0::<u32>()?, idx as u32);
    Ok(())
}

#[test]
fn argmin() -> Result<()> {
    let t = Tensor::rand(-0.1f32, 1.0f32, (3, 4, 5), &Device::Cpu)?;
    let v = t.to_vec3::<f32>()?;

    let argmin0 = t.argmin(0)?;
    let mut v0 = vec![];
    for j in 0..4 {
        for k in 0..5 {
            let mut min = v[0][j][k];
            let mut idx = 0;
            for i in 1..3 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                    idx = i;
                }
            }
            v0.push(idx as u32);
        }
    }
    let v0 = Tensor::from_vec(v0, (4, 5), &Device::Cpu)?;
    assert_eq!(argmin0.to_vec2::<u32>()?, v0.to_vec2::<u32>()?);
    let argmin0_keepdim = t.argmin_keepdim(0)?;
    assert_eq!(argmin0_keepdim.dims(), vec![1, 4, 5]);
    let v0 = v0.unsqueeze(0)?; // (4, 5) -> (1, 4, 5)
    assert_eq!(argmin0_keepdim.to_vec3::<u32>()?, v0.to_vec3::<u32>()?);

    let argmin1 = t.argmin(1)?;
    let mut v1 = vec![];
    for i in 0..3 {
        for k in 0..5 {
            let mut min = v[i][0][k];
            let mut idx = 0;
            for j in 1..4 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                    idx = j;
                }
            }
            v1.push(idx as u32);
        }
    }
    let v1 = Tensor::from_vec(v1, (3, 5), &Device::Cpu)?;
    assert_eq!(argmin1.to_vec2::<u32>()?, v1.to_vec2::<u32>()?);
    let argmin1_keepdim = t.argmin_keepdim(1)?;
    assert_eq!(argmin1_keepdim.dims(), vec![3, 1, 5]);
    let v1 = v1.unsqueeze(1)?; // (3, 5) -> (3, 1, 5)
    assert_eq!(argmin1_keepdim.to_vec3::<u32>()?, v1.to_vec3::<u32>()?);

    let argmin2 = t.argmin(2)?;
    let mut v2 = vec![];
    for i in 0..3 {
        for j in 0..4 {
            let mut min = v[i][j][0];
            let mut idx = 0;
            for k in 1..5 {
                if min > v[i][j][k] {
                    min = v[i][j][k];
                    idx = k;
                }
            }
            v2.push(idx as u32);
        }
    }
    let v2 = Tensor::from_vec(v2, (3, 4), &Device::Cpu)?;
    assert_eq!(argmin2.to_vec2::<u32>()?, v2.to_vec2::<u32>()?);
    let argmin2_keepdim = t.argmin_keepdim(2)?;
    assert_eq!(argmin2_keepdim.dims(), vec![3, 4, 1]);
    let v2 = v2.unsqueeze(2)?;
    assert_eq!(argmin2_keepdim.to_vec3::<u32>()?, v2.to_vec3::<u32>()?);

    Ok(())
}
