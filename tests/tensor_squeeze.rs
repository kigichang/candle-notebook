use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::squeeze 降維。
/// 被消滅維度的大小必須是 1，
/// 如果不是 1 則回傳原張量。
#[test]
fn squeeze() -> Result<()> {
    let t = Tensor::new(&[[1.0f32, 2., 3.]], &Device::Cpu)?;
    assert_eq!(t.dims(), vec![1, 3]);
    let s = t.squeeze(0)?;
    assert_eq!(s.dims(), vec![3]);
    let s = t.squeeze(1)?;
    assert_eq!(t.to_vec2::<f32>()?, s.to_vec2::<f32>()?);

    let t = Tensor::rand(-1.0f32, 1.0, (3, 4, 5), &Device::Cpu)?;

    let max0 = t.max(0)?;
    let max0_keepdim = t.max_keepdim(0)?.squeeze(0)?; // [1, 4, 5] -> [4, 5]
    assert_eq!(max0.dims(), vec![4, 5]);
    assert_eq!(max0_keepdim.dims(), vec![4, 5]);

    let max1 = t.max(1)?;
    let max1_keepdim = t.max_keepdim(1)?.squeeze(1)?; // [3, 1, 5] -> [3, 5]
    assert_eq!(max1.dims(), vec![3, 5]);
    assert_eq!(max1_keepdim.dims(), vec![3, 5]);

    let max2 = t.max(2)?;
    let max2_keepdim = t.max_keepdim(2)?.squeeze(2)?; // [3, 4, 1] -> [3, 4]
    assert_eq!(max2.dims(), vec![3, 4]);
    assert_eq!(max2_keepdim.dims(), vec![3, 4]);

    Ok(())
}

/// 使用 Tensor::unsqueeze 升維。
/// 張量的維度會增加一維。
/// 指定升維的維度大小會是 1。
#[test]
fn unsqueeze() -> Result<()> {
    let t = Tensor::new(&[[1.0f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let v = t.to_vec2::<f32>()?;
    let mut dims = t.dims().to_vec();
    dims.insert(0, 1);
    let s0 = t.unsqueeze(0)?;
    assert_eq!(s0.dims(), dims); // 1x2x3
    let v0 = vec![v.clone()];
    assert_eq!(s0.to_vec3::<f32>()?, v0);

    let s1 = t.unsqueeze(1)?;
    let mut dims = t.dims().to_vec();
    dims.insert(1, 1);
    assert_eq!(s1.dims(), dims); // 2x1x3
    let mut v1 = vec![];
    for i in 0..2 {
        v1.push(vec![v[i].clone()]); // 拆解第二維的元素(向量)，變成 1x3
    }
    assert_eq!(s1.to_vec3::<f32>()?, v1);

    let s2 = t.unsqueeze(2)?;
    let mut dims = t.dims().to_vec();
    dims.insert(2, 1);
    assert_eq!(s2.dims(), dims); // 2x3x1
    let mut v2 = vec![];
    for i in 0..2 {
        let mut v_i = vec![];
        for j in 0..3 {
            v_i.push(vec![v[i][j]]); // 拆解向量內元素，變成 1x1
        }
        v2.push(v_i);
    }
    assert_eq!(s2.to_vec3::<f32>()?, v2);

    let max0_keepdim = t.max_keepdim(0)?;
    let max0 = t.max(0)?.unsqueeze(0)?;
    assert_eq!(max0.dims(), max0_keepdim.dims());

    let max1_keepdim = t.max_keepdim(1)?;
    let max1 = t.max(1)?.unsqueeze(1)?;
    assert_eq!(max1.dims(), max1_keepdim.dims());

    Ok(())
}
