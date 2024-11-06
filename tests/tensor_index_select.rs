use candle_core::{DType, Device, Result, Tensor};

//// Tensor::index_select 在指定的維度上選取元素，
///  indexes 必須是整數向量，表示要選取的元素索引。
#[test]
fn index_select() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, (10, 20, 30), &Device::Cpu)?;
    let dim = t.dims();
    let v = t.to_vec3::<f32>()?;

    // 第一維取值
    let idx = Tensor::rand(0f32, dim[0] as f32, dim[0] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.index_select(&idx, 0)?;
    println!("{:?}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dim[0] / 2, dim[1], dim[2]]);
    let mut choose_v = vec![];
    for i in idx.to_vec1::<u32>()? {
        choose_v.push(v[i as usize].clone());
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // 第二維取值
    let idx = Tensor::rand(0f32, dim[1] as f32, dim[1] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.index_select(&idx, 1)?;
    println!("{:?}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dim[0], dim[1] / 2, dim[2]]);
    let mut choose_v = vec![];
    for i in 0..dim[0] {
        let mut v_j = vec![];
        for j in idx.to_vec1::<u32>()? {
            v_j.push(v[i][j as usize].clone());
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // 第三維取值
    let idx = Tensor::rand(0f32, dim[2] as f32, dim[2] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.index_select(&idx, 2)?;
    println!("{:?}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dim[0], dim[1], dim[2] / 2]);
    let mut choose_v = vec![];
    for i in 0..dim[0] {
        let mut v_j = vec![];
        for j in 0..dim[1] {
            let mut v_k = vec![];
            for k in idx.to_vec1::<u32>()? {
                v_k.push(v[i][j][k as usize]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    Ok(())
}
