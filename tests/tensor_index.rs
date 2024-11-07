use candle_core::{DType, Device, IndexOp, Result, Tensor};

// TensorIndexer
// pub enum TensorIndexer {
//     /// This selects the elements for which an index has some specific value.
//     Select(usize),
//     /// This is a regular slice, purely indexing a chunk of the tensor
//     Narrow(Bound<usize>, Bound<usize>),
//     /// Indexing via a 1d tensor
//     IndexSelect(Tensor),
//     Err(Error),
// }

#[test]
fn indexer_select() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0f32, (10, 20, 30), &Device::Cpu)?;
    let dims = t.dims();
    let v = t.to_vec3::<f32>()?;

    // dim0
    let idx0 = rand::random::<usize>() % dims[0];
    let choose = t.i(idx0)?;
    println!("{}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dims[1], dims[2]]);
    assert_eq!(choose.to_vec2::<f32>()?, v[idx0]);

    let choose = t.i((idx0,))?; // same as above
    println!("{}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dims[1], dims[2]]);
    assert_eq!(choose.to_vec2::<f32>()?, v[idx0]);

    // same as narrow((0, idx0, 1) and then squeeze(0)
    let choose_narrow = t.narrow(0, idx0, 1)?.squeeze(0)?;
    assert_eq!(choose.to_vec2::<f32>()?, choose_narrow.to_vec2::<f32>()?);

    // (dim0, dim1)
    let idx1 = rand::random::<usize>() % dims[1];
    let choose = t.i((idx0, idx1))?;
    println!("{}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[dims[2]]);
    assert_eq!(choose.to_vec1::<f32>()?, v[idx0][idx1]);

    let choose_narrow = t
        .narrow(0, idx0, 1)?
        .squeeze(0)?
        .narrow(0, idx1, 1)?
        .squeeze(0)?;
    assert_eq!(choose.to_vec1::<f32>()?, choose_narrow.to_vec1::<f32>()?);

    // (dim0, dim1, dim2)
    let idx2 = rand::random::<usize>() % dims[2];
    let choose = t.i((idx0, idx1, idx2))?;
    println!("{}", choose.is_contiguous());
    assert_eq!(choose.shape(), &().into());
    assert_eq!(choose.to_scalar::<f32>()?, v[idx0][idx1][idx2]);

    let choose_narrow = t
        .narrow(0, idx0, 1)?
        .squeeze(0)?
        .narrow(0, idx1, 1)?
        .squeeze(0)?
        .narrow(0, idx2, 1)?
        .squeeze(0)?;
    assert_eq!(
        choose.to_scalar::<f32>()?,
        choose_narrow.to_scalar::<f32>()?
    );

    Ok(())
}

#[test]
fn indexer_narrow() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0f32, (10, 20, 30), &Device::Cpu)?;
    let dims = t.dims();
    let v = t.to_vec3::<f32>()?;

    // dim0
    let idx0 = rand::random::<usize>() % dims[0];
    let choose = t.i(0..=idx0)?;
    println!("{}", choose.is_contiguous());
    assert_eq!(choose.dims(), &[idx0 + 1, dims[1], dims[2]]);
    let mut choose_v = vec![];
    for i in 0..=idx0 {
        choose_v.push(v[i].clone());
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    let choose_narrow = t.narrow(0, 0, idx0 + 1)?;
    assert_eq!(choose.to_vec3::<f32>()?, choose_narrow.to_vec3::<f32>()?);

    // dim1
    let idx1 = rand::random::<usize>() % dims[1];
    let choose = t.i((0..=idx0, 0..=idx1))?;
    assert_eq!(choose.dims(), &[idx0 + 1, idx1 + 1, dims[2]]);
    let choose = if !choose.is_contiguous() {
        println!("not contiguous");
        choose.contiguous()?
    } else {
        choose
    };

    let mut choose_v = vec![];
    for i in 0..=idx0 {
        let mut v_j = vec![];
        for j in 0..=idx1 {
            v_j.push(v[i][j].clone());
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // get all in dim0, and choose in dim1
    let choose = t.i((.., 0..=idx1))?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0], idx1 + 1, dims[2]]);
    let mut choose_v = vec![];
    for i in 0..dims[0] {
        let mut v_j = vec![];
        for j in 0..=idx1 {
            v_j.push(v[i][j].clone());
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // dim2
    let idx2 = rand::random::<usize>() % dims[2];
    let choose = t.i((0..=idx0, 0..=idx1, 0..=idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[idx0 + 1, idx1 + 1, idx2 + 1]);
    let mut choose_v = vec![];
    for i in 0..=idx0 {
        let mut v_j = vec![];
        for j in 0..=idx1 {
            let mut v_k = vec![];
            for k in 0..=idx2 {
                v_k.push(v[i][j][k]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    let choose = t.i((.., 0..=idx1, 0..=idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0], idx1 + 1, idx2 + 1]);
    let mut choose_v = vec![];
    for i in 0..dims[0] {
        let mut v_j = vec![];
        for j in 0..=idx1 {
            let mut v_k = vec![];
            for k in 0..=idx2 {
                v_k.push(v[i][j][k]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    let choose = t.i((0..=idx0, .., 0..=idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[idx0 + 1, dims[1], idx2 + 1]);
    let mut choose_v = vec![];
    for i in 0..=idx0 {
        let mut v_j = vec![];
        for j in 0..dims[1] {
            let mut v_k = vec![];
            for k in 0..=idx2 {
                v_k.push(v[i][j][k]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    Ok(())
}

#[test]
fn indexer_index_select() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0f32, (10, 20, 30), &Device::Cpu)?;
    let dims = t.dims();
    let v = t.to_vec3::<f32>()?;

    // dim0
    let idx0 =
        Tensor::rand(0f32, dims[0] as f32, dims[0] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.i(&idx0)?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0] / 2, dims[1], dims[2]]);

    let mut choose_v = vec![];
    for i in idx0.to_vec1::<u32>()? {
        choose_v.push(v[i as usize].clone());
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // dim1
    let idx1 =
        Tensor::rand(0f32, dims[1] as f32, dims[1] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.i((&idx0, &idx1))?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0] / 2, dims[1] / 2, dims[2]]);
    let mut choose_v = vec![];
    for i in idx0.to_vec1::<u32>()? {
        let mut v_j = vec![];
        for j in idx1.to_vec1::<u32>()? {
            v_j.push(v[i as usize][j as usize].clone());
        }
        choose_v.push(v_j);
    }

    let choose = t.i((.., &idx1))?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0], dims[1] / 2, dims[2]]);
    let mut choose_v = vec![];
    for i in 0..dims[0] {
        let mut v_j = vec![];
        for j in idx1.to_vec1::<u32>()? {
            v_j.push(v[i][j as usize].clone());
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // dim2
    let idx2 =
        Tensor::rand(0f32, dims[2] as f32, dims[2] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let choose = t.i((&idx0, &idx1, &idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[dims[0] / 2, dims[1] / 2, dims[2] / 2]);
    let mut choose_v = vec![];
    for i in idx0.to_vec1::<u32>()? {
        let mut v_j = vec![];
        for j in idx1.to_vec1::<u32>()? {
            let mut v_k = vec![];
            for k in idx2.to_vec1::<u32>()? {
                v_k.push(v[i as usize][j as usize][k as usize]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    Ok(())
}

#[test]
fn index() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0f32, (10, 20, 30), &Device::Cpu)?;
    let dims = t.dims();
    let v = t.to_vec3::<f32>()?;

    let idx0 = rand::random::<usize>() % dims[0];
    let idx1 =
        Tensor::rand(0f32, dims[1] as f32, dims[1] / 2, &Device::Cpu)?.to_dtype(DType::U32)?;
    let idx2 = rand::random::<usize>() % dims[2];

    // 最後一維如果是 Select(usize) 則會被消除。
    // 因為 Select(usize) 最後回傳時，會做 squeeze(0)。
    let choose = t.i((0..=idx0, &idx1, idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[idx0 + 1, dims[1] / 2]);
    let mut choose_v = vec![];
    for i in 0..=idx0 {
        let mut v_j = vec![];
        for j in idx1.to_vec1::<u32>()? {
            v_j.push(v[i][j as usize][idx2]);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec2::<f32>()?, choose_v);

    // 如果最後一維不是 Select(usize)，則不會被消除。
    let idx2 = Tensor::new(vec![idx2 as u32], &Device::Cpu)?;
    let choose = t.i((0..=idx0, &idx1, &idx2))?.contiguous()?;
    assert_eq!(choose.dims(), &[idx0 + 1, dims[1] / 2, idx2.dims1()?]);
    let mut choose_v = vec![];
    for i in 0..=idx0 {
        let mut v_j = vec![];
        for j in idx1.to_vec1::<u32>()? {
            let mut v_k = vec![];
            for k in idx2.to_vec1::<u32>()? {
                v_k.push(v[i][j as usize][k as usize]);
            }
            v_j.push(v_k);
        }
        choose_v.push(v_j);
    }
    assert_eq!(choose.to_vec3::<f32>()?, choose_v);

    // 如果是 Select(usize)，維度會與原本的維度少 1 維。
    let v = t.i(0)?;
    assert_eq!(v.dims(), &[dims[1], dims[2]]);
    let v = t.i((.., 0))?;
    assert_eq!(v.dims(), &[dims[0], dims[2]]);
    let v = t.i((.., .., 0))?;
    assert_eq!(v.dims(), &[dims[0], dims[1]]);

    Ok(())
}
