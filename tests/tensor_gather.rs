use candle_core::{Device, Result, Shape, Tensor};
use rand::prelude::*;

#[test]
fn gather() -> Result<()> {
    // sample from pytorch
    // out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    // out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    // out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    // torch.tensor([[1, 2], [3, 4]])
    // torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
    // tensor([[ 1,  1],
    //         [ 4,  3]])

    let t = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
    let v = t.to_vec2::<f32>()?;
    let index = Tensor::new(&[[0u32, 0], [1, 0]], &Device::Cpu)?;
    let idx = index.to_vec2::<u32>()?;

    let result = t.gather(&index, 1)?;

    let mut result_v = vec![];
    for i in 0..2 {
        let mut result_vv = vec![];
        for j in 0..2 {
            result_vv.push(v[i][idx[i][j] as usize])
        }
        result_v.push(result_vv);
    }

    assert_eq!(result.to_vec2::<f32>()?, result_v);
    Ok(())
}

fn gen_index<S: Into<Shape>>(len: u32, shape: S) -> Result<Tensor> {
    let shape = shape.into();
    let dims = shape.dims();

    let mut total = 1;
    for v in dims {
        total *= v;
    }

    let mut values = vec![0u32; total];
    rand::thread_rng().fill(&mut values[..]);
    Tensor::from_vec(
        values.into_iter().map(|v| v % len).collect::<Vec<u32>>(),
        shape,
        &Device::Cpu,
    )
}

#[test]
fn gather_formula() -> Result<()> {
    // out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    // out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    // out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    let t = Tensor::rand(-1.0f32, 1.0f32, (2, 3, 4), &Device::Cpu)?;
    let v = t.to_vec3::<f32>()?;

    let idx0 = gen_index(2, t.shape())?;
    let idx_v0 = idx0.to_vec3::<u32>()?;

    let mut result_v = vec![];
    for i in 0..2 {
        let mut v_i = vec![];
        for j in 0..3 {
            let mut v_j = vec![];
            for k in 0..4 {
                v_j.push(v[idx_v0[i][j][k] as usize][j][k])
            }
            v_i.push(v_j);
        }
        result_v.push(v_i);
    }
    let result = t.gather(&idx0, 0)?;
    assert_eq!(result.to_vec3::<f32>()?, result_v);

    let idx1 = gen_index(3, t.shape())?;
    let idx_v1 = idx1.to_vec3::<u32>()?;

    let mut result_v = vec![];
    for i in 0..2 {
        let mut v_i = vec![];
        for j in 0..3 {
            let mut v_j = vec![];
            for k in 0..4 {
                v_j.push(v[i][idx_v1[i][j][k] as usize][k])
            }
            v_i.push(v_j);
        }
        result_v.push(v_i);
    }
    let result = t.gather(&idx1, 1)?;
    assert_eq!(result.to_vec3::<f32>()?, result_v);

    let idx2 = gen_index(4, t.shape())?;
    let idx_v2 = idx2.to_vec3::<u32>()?;

    let mut result_v = vec![];
    for i in 0..2 {
        let mut v_i = vec![];
        for j in 0..3 {
            let mut v_j = vec![];
            for k in 0..4 {
                v_j.push(v[i][j][idx_v2[i][j][k] as usize])
            }
            v_i.push(v_j);
        }
        result_v.push(v_i);
    }
    let result = t.gather(&idx2, 2)?;
    assert_eq!(result.to_vec3::<f32>()?, result_v);

    Ok(())
}
