use candle_core::{Device, Result, Tensor};

#[test]
fn transpose() -> Result<()> {
    let t = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape((2, 3, 4))?;
    //println!("{:?}", t.to_vec3::<u32>()?);

    let tensor_values = t.to_vec3::<u32>()?;
    let t0 = t.transpose(0, 1)?;
    assert_eq!(t0.dims(), vec![3, 2, 4]);
    //println!("{:?}", t0.to_vec3::<u32>()?);
    let mut v0 = vec![];
    for j in 0..3 {
        let mut v1 = vec![];
        for i in 0..2 {
            v1.push(tensor_values[i][j].clone());
        }
        v0.push(v1);
    }
    //println!("{:?}", v0);
    assert_eq!(t0.to_vec3::<u32>()?, v0);

    let t1 = t.transpose(0, 2)?;
    assert_eq!(t1.dims(), vec![4, 3, 2]);
    //println!("{:?}", t1.to_vec3::<u32>()?);
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
    //println!("{:?}", v2);
    assert_eq!(t1.to_vec3::<u32>()?, v2);

    let t2 = t.transpose(1, 2)?;
    //println!("{:?}", t2.to_vec3::<u32>()?);
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
    //println!("{:?}", v0);
    assert_eq!(t2.to_vec3::<u32>()?, v0);

    Ok(())
}
