use candle_core::{Device, Result, Tensor};

#[test]
fn narrow() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;

    let values = t.to_vec3::<f32>()?;

    let v = t.narrow(0, 1, 1)?; // dim 0, start 1, length 1
    assert_eq!(v.dims(), &[1, 3, 4]);
    assert_eq!(v.to_vec3::<f32>()?, values[1..2]);

    let v = t.narrow(1, 1, 2)?; // dim 1, start 1, length 2
    assert_eq!(v.dims(), &[2, 2, 4]);
    assert_eq!(
        v.to_vec3::<f32>()?,
        vec![
            vec![values[0][1].clone(), values[0][2].clone()],
            vec![values[1][1].clone(), values[1][2].clone()]
        ]
    );

    let v = t.narrow(2, 1, 2)?; // dim 2, start 1, length 2
    assert_eq!(v.dims(), &[2, 3, 2]);
    assert_eq!(
        v.to_vec3::<f32>()?,
        vec![
            vec![
                vec![values[0][0][1], values[0][0][2]],
                vec![values[0][1][1], values[0][1][2]],
                vec![values[0][2][1], values[0][2][2]],
            ],
            vec![
                vec![values[1][0][1], values[1][0][2]],
                vec![values[1][1][1], values[1][1][2]],
                vec![values[1][2][1], values[1][2][2]],
            ]
        ]
    );

    Ok(())
}
