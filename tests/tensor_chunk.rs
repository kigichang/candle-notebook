use candle_core::{Device, Result, Tensor};

#[test]
fn tensor_chunk() -> Result<()> {
    let t = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v = t.to_vec3::<f32>()?;

    let chunks = t.chunk(2, 0)?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[1, 3, 4]);
    assert_eq!(chunks[1].dims(), &[1, 3, 4]);

    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.to_vec3::<f32>()?, v[i..(i + 1)]);
    }

    let chunks = t.chunk(2, 1)?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[2, 2, 4]);
    assert_eq!(chunks[1].dims(), &[2, 1, 4]);

    let chunks = t.chunk(3, 2)?;
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].dims(), &[2, 3, 2]);
    assert_eq!(chunks[1].dims(), &[2, 3, 1]);
    assert_eq!(chunks[2].dims(), &[2, 3, 1]);

    Ok(())
}
