use candle_core::{DType, Device, Result, Tensor, WithDType, D};

// U8, U32, I64, BF16, F16, F32, F64

const DTYPES: [DType; 7] = [
    DType::U8,
    DType::U32,
    DType::I64,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
];

#[test]
fn f64_to() -> Result<()> {
    let max = Tensor::new(std::f64::MAX, &Device::Cpu)?;
    let min = Tensor::new(std::f64::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn f32_to() -> Result<()> {
    let max = Tensor::new(std::f32::MAX, &Device::Cpu)?;
    let min = Tensor::new(std::f32::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn f16_to() -> Result<()> {
    let max = Tensor::new(half::f16::MAX, &Device::Cpu)?;
    let min = Tensor::new(half::f16::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn bf16_to() -> Result<()> {
    let max = Tensor::new(half::bf16::MAX, &Device::Cpu)?;
    let min = Tensor::new(half::bf16::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn i64_to() -> Result<()> {
    let max = Tensor::new(i64::MAX, &Device::Cpu)?;
    let min = Tensor::new(i64::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn u32_to() -> Result<()> {
    let max = Tensor::new(u32::MAX, &Device::Cpu)?;
    let min = Tensor::new(u32::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}

#[test]
fn u8_to() -> Result<()> {
    let max = Tensor::new(u8::MAX, &Device::Cpu)?;
    let min = Tensor::new(u8::MIN, &Device::Cpu)?;

    for dtype in DTYPES.iter() {
        let t = max.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);

        let t = min.to_dtype(*dtype)?;
        assert_eq!(t.dtype(), *dtype);
    }

    Ok(())
}
