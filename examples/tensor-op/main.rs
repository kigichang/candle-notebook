fn main() {}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    #[test]
    fn new() -> Result<()> {
        // scalar
        let t = Tensor::new(1.0f32, &Device::Cpu)?;
        println!("{:?}", t.shape());
        assert_eq!(t.to_scalar::<f32>()?, 1.0);
        assert_eq!(t.to_vec0::<f32>()?, 1.0);

        // vector
        let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
        println!("{:?}", t.shape());
        assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

        // 2x2 矩陣
        let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
        println!("{:?}", t.shape());
        assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // 2x2x2 矩陣
        let t = Tensor::new(
            &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            &Device::Cpu,
        )?;
        println!("{:?}", t.shape());
        assert_eq!(
            t.to_vec3::<f32>()?,
            vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]]
            ]
        );

        Ok(())
    }

    #[test]
    fn zeros() -> Result<()> {
        // vector
        let t = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
        assert_eq!(t.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0]);

        // 2x2 矩陣
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(t.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);

        // 2x2x2 矩陣
        let t = Tensor::zeros((2, 2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(
            t.to_vec3::<f32>()?,
            vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ]
        );

        // scalar
        let t = Tensor::new(1.0f32, &Device::Cpu)?;
        let z = t.zeros_like()?;
        assert_eq!(z.to_scalar::<f32>()?, 0.0);
        assert_eq!(z.shape(), t.shape());

        // vector
        let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
        let z = t.zeros_like()?;
        assert_eq!(z.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(z.shape(), t.shape());

        // 2x2 矩陣
        let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
        let z = t.zeros_like()?;
        assert_eq!(z.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(z.shape(), t.shape());

        // 2x2x2 矩陣
        let t = Tensor::new(
            &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            &Device::Cpu,
        )?;
        let z = t.zeros_like()?;
        assert_eq!(
            z.to_vec3::<f32>()?,
            vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ]
        );
        assert_eq!(z.shape(), t.shape());

        Ok(())
    }

    #[test]
    fn ones() -> Result<()> {
        // vector
        let t = Tensor::ones(3, DType::F32, &Device::Cpu)?;
        assert_eq!(t.to_vec1::<f32>()?, vec![1., 1., 1.]);

        // 2x2 矩陣
        let t = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(t.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);

        // 2x2x2 矩陣
        let t = Tensor::ones((2, 2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(
            t.to_vec3::<f32>()?,
            vec![
                vec![vec![1., 1.], vec![1., 1.]],
                vec![vec![1., 1.], vec![1., 1.]]
            ]
        );

        // scalar
        let t = Tensor::new(0.0f32, &Device::Cpu)?;
        let z = t.ones_like()?;
        assert_eq!(z.to_scalar::<f32>()?, 1.);
        assert_eq!(z.shape(), t.shape());

        // vector
        let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
        let z = t.ones_like()?;
        assert_eq!(z.to_vec1::<f32>()?, vec![1., 1., 1., 1.]);
        assert_eq!(z.shape(), t.shape());

        // 2x2 矩陣
        let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
        let z = t.ones_like()?;
        assert_eq!(z.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);
        assert_eq!(z.shape(), t.shape());

        // 2x2x2 矩陣
        let t = Tensor::new(
            &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            &Device::Cpu,
        )?;
        let z = t.ones_like()?;
        assert_eq!(
            z.to_vec3::<f32>()?,
            vec![
                vec![vec![1., 1.], vec![1., 1.]],
                vec![vec![1., 1.], vec![1., 1.]]
            ]
        );
        assert_eq!(z.shape(), t.shape());

        Ok(())
    }

    #[test]
    fn full() -> Result<()> {
        // vector
        let t = Tensor::full(1.0f32, 2, &Device::Cpu)?;
        assert_eq!(t.to_vec1::<f32>()?, [1., 1.]);
        let z = Tensor::ones(2, DType::F32, &Device::Cpu)?;
        assert_eq!(t.shape(), z.shape());
        assert_eq!(t.to_vec1::<f32>()?, z.to_vec1::<f32>()?);

        // 2x2 矩陣
        let t = Tensor::full(0.0f32, (2, 2), &Device::Cpu)?;
        assert_eq!(t.to_vec2::<f32>()?, vec![vec![0., 0.], vec![0., 0.]]);
        let z = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(t.shape(), z.shape());
        assert_eq!(t.to_vec2::<f32>()?, z.to_vec2::<f32>()?);

        // 2x2x2 矩陣
        let t = Tensor::full(-1.0f32, (2, 2, 2), &Device::Cpu)?;
        assert_eq!(
            t.to_vec3::<f32>()?,
            vec![
                vec![vec![-1., -1.], vec![-1., -1.]],
                vec![vec![-1., -1.], vec![-1., -1.]]
            ]
        );

        Ok(())
    }
}
