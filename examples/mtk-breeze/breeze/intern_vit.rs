use candle_core::{Module, Result, Tensor};
use candle_nn::Conv2d;
use candle_nn::VarBuilder;

// pub struct InternRMSNorm {
//     pub weight: Tensor,
//     pub variance_epsilon: f32,
// }

// impl InternRMSNorm {
//     pub fn load(vb: VarBuilder, variance_epsilon: f32) -> Result<Self> {
//         let weight = vb.get("weight")?;
//     }

//     pub fn forward(&self, input: &Tensor) -> Tensor {
//         let mean = input.mean(1, true);
//         let variance = input.variance(1, true, true);
//         let normalized = (input - mean) / (variance + self.variance_epsilon).sqrt();
//         normalized * self.weight
//     }
// }

/// A rough Candle equivalent of the Python `InternVisionEmbeddings`.
pub struct InternVisionEmbeddings {
    embed_dim: usize,
    image_size: usize,
    patch_size: usize,
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    num_patches: usize,
    num_positions: usize,
    position_embedding: Tensor,
}

impl InternVisionEmbeddings {
    /// Initializes parameters and Conv2d using Candle.
    pub fn new(vb: &VarBuilder, config: super::config::VisionConfig) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let image_size = config.image_size;
        let patch_size = config.patch_size;

        // Create trainable tensors/parameters
        let class_embedding = vb.get((1, 1, embed_dim), "class_embedding")?;
        let patch_embedding = candle_nn::conv2d(
            3,          // in_channels
            embed_dim,  // out_channels
            patch_size, // kernel_size
            candle_nn::Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let num_patches = (image_size / patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_embedding = vb.get((1, num_positions, embed_dim), "position_embedding")?;

        Ok(Self {
            embed_dim,
            image_size,
            patch_size,
            class_embedding,
            patch_embedding,
            num_patches,
            num_positions,
            position_embedding,
        })
    }

    /// A basic forward method that extracts patch embeddings, flattens, and prepends class embedding.
    /// (Interpolation is omitted, as Candle does not yet provide built-in bicubic interpolation.)
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // 1) Convolution to get patch embeddings
        //    shape after conv ~ [batch_size, embed_dim, H/patch_size, W/patch_size]
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        // 2) Flatten + transpose to [batch_size, H*W, embed_dim] style
        //    Candle note: reshape/transpose can be done via `reshaped`, `transpose`, etc.
        let shape = patch_embeds.shape();
        let batch_size = shape[0];
        let embed_dim = shape[1];
        let height = shape[2];
        let width = shape[3];

        // Flatten the last two dims, then transpose from [batch, embed_dim, h*w] -> [batch, h*w, embed_dim]
        let patch_embeds = patch_embeds
            .reshaped(&[batch_size, embed_dim, height * width])?
            .transpose(1, 2)?;

        // 3) Expand class embedding: shape [1, 1, embed_dim] -> [batch_size, 1, embed_dim]
        //    Then concatenate with patch_embeds across dim=1
        let class_embeds = self
            .class_embedding
            .broadcast_as(&[batch_size, 1, embed_dim])?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;

        // 4) Position embeddings: first token + the rest.
        //    Candle does not currently have built-in bicubic interpolation, so this step is simplified:
        let cls_pos = self.position_embedding.i(0)?.i((.., ..1, ..))?; // shape [1, 1, embed_dim]
        let patch_pos = self.position_embedding.i(0)?.i((.., 1.., ..))?; // shape [1, num_patches, embed_dim]

        // For demonstration, we simply slice them as-is:
        let pos_embed = Tensor::cat(&[cls_pos, patch_pos], 1)?.broadcast_as(&[
            batch_size,
            self.num_positions,
            embed_dim,
        ])?;

        // 5) Final embedding = (patch embeddings + class token) + position embeddings
        let output = embeddings + pos_embed;
        Ok(output)
    }
}
