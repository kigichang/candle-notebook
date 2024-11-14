use candle_core::{utils, DType, Device, Result, Tensor, D};
use candle_datasets::vision::{mnist, Dataset};
use candle_nn::{
    loss, ops, sequential, Conv2d, Linear, Module, ModuleT, Optimizer, Sequential, VarBuilder,
    VarMap, SGD,
};

use clap::{Parser, ValueEnum};

use rand::prelude::*;

const IMAGE_DIM: usize = 28 * 28;
const LABELS: usize = 10;

//------------------------------------------------------------------------------

trait Model: Sized + Module {
    fn new(vb: VarBuilder) -> Result<Self>;
}

//------------------------------------------------------------------------------

struct LinearModel {
    seq: Sequential,
}

impl Model for LinearModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let seq = sequential::seq();

        //let w = vb.get_with_hints((LABELS, IMAGE_DIM), "weight", candle_nn::init::ZERO)?;
        //let b = vb.get_with_hints(LABELS, "bias", candle_nn::init::ZERO)?;

        let linear = candle_nn::linear(IMAGE_DIM, LABELS, vb.pp("linear"))?;
        //let linear = Linear::new(w, Some(b));
        let seq = seq.add(linear);
        let seq = seq.add(LogSoftMax);
        Ok(Self { seq })
    }
}

impl Module for LinearModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.seq.forward(input)
    }
}

//------------------------------------------------------------------------------

struct Mlp {
    seq: Sequential,
}

impl Model for Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        let seq = sequential::seq();

        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vb.pp("ln2"))?;

        let seq = seq.add(ln1).add(Relu).add(ln2);
        let seq = seq.add(LogSoftMax);
        Ok(Self { seq })
    }
}

impl Module for Mlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.seq.forward(input)
    }
}

//------------------------------------------------------------------------------

struct LogSoftMax;

impl Module for LogSoftMax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        ops::log_softmax(input, D::Minus2)
    }
}

//------------------------------------------------------------------------------

struct Relu;

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }
}

//------------------------------------------------------------------------------

#[derive(Debug)]
struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
    use_gpu: bool,
}

//------------------------------------------------------------------------------

fn use_device(use_gpu: bool) -> Result<Device> {
    if use_gpu {
        if utils::cuda_is_available() {
            Device::new_cuda(0)
        } else if utils::metal_is_available() {
            Device::new_metal(0)
        } else {
            Ok(candle_core::Device::Cpu)
        }
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

// -----------------------------------------------------------------------------

fn training_loop<M: Model>(m: Dataset, args: TrainingArgs) -> anyhow::Result<()> {
    let device = use_device(args.use_gpu)?;
    println!("use device: {:?}", device);
    let train_images = m.train_images.to_device(&device)?;
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&device)?;

    let test_images = m.test_images.to_device(&device)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = M::new(vb.clone())?;

    if let Some(load) = args.load {
        println!("loading model from: {:?}", load);
        varmap.load(load)?;
    }

    let mut sgd = SGD::new(varmap.all_vars(), args.learning_rate)?;

    for epoch in 1..=args.epochs {
        // let logits = model.forward(&train_images)?;
        // let log_sm = ops::log_softmax(&logits, D::Minus2)?;
        let log_sm = model.forward(&train_images)?;
        let loss = loss::nll(&log_sm, &train_labels)?;

        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }

    if let Some(save) = args.save {
        println!("saving trained model to: {:?}", save);
        varmap.save(save)?;
    }

    Ok(())
}

//------------------------------------------------------------------------------

#[derive(ValueEnum, Debug, Clone)]
enum WhichModel {
    Linear,
    Mlp,
    Cnn,
}

//------------------------------------------------------------------------------

#[derive(Debug)]
struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

// -----------------------------------------------------------------------------

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let dev = use_device(args.use_gpu)?;
    println!("use device: {:?}", dev);

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::new(vs.clone())?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let n_batches = train_images.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    for epoch in 1..args.epochs {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward(&train_images, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    }
    Ok(())
}

//------------------------------------------------------------------------------

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_enum, default_value_t = WhichModel::Linear)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    #[arg(long)]
    save: Option<String>,

    #[arg(long)]
    load: Option<String>,

    #[arg(long)]
    local_mnist: Option<String>,

    #[arg(long, default_value_t = false)]
    use_gpu: bool,
}

//------------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let m = if let Some(directory) = args.local_mnist {
        mnist::load_dir(directory)?
    } else {
        mnist::load()?
    };

    println!("training_images: {:?}", m.train_images.shape());
    println!("training_labels: {:?}", m.train_labels.shape());
    println!("test_images: {:?}", m.test_images.shape());
    println!("test_labels: {:?}", m.test_labels.shape());

    let default_learning_rate = match args.model {
        WhichModel::Linear => 1.0,
        WhichModel::Mlp => 0.05,
        WhichModel::Cnn => 0.001,
    };

    let training_args = TrainingArgs {
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load: args.load,
        save: args.save,
        epochs: args.epochs,
        use_gpu: args.use_gpu,
    };

    println!("training model: {:?}", args.model);
    println!("training_args: {:?}", training_args);

    match args.model {
        WhichModel::Linear => training_loop::<LinearModel>(m, training_args),
        WhichModel::Mlp => training_loop::<Mlp>(m, training_args),
        WhichModel::Cnn => training_loop_cnn(m, training_args),
    }
}
