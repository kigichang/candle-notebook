use candle_core::{Device, Tensor};
use yew::prelude::*;

#[function_component]
fn App() -> Html {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0], &Device::Cpu).unwrap();
    let b = a.sum(0).unwrap().to_vec0::<f32>().unwrap();
    html! {
        <div>
            <h1>{"Hello, Yew!"}</h1>
            <h1>{b}</h1>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
