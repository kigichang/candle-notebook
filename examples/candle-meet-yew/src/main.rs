use candle_core::{Device, Tensor};
use yew::prelude::*;

#[function_component]
fn App() -> Html {
    html! {
        <div>
            <h1>{"Hello, Yew!"}</h1>
            <h1>{Tensor::new(0u32, &Device::Cpu).unwrap().to_scalar::<u32>().unwrap()}</h1>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
