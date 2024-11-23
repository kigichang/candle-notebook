use candle_core::{Device, Tensor};
use yew::prelude::*;

#[function_component]
pub fn ShowTensor() -> Html {
    html! {
        <h3>{"Test Tensor Only: "}{Tensor::new(0u32, &Device::Cpu).unwrap()}</h3>
    }
}
