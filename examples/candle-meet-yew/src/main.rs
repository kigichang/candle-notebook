use wasm_bindgen::{prelude::*, UnwrapThrowExt};
use web_sys::{console, window, Document, HtmlElement};
use yew::prelude::*;

mod bert_base_chinese;
mod pickle;
mod show_tensor;
use bert_base_chinese::BertBaseChinese;
use show_tensor::ShowTensor;

#[function_component]
fn App() -> Html {
    html! {
        <div class={classes!("container")}>
            <ShowTensor />
            <BertBaseChinese />
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
