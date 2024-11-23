use yew::prelude::*;

mod bert_base_chinese;
mod show_tensor;
use bert_base_chinese::BertBaseChinese;
use show_tensor::ShowTensor;

#[function_component]
fn App() -> Html {
    html! {
        <div class={classes!("container")}>
            <ShowTensor />
            <BertBaseChinese can_inference={false} />
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
