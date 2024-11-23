use std::{
    error::Error,
    fmt::{Debug, Display},
};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertForMaskedLM, Config};
use gloo::console;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::Response;
use yew::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct FetchError {
    err: JsValue,
}

impl Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.err, f)
    }
}
impl Error for FetchError {}

impl From<JsValue> for FetchError {
    fn from(err: JsValue) -> Self {
        Self { err }
    }
}

// pub enum FetchState<T> {
//     NotFetching,
//     Fetching,
//     Success(T),
//     Failed(FetchError),
// }

pub enum Msg {
    DownloadModel,
    TokenizerDownloaded(Tokenizer),
    ConfigDownloaded(Config),
    DownloadBertModel,
    ModelDownloaded(Vec<u8>),
    Input,
    Inference(String),
}

#[derive(Properties, PartialEq)]
pub struct Props {
    pub can_inference: bool,
}

fn change_status(new_state: &str) {
    gloo::utils::document()
        .get_element_by_id("result")
        .and_then(|elem| {
            let content = elem.inner_html();
            let content = if content.is_empty() {
                new_state.to_owned()
            } else {
                console::log!("old:", content.clone());
                content + "<br />" + new_state
            };
            elem.set_inner_html(&content);
            Some(())
        });
}

fn clear_status() {
    gloo::utils::document()
        .get_element_by_id("result")
        .and_then(|elem| {
            elem.set_inner_html("");
            Some(())
        });
}

async fn fetch_config() -> Result<String, JsValue> {
    console::log!("fetch config");
    let window = gloo::utils::window();
    let resp_value = JsFuture::from(window.fetch_with_str("/config.json")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}

async fn fetch_tokenizer() -> Result<Vec<u8>, JsValue> {
    console::log!("fetch tokenizer");
    let window = gloo::utils::window();
    let resp_value = JsFuture::from(window.fetch_with_str("/tokenizer.json")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let buf = JsFuture::from(resp.array_buffer()?).await?;

    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

async fn fetch_model() -> Result<Vec<u8>, JsValue> {
    console::log!("fetch model");
    let window = gloo::utils::window();
    let resp_value =
        JsFuture::from(window.fetch_with_str("/fix-bert-base-chinese.safetensors")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let buf = JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

pub struct BertBaseChinese {
    config: Option<Config>,
    model: Option<BertForMaskedLM>,
    tokenizer: Option<Tokenizer>,
}

impl BertBaseChinese {
    fn inference(&self, test_str: &str) {
        let device = &Device::Cpu;
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let mask_id: u32 = tokenizer.token_to_id("[MASK]").unwrap();
        let ids = tokenizer.encode(test_str, true).unwrap();
        let input_ids = Tensor::stack(&[Tensor::new(ids.get_ids(), &device).unwrap()], 0).unwrap();
        let token_type_ids =
            Tensor::stack(&[Tensor::new(ids.get_type_ids(), &device).unwrap()], 0).unwrap();
        let attention_mask = Tensor::stack(
            &[Tensor::new(ids.get_attention_mask(), &device).unwrap()],
            0,
        )
        .unwrap();
        let result = self
            .model
            .as_ref()
            .unwrap()
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .unwrap();

        let mask_idx = ids.get_ids().iter().position(|&x| x == mask_id).unwrap();
        let mask_token_logits = result.i((0, mask_idx, ..)).unwrap();
        let mask_token_probs = candle_nn::ops::softmax(&mask_token_logits, 0).unwrap();
        let mut top5_tokens: Vec<(usize, f32)> = mask_token_probs
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .enumerate()
            .collect();
        top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

        //println!("Input: {}", test_str);
        clear_status();
        for (idx, prob) in top5_tokens {
            change_status(&format!(
                "{:?}: {:.3}",
                tokenizer.id_to_token(idx as u32).unwrap(),
                prob
            ));
        }
    }
}

impl Component for BertBaseChinese {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            config: None,
            model: None,
            tokenizer: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::DownloadModel => {
                ctx.link().send_future(async {
                    change_status("downloading tokenizers");
                    let tokenizer_json = fetch_tokenizer().await.unwrap();
                    let tokenizer = Tokenizer::from_bytes(&tokenizer_json).unwrap();
                    Msg::TokenizerDownloaded(tokenizer)
                });

                ctx.link().send_future(async {
                    change_status("downloading config.json");
                    let config_json = fetch_config().await.unwrap();
                    let config = serde_json::from_str::<Config>(&config_json).unwrap();
                    Msg::ConfigDownloaded(config)
                });
            }
            Msg::TokenizerDownloaded(tokenizer) => {
                change_status("tokenizer downloaded");
                self.tokenizer = Some(tokenizer);
            }
            Msg::ConfigDownloaded(config) => {
                change_status("config downloaded");
                self.config = Some(config);
                ctx.link().send_message(Msg::DownloadBertModel);
            }
            Msg::DownloadBertModel => {
                change_status("downloading model");
                ctx.link().send_future(async {
                    let model_bytes = fetch_model().await.unwrap();
                    Msg::ModelDownloaded(model_bytes)
                });
            }
            Msg::ModelDownloaded(byte_buf) => {
                change_status("model downloaded");
                change_status("loading model");
                let vb = VarBuilder::from_buffered_safetensors(byte_buf, DType::F32, &Device::Cpu)
                    .unwrap();
                let model = BertForMaskedLM::load(vb, self.config.as_ref().unwrap()).unwrap();
                self.model = Some(model);
                change_status("model loaded");
                let btn = gloo::utils::document()
                    .get_element_by_id("btn_submit")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlElement>()
                    .unwrap();
                btn.class_list().remove_1("disabled").unwrap();
            }
            Msg::Input => {
                if self.model.is_none() {
                    change_status("model not loaded");
                    ctx.link().send_message(Msg::DownloadModel);
                    return false;
                }

                let input_str = gloo::utils::document()
                    .get_element_by_id("input_example")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlInputElement>()
                    .unwrap()
                    .value();
                let btn = gloo::utils::document()
                    .get_element_by_id("btn_submit")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlElement>()
                    .unwrap();
                btn.class_list().add_1("disabled").unwrap();
                clear_status();
                ctx.link().send_message(Msg::Inference(input_str));
            }
            Msg::Inference(test_str) => {
                self.inference(&test_str);
                let btn = gloo::utils::document()
                    .get_element_by_id("btn_submit")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlElement>()
                    .unwrap();
                btn.class_list().remove_1("disabled").unwrap();
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let do_inference = !ctx.props().can_inference;
        html! {
            <div>
                <h1>{"Bert-Base-Chinese"}</h1>
                <div>
                    <div class="mb-3">
                        <label for="input_example" class="form-label">{"例句"}</label>
                        <div class="row">
                            <input type="text" class="form-control col" id="input_example" placeholder="" value="巴黎是[MASK]国的首都。" />
                            <button id="btn_submit" class={classes!(
                                "btn",
                                "btn-primary",
                                "col-1",
                                "ms-3",
                                do_inference.then(|| Some("disabled"))
                                )}  onclick={ctx.link().callback(|_| Msg::Input)}>{"送出"}</button>
                            <button id="btn_download" class="btn btn-secondary col-1 ms-3" onclick={ctx.link().callback(|_| Msg::DownloadModel)}>{"下載模型"}</button>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div id="result" class="border border-primary" style="height:20rem"></div>
                    </div>
                </div>
            </div>
        }
    }
}
