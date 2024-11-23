use std::{
    error::Error,
    fmt::{Debug, Display},
    future,
    iter::Successors,
    str::FromStr,
};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::bert::{BertForMaskedLM, Config};
use gloo::console;
use safetensors;
use tokenizers::{tokenizer, Tokenizer};
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

pub enum FetchState<T> {
    NotFetching,
    Fetching,
    Success(T),
    Failed(FetchError),
}

pub enum Msg {
    SetTokenizers(FetchState<Tokenizer>),
    GetConfig(FetchState<Config>),
    GetModel(FetchState<BertForMaskedLM>),
    DownloadTokenizer,
    Inference(String),
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
    model: FetchState<BertForMaskedLM>,
    config: FetchState<Config>,
    tokenizers: FetchState<Tokenizer>,
}

impl Component for BertBaseChinese {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        Self {
            model: FetchState::NotFetching,
            config: FetchState::NotFetching,
            tokenizers: FetchState::NotFetching,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::DownloadTokenizer => {
                console::log!("download tokenizer");
                ctx.link().send_future(async {
                    gloo::utils::document()
                        .get_element_by_id("result")
                        .and_then(|elem| {
                            let content = elem.text_content().and_then(|s| {
                                Some(format!("{:?}\n{:?}", s, "downloading tokenizers"))
                            });
                            Some(elem.set_text_content(content.as_deref()))
                        });
                    let tokenizer_json = fetch_tokenizer().await.unwrap();
                    Msg::SetTokenizers(FetchState::Success(
                        Tokenizer::from_bytes(&tokenizer_json).unwrap(),
                    ))
                });

                ctx.link().send_future(async {
                    gloo::utils::document()
                        .get_element_by_id("result")
                        .and_then(|elem| {
                            let content = elem.text_content().and_then(|s| {
                                Some(format!("{:?}\n{:?}", s, "downloading config.json"))
                            });
                            Some(elem.set_text_content(content.as_deref()))
                        });
                    let config_json = fetch_config().await.unwrap();
                    let config = serde_json::from_str::<Config>(&config_json).unwrap();
                    Msg::GetConfig(FetchState::Success(config))
                });
            }
            Msg::SetTokenizers(tokenizers) => {
                console::log!("tokenizers loaded");
                self.tokenizers = tokenizers;
                gloo::utils::document()
                    .get_element_by_id("result")
                    .and_then(|elem| Some(elem.set_text_content(Some("tokenizers downloaded"))));
            }
            Msg::GetConfig(config) => {
                self.config = config;
                ctx.link().send_future(async {
                    let model_bytes = fetch_model().await.unwrap();
                    let vb = VarBuilder::from_buffered_safetensors(
                        model_bytes,
                        DType::F32,
                        &Device::Cpu,
                    )
                    .unwrap();
                    match config {
                        FetchState::Success(cfg) => {
                            let model = BertForMaskedLM::load(vb, &cfg).unwrap();
                        }
                        _ => {
                            console::error!("failed to load config");
                        }
                    }

                    Msg::Inference("".to_owned())
                });
            }
            Msg::GetModel(model) => {
                self.model = model;
            }
            Msg::Inference(input) => {
                console::log!("inference: {:?}", input);
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div>
                <h1>{"Bert-Base-Chinese"}</h1>
                <div>
                    <div class="mb-3">
                        <label for="input_example" class="form-label">{"例句"}</label>
                        <div class="row">
                            <input type="text" class="form-control col" id="input_example" placeholder="" />
                            <button id="btn_submit" class="btn btn-primary col-1 ms-3" >{"送出"}</button>
                            <button id="btn_download" class="btn btn-secondary col-1 ms-3" onclick={ctx.link().callback(|_| Msg::DownloadTokenizer)}>{"下載模型"}</button>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label for="result" class="form-label">{"結果"}</label>
                        <textarea class="form-control" id="result" rows="10"></textarea>
                    </div>
                </div>
            </div>
        }
    }
}

// impl Component for BertBaseChinese {
//     type Message = Msg;
//     type Properties = ();

//     fn create(_: Self::Properties, link: ComponentLink<Self>) -> Self {
//         Self {
//             model: FetchState::NotFetching,
//         }
//     }

//     fn update(&mut self, msg: Self::Message) -> ShouldRender {
//         match msg {
//             Msg::DownloadModel => {
//                 self.model = FetchState::Fetching;
//                 let future = async {
//                     let config_json = fetch_config().await.unwrap();
//                     console::log!("config_json: {:?}", config_json);
//                     let config = serde_json::from_str::<Config>(&config_json).unwrap();
//                     let model = BertForMaskedLM::load(config).unwrap();
//                     FetchState::Success(model)
//                 };
//                 wasm_bindgen_futures::spawn_local(async {
//                     match future.await {
//                         FetchState::Success(model) => {
//                             console::log!("model loaded");
//                             self.model = FetchState::Success(model);
//                         }
//                         FetchState::Failed(err) => {
//                             console::error!("failed to load model: {:?}", err);
//                             self.model = FetchState::Failed(err);
//                         }
//                         _ => unreachable!(),
//                     }
//                 });
//             }
//             Msg::Inference(input) => {
//                 console::log!("inference: {:?}", input);
//             }
//         }
//         true
//     }

//     fn change(&mut self, _: Self::Properties) -> ShouldRender {
//         false
//     }

//     fn view(&self) -> Html {
//         let download = Callback::from(|_| Msg::DownloadModel);
//         let inference = Callback::from(|input: InputData| Msg::Inference(input.value));
//         html! {
//             <div>
//                 <h1>{"Bert-Base-Chinese"}</h1>
//                 <div>
//                     <div class="mb-3">
//                         <label for="input_example" class="form-label">{"例句"}</label>
//                         <div class="row">
//                             <input type="text" class="form-control col" id="input_example" placeholder="" oninput={inference} />
//                             <button id="btn_submit" class="btn btn-primary col-1 ms-3" >{"送出"}</button>
//                             <button id="btn_download" class="btn btn-secondary col-1 ms-3" onclick={download}>{"下載模型"}</button>
//                         </div>

//                     </div>
// }

// #[function_component]
// pub fn BertBaseChinese() -> Html {
//     let download = Callback::from(|_| {
//         console::log!("download config");
//         let _config_json = fetch_config();
//     });

//     html! {
//         <div>
//             <h1>{"Bert-Base-Chinese"}</h1>
//             <div>
//                 <div class="mb-3">
//                     <label for="input_example" class="form-label">{"例句"}</label>
//                     <div class="row">
//                         <input type="text" class="form-control col" id="input_example" placeholder="" />
//                         <button id="btn_submit" class="btn btn-primary col-1 ms-3" >{"送出"}</button>
//                         <button id="btn_download" class="btn btn-secondary col-1 ms-3" onclick={download}>{"下載模型"}</button>
//                     </div>

//                 </div>
//                 <div class="mb-3">
//                     <label for="result" class="form-label">{"結果"}</label>
//                     <textarea class="form-control" id="result" rows="3"></textarea>
//                 </div>
//             </div>
//         </div>
//     }
// }
