use std::time::{Duration, Instant};

use anyhow::{Ok, Result};
use candle_core::{Device, IndexOp, Tensor};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode};
use ratatui::{
    symbols::Marker,
    widgets::{
        canvas::{Canvas, Line},
        Widget,
    },
};

#[derive(Debug, PartialEq)]
enum Collision {
    None,
    X,
    Y,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 0.1)]
    width: f32,

    #[arg(long, default_value_t = 0.1)]
    height: f32,

    #[arg(long, default_value_t = 32u64)]
    tick_rate: u64,

    #[arg(long, short = 'x', default_value_t = 1.0)]
    viewport_x: f32,

    #[arg(long, short = 'y', default_value_t = 1.0)]
    viewport_y: f32,

    #[arg(long, short = 'd', default_value_t = 0.01)]
    displacement: f32,

    #[arg(long, short = 'r', default_value_t = 0.001)]
    rotation: f32,
}

struct App {
    args: Args,
    points: Tensor,
    device: Device,
    dx: f32,
    dy: f32,
    theta: f32,
    d_theta: f32,
}

impl TryFrom<Args> for App {
    type Error = anyhow::Error;

    fn try_from(args: Args) -> Result<Self> {
        let device = candle_notebook::device(args.cpu)?;

        let w = args.width / 2.0;
        let h = args.height / 2.0;
        let points = Tensor::new(&[[w, h], [-w, h], [-w, -h], [w, -h]], &device)?;

        Ok(Self {
            args,
            points,
            device,
            dx: 0.0,
            dy: 0.0,
            theta: 0.0,
            d_theta: 0.0,
        })
    }
}

impl App {
    pub fn run(&mut self, mut terminal: ratatui::DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(self.args.tick_rate);
        let mut last_tick = Instant::now();
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                        KeyCode::Up | KeyCode::Char('k') => {
                            if self.dy <= 0.0 {
                                self.dy = self.args.displacement;
                            } else {
                                self.dy += self.args.displacement;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if self.dy >= 0.0 {
                                self.dy = -self.args.displacement;
                            } else {
                                self.dy -= self.args.displacement;
                            }
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            if self.dx >= 0.0 {
                                self.dx = -self.args.displacement;
                            } else {
                                self.dx -= self.args.displacement;
                            }
                        }
                        KeyCode::Right | KeyCode::Char('l') => {
                            if self.dx <= 0.0 {
                                self.dx = self.args.displacement;
                            } else {
                                self.dx += self.args.displacement;
                            }
                        }
                        KeyCode::Char('a') => {
                            if self.d_theta <= 0.0 {
                                self.d_theta = self.args.rotation;
                            } else {
                                self.d_theta += self.args.rotation;
                            }
                        }
                        KeyCode::Char('s') => {
                            if self.d_theta >= 0.0 {
                                self.d_theta = -self.args.rotation;
                            } else {
                                self.d_theta -= self.args.rotation;
                            }
                        }
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                self.on_tick()?;
                last_tick = Instant::now();
            }
        }
    }

    /// 重新繪製畫面
    fn draw(&self, frame: &mut ratatui::Frame) {
        frame.render_widget(self.rectangle_canvas(), frame.area());
    }

    /// 繪畫矩形
    fn rectangle_canvas(&self) -> impl Widget + '_ {
        Canvas::default()
            .marker(Marker::Dot)
            .paint(|ctx| {
                let points = self.points.to_vec2::<f32>().unwrap();
                let line1 = Line {
                    x1: points[0][0] as f64,
                    y1: points[0][1] as f64,
                    x2: points[1][0] as f64,
                    y2: points[1][1] as f64,
                    color: ratatui::style::Color::Red,
                };

                let line2 = Line {
                    x1: points[1][0] as f64,
                    y1: points[1][1] as f64,
                    x2: points[2][0] as f64,
                    y2: points[2][1] as f64,
                    color: ratatui::style::Color::Blue,
                };

                let line3 = Line {
                    x1: points[2][0] as f64,
                    y1: points[2][1] as f64,
                    x2: points[3][0] as f64,
                    y2: points[3][1] as f64,
                    color: ratatui::style::Color::Green,
                };

                let line4 = Line {
                    x1: points[3][0] as f64,
                    y1: points[3][1] as f64,
                    x2: points[0][0] as f64,
                    y2: points[0][1] as f64,
                    color: ratatui::style::Color::White,
                };

                ctx.draw(&line1);
                ctx.draw(&line2);
                ctx.draw(&line3);
                ctx.draw(&line4);
            })
            .x_bounds([-self.args.viewport_x as f64, self.args.viewport_x as f64])
            .y_bounds([-self.args.viewport_y as f64, self.args.viewport_y as f64])
    }

    /// 偵測是否碰撞到邊界
    fn detect(&self, points: &Tensor) -> Result<Collision> {
        let x = points.i((.., 0))?;
        let y = points.i((.., 1))?;
        let x_max = x.max(0)?.to_scalar::<f32>()?;
        let x_min = x.min(0)?.to_scalar::<f32>()?;
        let y_max = y.max(0)?.to_scalar::<f32>()?;
        let y_min = y.min(0)?.to_scalar::<f32>()?;

        if x_max > self.args.viewport_x || x_min < -self.args.viewport_x {
            return Ok(Collision::X);
        }

        if y_max > self.args.viewport_y || y_min < -self.args.viewport_y {
            return Ok(Collision::Y);
        }

        Ok(Collision::None)
    }

    // 每次 tick 時的重新計算矩形的 4 個頂點
    fn on_tick(&mut self) -> Result<()> {
        let centroid = centroid(&self.points)?;
        let points = self.points.broadcast_sub(&centroid)?; // 將中心點移至原點

        // 旋轉
        self.theta += self.d_theta;
        if self.theta > std::f32::consts::FRAC_2_PI || self.theta < -std::f32::consts::FRAC_2_PI {
            self.theta = 0.0;
        }
        let rotate = rotation(self.theta, &self.device)?.t()?;
        let points = points.matmul(&rotate)?;

        // 位移
        let displacement = displacement(self.dx, self.dy, &self.device)?;
        let points = points.broadcast_add(&displacement)?; // 位移

        let points = points.broadcast_add(&centroid)?;

        // 偵測是否碰撞到邊界
        match self.detect(&points)? {
            Collision::None => {
                self.points = points;
            }
            Collision::X => self.dx = -self.dx,
            Collision::Y => self.dy = -self.dy,
        }
        Ok(())
    }
}

// displacement 位移向量
fn displacement(dx: f32, dy: f32, device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(&[dx, dy], &device)?)
}

// rotation 旋轉矩陣
fn rotation(theta: f32, device: &Device) -> Result<Tensor> {
    let (s, c) = theta.sin_cos();
    Ok(Tensor::new(&[[c, -s], [s, c]], &device)?)
}

// centroid 計算中心點
fn centroid(points: &Tensor) -> Result<Tensor> {
    Ok(points.mean(0)?)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut app = App::try_from(args)?;
    let terminal = ratatui::init();
    let app_result = app.run(terminal);
    ratatui::restore();
    app_result
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn add_and_broadcast_add() -> Result<()> {
        todo!("compare add and broadcast_add")
    }

    #[test]
    fn sum_and_mean() -> Result<()> {
        let t = Tensor::from_iter(0i64..24, &Device::Cpu)?
            .reshape((2, 3, 4))?
            .to_dtype(DType::F32)?; // 3x3x3

        println!("{:?}", t.to_vec3::<f32>()?);
        println!("{:?}", t.dims3()?);

        let sum0 = t.sum(0)?;
        println!("sum0: {:?}", sum0.to_vec2::<f32>()?);
        let sum1 = t.sum(1)?;
        println!("sum1: {:?}", sum1.to_vec2::<f32>()?);
        let sum2 = t.sum(2)?;
        println!("sum2: {:?}", sum2.to_vec2::<f32>()?);

        let mean0 = t.mean(0)?;
        println!("mean0: {:?}", mean0.to_vec2::<f32>()?);
        let mean1 = t.mean(1)?;
        println!("mean1: {:?}", mean1.to_vec2::<f32>()?);
        let mean2 = t.mean(2)?;
        println!("mean2: {:?}", mean2.to_vec2::<f32>()?);

        Ok(())
    }
}
