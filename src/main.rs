#[macro_use]
extern crate lazy_static;
extern crate png;
extern crate rand;

use std::f32::consts::PI;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter::Sum;
use std::ops::Add;
use std::path::Path;
use std::result::Result;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{SeedableRng, Rng};
use rand_distr::{Normal, Uniform, WeightedIndex};
use rand_chacha::ChaCha8Rng;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const AREA_MARGIN: u32 = 180;
const NUM_SOURCES: usize = 14;
const SAMPLENOISE_STDDEV: f32 = 40.0;
const SAMPLE_MINDIST: f32 = 1e-10;
// Should be between -inf and 0.  "closer to -inf" makes the colorful blobs "sharper".
// Positive values make everything weird.
const SAMPLE_DISTALPHA: f32 = -3.0; // -1.9
const COLORSPACE_GAMMA: f32 = 1.8;
const SEED_COLOR_SOURCES: u64 = 4;
const MAX_WEIGHT_COLOR_SOURCE: f32 = 1.5;
const SEED_FUZZINESS: u64 = 1;

const NUM_ARMS_WEIGHTS: [u32; 5] = [1, 2, 2, 1, 1];
const ARM_FACTOR_WEIGHTS: [u8; 5] = [0, 4, 2, 1, 1];
const ARM_RADIUS_MEAN: f32 = 90.0;
const ARM_RADIUS_STDDEV: f32 = 120.0;

const NUM_FRAMES: usize = 90;
const NUM_THREADS: usize = 16;

const SVG_ARM_RADIUS: f32 = 5.0;
const SVG_SOURCE_RADIUS: f32 = 10.0;
const SVG_PERIOD_SECONDS: f32 = 6.0;
const SVG_PATH_WAYPOINTS: usize = 200;

lazy_static! {
    static ref FUZZINESS_DISTRIBUTION: Normal<f32> = Normal::new(0.0, SAMPLENOISE_STDDEV).expect("lolwut");
    static ref NUM_ARMS_DISTRIBUTION: WeightedIndex<u32> = WeightedIndex::new(&NUM_ARMS_WEIGHTS).expect("lolwut");
    static ref ARM_RADIUS_DISTRIBUTION: Normal<f32> = Normal::new(ARM_RADIUS_MEAN, ARM_RADIUS_STDDEV).expect("lolwut");
    static ref PHASE_DISTRIBUTION: Uniform<f32> = Uniform::new(0.0, 2.0 * PI);
    static ref ARM_FACTOR_DISTRIBUTION: WeightedIndex<u8> = WeightedIndex::new(&ARM_FACTOR_WEIGHTS).expect("lolwut");
}

#[derive(Clone, Copy, Debug)]
struct Color {
    // "Percieved fraction", i.e. values are in [0, weight], and can be mixed linearly with each other, at the cost of requiring gamma-correction before being converted to a usable RGB-value.
    r_pf: f32,
    g_pf: f32,
    b_pf: f32,
    weight: f32,
}
impl Color {
    fn sample_for_color_source<T: Rng>(rng: &mut T) -> Color {
        assert!(MAX_WEIGHT_COLOR_SOURCE >= 1.0);
        let weight = rng.gen_range(1.0..=MAX_WEIGHT_COLOR_SOURCE);
        Color {
            r_pf: rng.gen_range(0.0..=weight),
            g_pf: rng.gen_range(0.0..=weight),
            b_pf: rng.gen_range(0.0..=weight),
            weight,
        }
    }
    fn export_channel(&self, percieved_fraction: f32) -> u8 {
        let energy = (percieved_fraction / self.weight).powf(1.0 / COLORSPACE_GAMMA);
        assert!(0.0 <= energy && energy <= 1.0, "{:?}", (percieved_fraction, self.weight, energy));
        (255.0 * energy).round().clamp(0.0, 255.0) as u8
    }
    fn to_rgb(&self) -> [u8; 3] {
        let r = self.export_channel(self.r_pf);
        let g = self.export_channel(self.g_pf);
        let b = self.export_channel(self.b_pf);
        [r, g, b]
    }
    fn times(&self, weight_factor: f32) -> Color {
        Color {
            r_pf: self.r_pf * weight_factor,
            g_pf: self.g_pf * weight_factor,
            b_pf: self.b_pf * weight_factor,
            weight: self.weight * weight_factor,
        }
    }
}
impl Add for Color {
    type Output = Color;
    fn add(self, rhs: Color) -> Self::Output {
        Color {
            r_pf: self.r_pf + rhs.r_pf,
            g_pf: self.g_pf + rhs.g_pf,
            b_pf: self.b_pf + rhs.b_pf,
            weight: self.weight + rhs.weight,
        }
    }
}
impl Sum<Color> for Color {
    /* Why do I still have to implement that?! */
    fn sum<I>(iter: I) -> Self where I: Iterator<Item = Color> {
        iter.reduce(Color::add).expect("was empty?!")
    }
}

#[derive(Debug)]
struct ColorSource {
    x: f32,
    y: f32,
    col: Color,
}
impl ColorSource {
    fn influence_on(&self, x: f32, y: f32) -> Color {
        let dx = self.x - x;
        let dy = self.y - y;
        let dist = (dx * dx + dy * dy).sqrt().max(SAMPLE_MINDIST);
        let weight_factor = dist.powf(SAMPLE_DISTALPHA);
        self.col.times(weight_factor)
    }
}

fn fuzzify<T: Rng>(rng: &mut T, x: u32, y: u32) -> (f32, f32) {
    (
        (x as f32) + rng.sample(*FUZZINESS_DISTRIBUTION),
        (y as f32) + rng.sample(*FUZZINESS_DISTRIBUTION),
    )
}

fn render_sources(sources: &[ColorSource], out_data: &mut Vec<u8>) {
    let mut rng_fuzziness = ChaCha8Rng::seed_from_u64(SEED_FUZZINESS);
    for y_exact in 0..HEIGHT {
        for x_exact in 0..WIDTH {
            let (x, y) = fuzzify(&mut rng_fuzziness, x_exact, y_exact);
            let color = sources.iter().map(|cs| cs.influence_on(x, y)).sum::<Color>();
            out_data.extend(color.to_rgb());
        }
    }
}

#[derive(Debug)]
struct Arm {
    radius: f32,
    factor: i8,
    initial_phase_radians: f32,
}
impl Arm {
    fn new<T: Rng>(rng: &mut T) -> Arm {
        let unsigned_factor = rng.sample(&*ARM_FACTOR_DISTRIBUTION) as i8;
        let factor = if rng.gen::<bool>() { unsigned_factor } else { -unsigned_factor };
        Arm {
            radius: rng.sample(&*ARM_RADIUS_DISTRIBUTION),
            factor,
            initial_phase_radians: rng.sample(&*PHASE_DISTRIBUTION),
        }
    }
    fn instantiate(&self, t_01: f32) -> (f32, f32) {
        let phase = self.initial_phase_radians + t_01 * 2.0 * PI * (self.factor as f32);
        (
            phase.sin() * self.radius,
            phase.cos() * self.radius,
        )
    }
    fn sample_several<T: Rng>(rng: &mut T) -> Vec<Arm> {
        let num_arms = rng.sample(&*NUM_ARMS_DISTRIBUTION);
        (0..num_arms).map(|_| Arm::new(rng)).collect()
    }
}

fn evaluate(xy: (f32, f32), arms: &[Arm], t_01: f32) -> (f32, f32) {
    arms.iter().fold(xy, |xy1, arm| {
        let xy2 = arm.instantiate(t_01);
        (xy1.0 + xy2.0, xy1.1 + xy2.1)
    })
}

#[derive(Debug)]
struct MovingSource {
    x: f32,
    y: f32,
    col: Color,
    arms: Vec<Arm>,
}
impl MovingSource {
    fn new<T: Rng>(rng: &mut T) -> MovingSource {
        MovingSource {
            x: rng.gen_range(AREA_MARGIN as f32..=(WIDTH - AREA_MARGIN) as f32),
            y: rng.gen_range(AREA_MARGIN as f32..=(HEIGHT - AREA_MARGIN) as f32),
            col: Color::sample_for_color_source(rng),
            arms: Arm::sample_several(rng),
        }
    }
    fn instantiate(&self, t_01: f32) -> ColorSource {
        let xy = evaluate((self.x, self.y), &self.arms, t_01);
        ColorSource {
            x: xy.0,
            y: xy.1,
            col: self.col,
        }
    }
}

struct Tracer<'a> {
    source: &'a MovingSource,
    num_arms: usize,
    waypoint: usize,
}
impl<'a> Tracer<'a> {
    fn new(source: &'a MovingSource, num_arms: usize) -> Tracer {
        Tracer {
            source,
            num_arms,
            waypoint: 0,
        }
    }
}
impl<'a> Iterator for Tracer<'a> {
    type Item = (f32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.waypoint > SVG_PATH_WAYPOINTS {
            return None;
        }
        let t_01 = (self.waypoint as f32) / (SVG_PATH_WAYPOINTS as f32);
        self.waypoint += 1;
        Some(evaluate((self.source.x, self.source.y), &self.source.arms[0..self.num_arms], t_01))
    }
}

fn format_line_arm_animate(f: &mut fmt::Formatter<'_>, moving_source: &MovingSource, arms: usize, do_x: bool, point_index: usize) -> Result<(), fmt::Error> {
    writeln!(
        f,
        "<animate attributeName=\"{}{point_index}\" values=\"",
        if do_x { "x" } else {"y"}
    )?;
    let mut is_first = true;
    for trace_point in Tracer::new(moving_source, arms) {
        if is_first {
            is_first = false;
        } else {
            write!(f, ";")?;
        }
        write!(f, "{}", if do_x { trace_point.0 } else { trace_point.1 })?;
    }
    writeln!(
        f,
        "\" dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" />",
    )?;
    Ok(())
}
struct SvgView<'a>(&'a Collection);
impl<'a> fmt::Display for SvgView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let padding = NUM_ARMS_WEIGHTS.len() as u32 * ARM_RADIUS_MEAN as u32;
        writeln!(
            f,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-{} -{} {} {}\">",
            padding,
            padding,
            WIDTH + 2 * padding,
            HEIGHT + 2 * padding,
        )?;
        writeln!(
            f,
            "<rect x=\"-{}\" y=\"-{}\" width=\"{}\" height=\"{}\" fill=\"white\" stroke=\"red\" />",
            padding,
            padding,
            WIDTH + 2 * padding,
            HEIGHT + 2 * padding,
        )?;
        writeln!(
            f,
            "<rect x=\"-1\" y=\"-1\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"black\" />",
            WIDTH + 2,
            HEIGHT + 2,
        )?;
        for (source_id, moving_source) in self.0.sources.iter().enumerate() {
            println!("{:?}", moving_source);
            let rgb_string = {
                let rgb = moving_source.col.to_rgb();
                format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
            };
            writeln!(
                f,
                "<path d=\"M{} {}\" id=\"path_{}_0\" stroke=\"none\" />",
                moving_source.x,
                moving_source.y,
                source_id,
            )?;
            for (arm_id, arm) in moving_source.arms.iter().enumerate() {
                let opacity = (arm_id + 1) as f32 / moving_source.arms.len() as f32;
                // ==== rotate: somehow desyncs?!?!?!
                //writeln!(
                //    f,
                //    "<line x1=\"0\" y1=\"0\" x2=\"{}\" y2=\"0\" stroke=\"green\"><animateTransform attributeName=\"transform\" attributeType=\"XML\" type=\"rotate\" from=\"{}\" to=\"{}\" dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" /><animateMotion dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" calcMode=\"linear\"><mpath href=\"#path_{source_id}_{arm_id}\" /></animateMotion></line>",
                //    arm.radius,
                //    arm.initial_phase_radians * 180.0 / PI + 180.0 +   3.6 * (-arm.factor) as f32,
                //    arm.initial_phase_radians * 180.0 / PI + 180.0 + 363.6 * (-arm.factor) as f32,
                //)?;
                // ==== x1x2y1y2 animate refactor: Terrible, but works.
                writeln!(f, "<line stroke=\"green\">")?;
                format_line_arm_animate(f, moving_source, arm_id, true, 1)?;
                format_line_arm_animate(f, moving_source, arm_id, false, 1)?;
                format_line_arm_animate(f, moving_source, arm_id + 1, true, 2)?;
                format_line_arm_animate(f, moving_source, arm_id + 1, false, 2)?;
                writeln!(
                    f,
                    "\" dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" /></line>",
                )?;


                writeln!(
                    f,
                    "<circle r=\"{SVG_ARM_RADIUS}\" stroke=\"black\" fill=\"black\" fill-opacity=\"{opacity}\" id=\"joint_{source_id}_{arm_id}\"><animateMotion dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" calcMode=\"linear\"><mpath href=\"#path_{source_id}_{arm_id}\" /></animateMotion></circle>",
                )?;
                write!(f, "<path d=\"M")?;
                for trace_point in Tracer::new(moving_source, arm_id + 1) {
                    write!(f, "{} {} ", trace_point.0, trace_point.1)?;
                }
                writeln!(
                    f,
                    "\" id=\"path_{}_{}\" fill=\"none\" stroke=\"{rgb_string}\" stroke-opacity=\"{opacity}\" />",
                    source_id,
                    arm_id + 1,
                )?;
            }
            writeln!(
                f,
                "<circle r=\"{SVG_SOURCE_RADIUS}\" stroke=\"black\" stroke-width=\"2\" fill=\"{rgb_string}\" id=\"joint_{source_id}_{0}\"><animateMotion dur=\"{SVG_PERIOD_SECONDS}s\" repeatCount=\"indefinite\" calcMode=\"linear\"><mpath href=\"#path_{source_id}_{0}\" /></animateMotion></circle>",
                moving_source.arms.len(),
            )?;
        }
        writeln!(f, "</svg>")?;
        Ok(())
    }
}

#[derive(Debug)]
struct Collection {
    n_secs: u64,
    sources: Vec<MovingSource>,
}
impl Collection {
    fn new() -> Collection {
        let n_secs = SystemTime::now().duration_since(UNIX_EPOCH).expect("SystemTime before UNIX EPOCH?!").as_secs();
        let mut sources = Vec::with_capacity(NUM_SOURCES);
        let mut rng = ChaCha8Rng::seed_from_u64(SEED_COLOR_SOURCES);
        for _ in 0..NUM_SOURCES {
            sources.push(MovingSource::new(&mut rng));
        }
        Collection {
            n_secs,
            sources,
        }
    }
    fn render_into(&self, frame: usize, out_data: &mut Vec<u8>) {
        let t_01 = (frame as f32) / (NUM_FRAMES as f32);
        let sources = self.sources.iter().map(|s| s.instantiate(t_01)).collect::<Vec<_>>();
        render_sources(&sources, out_data);
    }
    fn write_svg(&self) {
        let svg_text = format!("{}", SvgView(&self));
        //let filename = format!("build/overview_{}.svg", self.n_secs);
        let filename = "build/overview.svg";
        let path = Path::new(&filename);
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);
        w.write(svg_text.as_bytes()).expect("cannot write?!");
    }
    fn render_all_as_thread(&self, own_thread_id: usize) {
        let mut data = Vec::with_capacity((3 * WIDTH * HEIGHT) as usize);
        for frame in 0..NUM_FRAMES {
            if frame % NUM_THREADS != own_thread_id {
                // TODO: This is a bit inefficient for large values of NUM_THREADS.
                continue;
            }
            let filename = format!("build/image_{}_frame{frame:05}.png", self.n_secs);
            println!("Writing {filename} ...");
            let path = Path::new(&filename);
            let file = File::create(path).unwrap();
            let ref mut w = BufWriter::new(file);
            let mut encoder = png::Encoder::new(w, WIDTH, HEIGHT);
            encoder.set_color(png::ColorType::Rgb);
            let mut writer = encoder.write_header().unwrap();
            self.render_into(frame, &mut data);
            writer.write_image_data(&data).unwrap();
            println!("Finished {filename} .");
            data.clear(); // Retains capacity
        }
    }
}

fn render_all(collection: Arc<Collection>) {
    assert!(0 < NUM_THREADS);
    assert!(NUM_THREADS <= 128, "Are you insane?!");
    let handles = (0..NUM_THREADS)
        .map(|id| {
            let thread_collection = collection.clone();
            thread::spawn(move || thread_collection.render_all_as_thread(id))
        })
        .collect::<Vec<_>>();
    handles.into_iter().map(|handle| handle.join()).count(); // TODO: Nicer consume()?
}

fn main() {
    let collection = Arc::new(Collection::new());
    collection.write_svg();
    if false {
        render_all(collection);
    }
}
