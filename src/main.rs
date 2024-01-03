#[macro_use]
extern crate lazy_static;
extern crate png;
extern crate rand;

use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::iter::Sum;
use std::ops::Add;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{SeedableRng, Rng};
use rand_distr::{Normal, Uniform, WeightedIndex};
use rand_chacha::ChaCha8Rng;

// Heavily inspired by https://github.com/BenWiederhake/noisycols/blob/master/generate.py

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const AREA_MARGIN: u32 = 180;
const NUM_SOURCES: usize = 13;
const SAMPLENOISE_STDDEV: f32 = 40.0;
const SAMPLE_MINDIST: f32 = 1e-10;
// Should be between -inf and 0.  "closer to -inf" makes the colorful blobs "sharper".
// Positive values make everything weird.
const SAMPLE_DISTALPHA: f32 = -3.0; // -1.9
const COLORSPACE_GAMMA: f32 = 1.8;
const SEED_COLOR_SOURCES: u64 = 3;
const MAX_WEIGHT_COLOR_SOURCE: f32 = 1.5;
const SEED_FUZZINESS: u64 = 1;

const NUM_ARMS_WEIGHTS: [u32; 5] = [1, 2, 2, 1, 1];
const ARM_FACTOR_WEIGHTS: [u8; 5] = [0, 4, 2, 1, 1];
const ARM_RADIUS_MEAN: f32 = 130.0;
const ARM_RADIUS_STDDEV: f32 = 160.0;

const NUM_FRAMES: usize = 120;
const NUM_THREADS: usize = 16;

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
        let xy = self.arms.iter().fold((self.x, self.y), |xy1, arm| {
            let xy2 = arm.instantiate(t_01);
            (xy1.0 + xy2.0, xy1.1 + xy2.1)
        });
        ColorSource {
            x: xy.0,
            y: xy.1,
            col: self.col,
        }
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
    render_all(collection);
}
