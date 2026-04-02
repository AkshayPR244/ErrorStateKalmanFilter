use image::GrayImage;
use anyhow::{Context as _, Result};

/// A 2D pixel coordinate detected as a corner.
///
/// RUST: `u32` for pixel coordinates — image dimensions are always non-negative
///       so a signed type would be wasteful and semantically wrong.
///       In C++ you'd use `cv::Point2i` or `std::pair<int,int>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
    /// FAST corner response score — sum of absolute pixel differences from centre.
    /// Higher = stronger corner.
    pub score: u32,
}

/// Which direction a Bresenham circle pixel is relative to the centre.
///
/// RUST: `enum` is a *sum type* — a value is exactly ONE of these variants,
///       with zero runtime cost for this simple tag-only enum (stored as u8).
///       Unlike C's enum (just named integers), Rust's enum variants are
///       distinct types. The compiler forces you to handle every variant in
///       a `match` — there's no "fall through an unhandled case" bug.
///       C++ equivalent: `enum class Brightness { Brighter, Darker, Similar };`
#[derive(Debug, Clone, Copy, PartialEq)]
enum Brightness {
    Brighter,   // pixel > centre + threshold
    Darker,     // pixel < centre - threshold
    Similar,    // within threshold — neither
}

/// The 16 pixel offsets for a Bresenham circle of radius 3 (FAST-16).
///
/// RUST: `const` items are evaluated at compile time and inlined at every use
///       site — same as `constexpr` in C++.
///       `[(i32, i32); 16]` is a fixed-size *array* living on the stack.
///       The `;16` part is the length — part of the TYPE, not just a comment.
///       `Vec<(i32, i32)>` would be the heap-allocated, growable version.
const CIRCLE_OFFSETS: [(i32, i32); 16] = [
    ( 0, -3), ( 1, -3), ( 2, -2), ( 3, -1),
    ( 3,  0), ( 3,  1), ( 2,  2), ( 1,  3),
    ( 0,  3), (-1,  3), (-2,  2), (-3,  1),
    (-3,  0), (-3, -1), (-2, -2), (-1, -3),
];

/// Classify one Bresenham-circle pixel against the centre intensity.
///
/// RUST: standalone `fn` — not inside an `impl` block because it belongs to
///       no particular type. Same as a C++ file-scope static helper.
///
/// Note `centre.saturating_add(threshold)`:
///   Plain `centre + threshold` on u8 would *panic* in debug builds if it
///   overflows 255 (Rust checks integer overflow in debug mode by default).
///   `saturating_add` clamps to 255 instead of wrapping or panicking —
///   same intent as `std::clamp` in C++.
fn classify(centre: u8, pixel: u8, threshold: u8) -> Brightness {
    if pixel > centre.saturating_add(threshold) {
        Brightness::Brighter
    } else if pixel < centre.saturating_sub(threshold) {
        Brightness::Darker
    } else {
        Brightness::Similar
    }
}

/// Test a single pixel at (cx, cy) for FAST cornerness.
/// Returns Some(score) if it is a corner, None otherwise.
///
/// RUST: `Option<u32>` — Rust's null-safety mechanism. An enum with two variants:
///   `Some(value)` — there IS a result
///   `None`        — there is no result
///
///   You cannot accidentally use None as a number; the type system prevents it.
///   The caller unwraps it with `if let Some(s) = test_pixel(...)`.
///   C++ equivalent: `std::optional<uint32_t>` (C++17).
fn test_pixel(img: &GrayImage, cx: u32, cy: u32, threshold: u8, n: usize) -> Option<u32> {
    let centre = img.get_pixel(cx, cy)[0];

    // RUST: `Vec<T>` — heap-allocated growable array, like `std::vector<T>`.
    //       `with_capacity(16)` pre-allocates space for 16 items to avoid
    //       reallocation as we push — same as `vector.reserve(16)` in C++.
    let mut classes: Vec<Brightness> = Vec::with_capacity(16);

    for &(dx, dy) in &CIRCLE_OFFSETS {
        let px = (cx as i32 + dx) as u32;
        let py = (cy as i32 + dy) as u32;
        classes.push(classify(centre, img.get_pixel(px, py)[0], threshold));
    }

    // ── High-speed pre-test (YOUR TURN #1) ────────────────────────────────
    // Check the 4 compass pixels: indices 0 (N), 4 (E), 8 (S), 12 (W).
    // If fewer than 3 are Brighter AND fewer than 3 are Darker, return None.
    //
    // Hint: build an array `compass = [classes[0], classes[4], ...]`
    //       then count with `.iter().filter(|&&b| b == Brightness::Brighter).count()`
    //
    // High-speed pre-test
    let compass = [classes[0], classes[4], classes[8], classes[12]];
    let brighter_count = compass.iter().filter(|&&b| b == Brightness::Brighter).count();
    let darker_count = compass.iter().filter(|&&b| b == Brightness::Darker).count();
    if brighter_count < 3 && darker_count < 3 {
        return None;
    }
    // ── Full arc test ──────────────────────────────────────
    // Double the circle to handle wrap-around: [0..15, 0..15] (32 elements).
    // Then check whether any window of length `n` is ALL Brighter or ALL Darker.
    //
    // Hint 1: `classes.iter().chain(classes.iter()).copied().collect::<Vec<_>>()`
    //         gives you the doubled circle.
    // Hint 2: `doubled.windows(n).any(|w| w.iter().all(|&b| b == target))`
    //         checks for n consecutive matching elements.
    //
    // If neither all-Brighter nor all-Darker arc exists, return None.
    
    let doubled = classes.iter().chain(classes.iter()).copied().collect::<Vec<_>>(); 
    let has_brighter_arc = doubled.windows(n).any(|w| w.iter().all(|&b| b == Brightness::Brighter)); 
    let has_darker_arc = doubled.windows(n).any(|w| w.iter().all(|&b| b == Brightness::Darker));
    if !has_brighter_arc && !has_darker_arc {
        return None;
    }

    // ── Score ──────────────────────────────────────────────
    // Score = sum of |pixel - centre| for all 16 ring pixels.
    //
    // Iterate over CIRCLE_OFFSETS, look up each pixel, compute
    //       `(img.get_pixel(px,py)[0] as i32 - centre as i32).unsigned_abs()`
    //       then `.sum()` the iterator.
    //
    let score: u32 = CIRCLE_OFFSETS.iter().map(|&(dx, dy)| {
        let px = (cx as i32 + dx) as u32;
        let py = (cy as i32 + dy) as u32;
        (img.get_pixel(px, py)[0] as i32 - centre as i32).unsigned_abs()
    }).sum();

    Some(score)
}

/// Non-maximum suppression: remove weaker corners within `radius` pixels
/// of a stronger one.
///
/// RUST: `&mut Vec<KeyPoint>` — mutable borrow. We sort and filter the
///       caller's Vec in-place; no copy of the whole collection is made.
///       `.retain(|kp| ...)` keeps only elements where the closure returns
///       true — equivalent to `std::erase_if` in C++23.
fn non_max_suppression(keypoints: &mut Vec<KeyPoint>, radius: u32) {
    // Sort descending by score so we always keep the strongest corner.
    keypoints.sort_unstable_by(|a, b| b.score.cmp(&a.score));

    let r2 = (radius * radius) as i64;
    let mut keep = vec![true; keypoints.len()];

    for i in 0..keypoints.len() {
        if !keep[i] { continue; }
        for j in (i + 1)..keypoints.len() {
            if !keep[j] { continue; }
            let dx = keypoints[i].x as i64 - keypoints[j].x as i64;
            let dy = keypoints[i].y as i64 - keypoints[j].y as i64;
            if dx * dx + dy * dy <= r2 {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    keypoints.retain(|_| { let ok = keep[idx]; idx += 1; ok });
}

/// Parameters for the FAST detector.
pub struct FastParams {
    /// Pixel intensity threshold for brighter/darker classification.
    pub threshold: u8,
    /// Minimum consecutive arc length to declare a corner (9 = FAST-9).
    pub arc_length: usize,
    /// Suppress corners within this many pixels of a stronger one.
    pub nms_radius: u32,
    /// Maximum features to return per frame.
    pub max_features: usize,
}

impl FastParams {
    pub fn default() -> Self {
        Self { threshold: 20, arc_length: 9, nms_radius: 5, max_features: 500 }
    }
}

/// Detect FAST corners in a grayscale image.
/// Returns keypoints sorted by score (strongest first), capped at max_features.
pub fn detect_fast(img: &GrayImage, params: &FastParams) -> Vec<KeyPoint> {
    let (w, h) = img.dimensions();
    let margin = 3u32;

    // RUST: `Vec::new()` — empty Vec, no allocation yet. Equivalent to
    //       `std::vector<KeyPoint> keypoints;` in C++.
    let mut keypoints: Vec<KeyPoint> = Vec::new();

    // RUST: `margin..(h - margin)` is an exclusive-end range — same semantics
    //       as `for (uint32_t y = margin; y < h - margin; y++)` in C++.
    for y in margin..(h - margin) {
        for x in margin..(w - margin) {
            // `if let Some(score)` unpacks the Option — only enters the block
            // when test_pixel returned Some(score), ignores None silently.
            if let Some(score) = test_pixel(img, x, y, params.threshold, params.arc_length) {
                keypoints.push(KeyPoint { x, y, score });
            }
        }
    }

    non_max_suppression(&mut keypoints, params.nms_radius);
    keypoints.truncate(params.max_features);
    keypoints
}

/// Load a KITTI camera frame as a grayscale image.
pub fn load_frame(path: &std::path::Path) -> Result<GrayImage> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?;
    Ok(img.to_luma8())
}
