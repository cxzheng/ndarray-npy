use ndarray_npy::NpyOutStreamBuilder;
use std::time::{Duration, Instant};

fn mean(numbers: &Vec<Duration>) -> Duration {
    let sum: Duration = numbers.iter().sum();
    sum / (numbers.len() as u32)
}

fn median(numbers: &mut Vec<Duration>) -> Duration {
    numbers.sort();

    let mid = numbers.len() / 2;
    if numbers.len() % 2 == 0 {
        mean(&vec![numbers[mid - 1], numbers[mid]])
    } else {
        numbers[mid]
    }
}

#[test]
#[ignore]
fn stream_perf() {
    const LEN: usize = 8000;
    const N: usize = 1000;
    let vec: Vec<f64> = (1..(LEN + 1)).map(|x| x as f64).collect(); //collect();

    let mut stream = NpyOutStreamBuilder::<f64>::new("bench.npy")
        .for_dim((N, LEN))
        .build()
        .unwrap();

    let mut ts: Vec<Duration> = Vec::with_capacity(N);

    let mut tot_time = Duration::new(0, 0);
    for _ in 0..1000 {
        let start = Instant::now();
        stream.write_slice(&vec).unwrap();
        let duration = start.elapsed();
        eprintln!("{}", duration.as_micros());
        tot_time += duration;
        ts.push(duration);
    }

    println!("time: {:?}", tot_time / (N as u32));
    println!("Mean: {:?}", mean(&ts));
    println!("Median: {:?} of {} samples", median(&mut ts), ts.len());
    println!("Min: {:?}, Max: {:?}", ts[0], ts.last().unwrap());
}
