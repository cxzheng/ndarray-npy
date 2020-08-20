use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion}; //, black_box};
use ndarray_npy::WriteNpyExt;
use std::fs::File;

fn write_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_vec");
    let sizes = [500_usize, 1000, 2000, 4000, 8000];
    let mut vecs = Vec::with_capacity(sizes.len());
    for s in &sizes {
        let mut arr = vec![0_f64; *s];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as f64;
        }
        vecs.push(arr);
    }

    for i in 0..sizes.len() {
        group.bench_with_input(BenchmarkId::from_parameter(sizes[i]), &i, |b, &i| {
            b.iter(|| {
                //black_box(File::create("vec.npy").unwrap());
                let writer = File::create("vec.npy").unwrap();
                vecs[i].write_npy(writer).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, write_vec);
criterion_main!(benches);
