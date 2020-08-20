use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion}; //, black_box};
use std::ptr::copy_nonoverlapping;

// benchmarking memory copy performance
fn mem_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memcopy");
    let sizes = [500_usize, 1000, 2000, 4000, 8000, 16000];
    let mut vecs = Vec::with_capacity(sizes.len());
    for s in &sizes {
        let mut arr = vec![0_f64; *s];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as f64;
        }
        vecs.push(arr);
    }
    let mut dest = vec![0_f64; *sizes.last().unwrap()];

    for i in 0..sizes.len() {
        group.bench_with_input(BenchmarkId::from_parameter(sizes[i]), &i, |b, &i| {
            b.iter(|| {
                unsafe {
                    *vecs[i].get_unchecked_mut(0) += i as f64;
                    copy_nonoverlapping(vecs[i].as_ptr(), dest.as_mut_ptr(), sizes[i]);
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, mem_copy);
criterion_main!(benches);
