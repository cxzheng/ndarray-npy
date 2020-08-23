use super::{
    error::{WriteDataError, WriteNpyError},
    header::Header,
    WritableElement,
};
use ndarray::{Dimension, IntoDimension};
use std::{
    fs::File,
    marker,
    path::{Path, PathBuf},
};

/// This define a stream that allows progressively output a stream of array data
/// into a `.npy` file.
pub struct NpyOutStream<T: WritableElement> {
    tot_elems: usize,     // total number of elements to output
    written_elems: usize, // how many elements have been written
    writer: File,
    closed: bool,
    _marker: marker::PhantomData<T>,
}

/// This is the builder for creating an output stream that write a NPY array into
/// a file.
///
/// The builder is created from specifying the file name using [`new`](#method.from_path).
///
/// # Example
///
/// ```no_run
/// use ndarray_npy::NpyOutStreamBuilder;
/// # use ndarray_npy::WriteNpyError;
///
/// let mut stream = NpyOutStreamBuilder::<f32>::new("out.npy").for_arr2([2, 2]).build()?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub struct NpyOutStreamBuilder<T: WritableElement> {
    path: PathBuf,
    header: Header,
    _marker: marker::PhantomData<T>,
}

impl<T: WritableElement> NpyOutStream<T> {
    /// Incrementally output to the stream a slice of data.
    ///
    /// An error will be raised if the total number of array elements that are put into the stream
    /// exceeds the total number of elements defined by the array shape.
    pub fn write_slice(&mut self, slice: &[T]) -> Result<usize, WriteNpyError> {
        if self.written_elems + slice.len() > self.tot_elems {
            Err(
                WriteDataError::TooManyElements(self.tot_elems, self.written_elems + slice.len())
                    .into(),
            )
        } else {
            T::write_slice(slice, &mut self.writer)?;
            self.written_elems += slice.len();
            Ok(self.written_elems)
        }
    }

    /// Return the total number of elements expected to be put into the stream.
    #[inline(always)]
    pub fn tot_elems(&self) -> usize {
        self.tot_elems
    }

    /// Check if all the expected elements have been written into the stream.
    #[inline(always)]
    pub fn finished(&self) -> bool {
        self.tot_elems == self.written_elems
    }

    pub fn close(mut self) -> Result<(), WriteDataError> {
        self.closed = true;

        if self.written_elems < self.tot_elems  {
            Err(WriteDataError::TooFewElements(self.tot_elems(), self.written_elems))
        } else {
            Ok(())
        }
    }
}

impl<T: WritableElement> Drop for NpyOutStream<T> {
    fn drop(&mut self) {
        if !self.closed && !self.finished() {
            eprintln!("WARNING: The NpyOutStream is closed without receiving all elements: expect {} elements, received {} elements",
                      self.tot_elems(), self.written_elems);
        }
    }
}

impl<T: WritableElement> NpyOutStreamBuilder<T> {
    /// Start to build an output stream to the given file.
    pub fn new<P: AsRef<Path>>(path: P) -> NpyOutStreamBuilder<T> {
        NpyOutStreamBuilder {
            path: path.as_ref().to_path_buf(),
            header: Header {
                type_descriptor: T::type_descriptor(),
                fortran_order: false,
                shape: Vec::with_capacity(3),
            },
            _marker: marker::PhantomData,
        }
    }

    pub fn for_dim<D: IntoDimension>(mut self, dim: D) -> NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header
            .shape
            .extend_from_slice(dim.into_dimension().slice());
        self
    }

    /// Set the output dimentsion as a 1D array of the given size.
    pub fn for_arr1(mut self, len: usize) -> NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.push(len);
        self
    }

    /// Set the output dimentsion as a 2D array of the given size.
    pub fn for_arr2(mut self, dim: [usize; 2]) -> NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.extend_from_slice(&dim);
        self
    }

    pub fn for_arr3(mut self, dim: [usize; 3]) -> NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.extend_from_slice(&dim);
        self
    }

    /// Set to store the array in Fortran order (column major).
    pub fn f(mut self) -> NpyOutStreamBuilder<T> {
        self.header.fortran_order = true;
        self
    }

    pub fn c(mut self) -> NpyOutStreamBuilder<T> {
        self.header.fortran_order = false;
        self
    }

    pub fn build(self) -> Result<NpyOutStream<T>, WriteNpyError> {
        let mut writer = File::create(self.path)?;
        self.header.write(&mut writer)?;

        let tot_elems = self.header.shape.iter().fold(1, |s, &a| s * a);
        Ok(NpyOutStream {
            tot_elems,
            written_elems: 0,
            writer,
            closed: false,
            _marker: marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod test {
    use super::NpyOutStreamBuilder;

    #[test]
    fn test_2x3() {
        let mut stream = NpyOutStreamBuilder::<f32>::new("out.npy")
            .for_dim((2, 3))
            .build()
            .unwrap();
        assert_eq!(stream.tot_elems(), 6);
        let vec1 = vec![1., 2.];
        let vec2 = vec![3., 4., 5., 6.];
        stream.write_slice(&vec1).unwrap();
        stream.write_slice(&vec2).unwrap();
    }

    #[test]
    fn test_100x3() {
        let mut stream = NpyOutStreamBuilder::<f32>::new("out2.npy")
            .for_dim((100, 3))
            .f()
            .build()
            .unwrap();
        assert_eq!(stream.tot_elems(), 300);
        let mut vec = vec![0.; 100];
        for i in 0..3 {
            for v in &mut vec {
                *v = (i + 1) as f32;
            }
            stream.write_slice(&vec).unwrap();
        }
        assert!(stream.finished());
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let mut stream = NpyOutStreamBuilder::<f32>::new("out.npy")
            .for_dim((2, 3))
            .build()
            .unwrap();
        assert_eq!(stream.tot_elems(), 6);
        let vec1 = vec![1., 2.];
        let vec2 = vec![3., 4., 5.];
        stream.write_slice(&vec1).unwrap();
        stream.write_slice(&vec2).unwrap();
        stream.close().unwrap();
    }
}
