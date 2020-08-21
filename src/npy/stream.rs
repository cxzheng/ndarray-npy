use super::{
    error::{WriteDataError, WriteNpyError},
    header::Header,
    WritableElement,
};
use ndarray::Dimension;
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
/// let stream = NpyOutStreamBuilder::<f32>::new("out.npy").for_arr2((2, 2)).f().build()?;
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
                WriteDataError::TooManyElement(self.tot_elems, self.written_elems + slice.len())
                    .into(),
            )
        } else {
            T::write_slice(slice, &mut self.writer)?;
            self.written_elems += slice.len();
            Ok(self.written_elems)
        }
    }
}

impl<T: WritableElement> NpyOutStreamBuilder<T> {
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

    pub fn set_dim<'a, D: Dimension>(&'a mut self, dim: D) -> &'a mut NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.extend_from_slice(dim.slice());
        self
    }

    pub fn for_arr1<'a>(&'a mut self, len: usize) -> &'a mut NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.push(len);
        self
    }

    pub fn for_arr2<'a>(&'a mut self, dim: [usize; 2]) -> &'a mut NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.extend_from_slice(&dim);
        self
    }

    pub fn for_arr3<'a>(&'a mut self, dim: [usize; 3]) -> &'a mut NpyOutStreamBuilder<T> {
        self.header.shape.clear();
        self.header.shape.extend_from_slice(&dim);
        self
    }

    pub fn f<'a>(&'a mut self) -> &'a mut NpyOutStreamBuilder<T> {
        self.header.fortran_order = true;
        self
    }

    pub fn c<'a>(&'a mut self) -> &'a mut NpyOutStreamBuilder<T> {
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
            _marker: marker::PhantomData,
        })
    }
}
