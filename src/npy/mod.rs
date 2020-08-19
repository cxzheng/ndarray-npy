mod error;
pub mod header;
pub use error::*;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use header::Header;
use ndarray::prelude::*;
use ndarray::{Data, DataOwned, IntoDimension};
use py_literal::Value as PyValue;
use std::io;
use std::mem;

/// Read an `.npy` file located at the specified path.
///
/// This is a convience function for using `File::open` followed by
/// [`ReadNpyExt::read_npy`](trait.ReadNpyExt.html#tymethod.read_npy).
///
/// # Example
///
/// ```no_run
/// use ndarray::Array2;
/// use ndarray_npy::read_npy;
/// # use ndarray_npy::ReadNpyError;
///
/// let arr: Array2<i32> = read_npy("array.npy")?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub fn read_npy<P, T>(path: P) -> Result<T, ReadNpyError>
where
    P: AsRef<std::path::Path>,
    T: ReadNpyExt,
{
    T::read_npy(std::fs::File::open(path)?)
}

/// Writes an array to an `.npy` file at the specified path.
///
/// This function will create the file if it does not exist, or overwrite it if
/// it does.
///
/// This is a convenience function for using `File::create` followed by
/// [`WriteNpyExt::write_npy`](trait.WriteNpyExt.html#tymethod.write_npy).
///
/// # Example
///
/// ```no_run
/// use ndarray::array;
/// use ndarray_npy::write_npy;
/// # use ndarray_npy::WriteNpyError;
///
/// let arr = array![[1, 2, 3], [4, 5, 6]];
/// write_npy("array.npy", arr)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub fn write_npy<P, T>(path: P, array: T) -> Result<(), WriteNpyError>
where
    P: AsRef<std::path::Path>,
    T: WriteNpyExt,
{
    array.write_npy(std::fs::File::create(path)?)
}

/// An array element type that can be written to an `.npy` or `.npz` file.
pub unsafe trait WritableElement: Sized {
    /// Returns a descriptor of the type that can be used in the header.
    fn type_descriptor() -> PyValue;

    /// Writes a single instance of `Self` to the writer.
    fn write<W: io::Write>(&self, writer: W) -> Result<(), WriteDataError>;

    /// Writes a slice of `Self` to the writer.
    fn write_slice<W: io::Write>(slice: &[Self], writer: W) -> Result<(), WriteDataError>;
}

/// Extension trait for writing `ArrayBase` to `.npy` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::{array, Array2};
/// use ndarray_npy::WriteNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::WriteNpyError;
///
/// let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let writer = File::create("array.npy")?;
/// arr.write_npy(writer)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub trait WriteNpyExt {
    /// Writes the array to `writer` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.save`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html).
    fn write_npy<W: io::Write>(&self, writer: W) -> Result<(), WriteNpyError>;
}

impl<A, S, D> WriteNpyExt for ArrayBase<S, D>
where
    A: WritableElement,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn write_npy<W: io::Write>(&self, mut writer: W) -> Result<(), WriteNpyError> {
        let write_contiguous = |mut writer: W, fortran_order: bool| {
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?; // write header
            A::write_slice(self.as_slice_memory_order().unwrap(), &mut writer)?;
            Ok(())
        };
        if self.is_standard_layout() {
            // contiguous “C order”
            write_contiguous(writer, false)
        } else if self.view().reversed_axes().is_standard_layout() {
            // contiguous "F order"
            write_contiguous(writer, true)
        } else {
            // e.g., has custom strides
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order: false,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?;
            for elem in self.iter() {
                elem.write(&mut writer)?;
            }
            Ok(())
        }
    }
}

/// Write a slice of primitive types to `.npy` files.
///
/// # Example
///
/// ```no_run
/// use ndarray_npy::WriteNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::WriteNpyError;
///
/// let arr: Vec<f64> = vec![1., 2., 3., 4., 5., 6.];
/// let writer = File::create("vec.npy")?;
/// arr.write_npy(writer)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
impl<A> WriteNpyExt for [A]
where
    A: WritableElement,
{
    fn write_npy<W: io::Write>(&self, mut writer: W) -> Result<(), WriteNpyError> {
        Header {
            type_descriptor: A::type_descriptor(),
            fortran_order: false,
            shape: vec![self.len()],
        }
        .write(&mut writer)?;
        A::write_slice(&self, &mut writer)?;
        Ok(())
    }
}

/// An array element type that can be read from an `.npy` or `.npz` file.
pub trait ReadableElement: Sized {
    /// Reads to the end of the `reader`, creating a `Vec` of length `len`.
    ///
    /// This method should return `Err(_)` in at least the following cases:
    ///
    /// * if the `type_desc` does not match `Self`
    /// * if the `reader` has fewer elements than `len`
    /// * if the `reader` has extra bytes after reading `len` elements
    fn read_to_end_exact_vec<R: io::Read>(
        reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError>;
}

/// Extension trait for reading `Array` from `.npy` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::Array2;
/// use ndarray_npy::ReadNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::ReadNpyError;
///
/// let reader = File::open("array.npy")?;
/// let arr = Array2::<i32>::read_npy(reader)?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub trait ReadNpyExt: Sized {
    /// Reads the array from `reader` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.load`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)
    /// for `.npy` files.
    fn read_npy<R: io::Read>(reader: R) -> Result<Self, ReadNpyError>;
}

impl<A, S, D> ReadNpyExt for ArrayBase<S, D>
where
    A: ReadableElement,
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    fn read_npy<R: io::Read>(mut reader: R) -> Result<Self, ReadNpyError> {
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = match shape.size_checked() {
            Some(len) if len <= std::isize::MAX as usize => len,
            _ => return Err(ReadNpyError::LengthOverflow),
        };
        let data = A::read_to_end_exact_vec(&mut reader, &header.type_descriptor, len)?;
        ArrayBase::from_shape_vec(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ReadNpyError::WrongNdim(D::NDIM, ndim))
    }
}

macro_rules! impl_writable_primitive {
    ($elem:ty, $little_desc:expr, $big_desc:expr) => {
        unsafe impl WritableElement for $elem {
            fn type_descriptor() -> PyValue {
                if cfg!(target_endian = "little") {
                    PyValue::String($little_desc.into())
                } else if cfg!(target_endian = "big") {
                    PyValue::String($big_desc.into())
                } else {
                    unreachable!()
                }
            }

            fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteDataError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(self_: &$elem) -> &[u8] {
                    unsafe {
                        let ptr: *const $elem = self_;
                        std::slice::from_raw_parts(ptr.cast::<u8>(), mem::size_of::<$elem>())
                    }
                }
                writer.write_all(cast(self))?;
                Ok(())
            }

            fn write_slice<W: io::Write>(
                slice: &[Self],
                mut writer: W,
            ) -> Result<(), WriteDataError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(slice: &[$elem]) -> &[u8] {
                    unsafe {
                        std::slice::from_raw_parts(
                            slice.as_ptr().cast::<u8>(),
                            slice.len() * mem::size_of::<$elem>(),
                        )
                    }
                }
                writer.write_all(cast(slice))?;
                Ok(())
            }
        }
    };
}

/// Returns `Ok(_)` iff the `reader` had no more bytes on entry to this
/// function.
///
/// **Warning** This will consume the remainder of the reader.
pub fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadDataError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadDataError::ExtraBytes(num_extra_bytes))
    }
}

macro_rules! impl_readable_primitive_one_byte {
    ($elem:ty, [$($desc:expr),*], $zero:expr, $read_into:ident) => {
        impl ReadableElement for $elem {
            fn read_to_end_exact_vec<R: io::Read>(
                mut reader: R,
                type_desc: &PyValue,
                len: usize,
            ) -> Result<Vec<Self>, ReadDataError> {
                match *type_desc {
                    PyValue::String(ref s) if $(s == $desc)||* => {
                        let mut out = vec![$zero; len];
                        reader.$read_into(&mut out)?;
                        check_for_extra_bytes(&mut reader)?;
                        Ok(out)
                    }
                    ref other => Err(ReadDataError::WrongDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_primitive_one_byte {
    ($elem:ty, $write_desc:expr, [$($read_desc:expr),*], $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $write_desc, $write_desc);
        impl_readable_primitive_one_byte!($elem, [$($read_desc),*], $zero, $read_into);
    };
}

impl_primitive_one_byte!(i8, "|i1", ["|i1", "i1", "b"], 0, read_i8_into);
impl_primitive_one_byte!(u8, "|u1", ["|u1", "u1", "B"], 0, read_exact);

macro_rules! impl_readable_primitive_multi_byte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl ReadableElement for $elem {
            fn read_to_end_exact_vec<R: io::Read>(
                mut reader: R,
                type_desc: &PyValue,
                len: usize,
            ) -> Result<Vec<Self>, ReadDataError> {
                let mut out = vec![$zero; len];
                match *type_desc {
                    PyValue::String(ref s) if s == $little_desc => {
                        reader.$read_into::<LittleEndian>(&mut out)?;
                    }
                    PyValue::String(ref s) if s == $big_desc => {
                        reader.$read_into::<BigEndian>(&mut out)?;
                    }
                    ref other => {
                        return Err(ReadDataError::WrongDescriptor(other.clone()));
                    }
                }
                check_for_extra_bytes(&mut reader)?;
                Ok(out)
            }
        }
    };
}

macro_rules! impl_primitive_multi_byte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $little_desc, $big_desc);
        impl_readable_primitive_multi_byte!($elem, $little_desc, $big_desc, $zero, $read_into);
    };
}

impl_primitive_multi_byte!(i16, "<i2", ">i2", 0, read_i16_into);
impl_primitive_multi_byte!(i32, "<i4", ">i4", 0, read_i32_into);
impl_primitive_multi_byte!(i64, "<i8", ">i8", 0, read_i64_into);

impl_primitive_multi_byte!(u16, "<u2", ">u2", 0, read_u16_into);
impl_primitive_multi_byte!(u32, "<u4", ">u4", 0, read_u32_into);
impl_primitive_multi_byte!(u64, "<u8", ">u8", 0, read_u64_into);

impl_primitive_multi_byte!(f32, "<f4", ">f4", 0., read_f32_into);
impl_primitive_multi_byte!(f64, "<f8", ">f8", 0., read_f64_into);

impl ReadableElement for bool {
    fn read_to_end_exact_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                // Read the data.
                let mut bytes: Vec<u8> = vec![0; len];
                reader.read_exact(&mut bytes)?;
                check_for_extra_bytes(&mut reader)?;

                // Check that all the data is valid, because creating a `bool`
                // with an invalid value is undefined behavior. Rust guarantees
                // that `false` is represented as `0x00` and `true` is
                // represented as `0x01`.
                for &byte in &bytes {
                    if byte > 1 {
                        return Err(ReadDataError::ParseBoolError(byte));
                    }
                }

                // Cast the `Vec<u8>` to `Vec<bool>`.
                {
                    let ptr: *mut u8 = bytes.as_mut_ptr();
                    let len: usize = bytes.len();
                    let cap: usize = bytes.capacity();
                    mem::forget(bytes);
                    // This is safe because:
                    //
                    // * All elements are valid `bool`s. (See the loop above.)
                    //
                    // * `ptr` was originally allocated by `Vec`.
                    //
                    // * `bool` has the same size and alignment as `u8`.
                    //
                    // * `len` and `cap` are copied directly from the
                    //   `Vec<u8>`, so `len <= cap` and `cap` is the capacity
                    //   `ptr` was allocated with.
                    Ok(unsafe { Vec::from_raw_parts(ptr.cast::<bool>(), len, cap) })
                }
            }
            ref other => Err(ReadDataError::WrongDescriptor(other.clone())),
        }
    }
}

// Rust guarantees that `bool` is one byte, the bitwise representation of
// `false` is `0x00`, and the bitwise representation of `true` is `0x01`, so we
// can just cast the data in-place.
impl_writable_primitive!(bool, "|b1", "|b1");

#[cfg(test)]
mod test {
    use super::ReadableElement;
    use py_literal::Value as PyValue;
    use std::io::Cursor;

    #[test]
    fn read_bool() {
        let data = &[0x00, 0x01, 0x00, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::read_to_end_exact_vec(Cursor::new(data), &type_desc, data.len()).unwrap();
        assert_eq!(out, vec![false, true, false, false, true]);
    }

    #[test]
    fn read_bool_bad_value() {
        let data = &[0x00, 0x01, 0x05, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        match <bool>::read_to_end_exact_vec(Cursor::new(data), &type_desc, data.len()) {
            Err(err) => {
                assert_eq!(format!("{}", err), "cannot parse value 0x05 as a bool");
            }
            _ => panic!(),
        }
    }
}
