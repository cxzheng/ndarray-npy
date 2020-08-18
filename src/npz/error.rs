use crate::{ReadNpyError, WriteNpyError};
use thiserror::Error;
use zip::result::ZipError;

/// An error writing a `.npz` file.
#[derive(Error, Debug)]
pub enum WriteNpzError {
    /// An error caused by the zip file.
    #[error("zip file error")]
    Zip(#[from] ZipError),

    /// An error caused by writing an inner `.npy` file.
    #[error("cannot write npy file to npz archive")]
    Npy(#[from] WriteNpyError),
}

/// An error reading a `.npz` file.
#[derive(Error, Debug)]
pub enum ReadNpzError {
    /// An error caused by the zip archive.
    #[error("zip file error")]
    Zip(#[from] ZipError),

    /// An error caused by reading an inner `.npy` file.
    #[error("cannot read npy file in npz archive")]
    Npy(#[from] ReadNpyError),
}
