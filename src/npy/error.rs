use py_literal::{
    FormatError as PyValueFormatError, ParseError as PyValueParseError, Value as PyValue,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseHeaderError {
    #[error("start does not match magic string")]
    MagicString,

    #[error("unknown version number: {major}.{minor}")]
    Version { major: u8, minor: u8 },

    /// Indicates that the `HEADER_LEN` doesn't fit in `usize`.
    #[error("HEADER_LEN {0} does not fit in `usize`")]
    HeaderLengthOverflow(u32),

    /// Indicates that the array format string contains non-ASCII characters.
    /// This is an error for .npy format versions 1.0 and 2.0.
    #[error("non-ascii in array format string; this is not supported in .npy format versions 1.0 and 2.0")]
    NonAscii,

    /// Error parsing the array format string as UTF-8. This does not apply to
    /// .npy format versions 1.0 and 2.0, which require the array format string
    /// to be ASCII.
    #[error("error parsing array format string as UTF-8")]
    Utf8Parse(#[from] std::str::Utf8Error),

    #[error("unknown key: {0}")]
    UnknownKey(PyValue),

    #[error("missing key: {0}")]
    MissingKey(String),

    #[error("illegal value for key: {key} -> {value}")]
    IllegalValue { key: String, value: PyValue },

    #[error("error parsing metadata dict")]
    DictParse(#[from] PyValueParseError),

    #[error("metadata is not a dict: {0}")]
    MetaNotDict(PyValue),

    #[error("missing newline at end of header")]
    MissingNewline,
}

#[derive(Error, Debug)]
pub enum ReadHeaderError {
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] ParseHeaderError),
}

#[derive(Error, Debug)]
pub enum FormatHeaderError {
    #[error("Cannot format Python value")]
    PyValue(#[from] PyValueFormatError),

    /// The total header length overflows `usize`, or `HEADER_LEN` exceeds the
    /// maximum encodable value.
    #[error("the header is too long")]
    HeaderTooLong,
}

#[derive(Error, Debug)]
pub enum WriteHeaderError {
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("cannot format header")]
    Format(#[from] FormatHeaderError),
}

/// An error writing array data.
#[derive(Error, Debug)]
pub enum WriteDataError {
    /// An error caused by I/O.
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("Number of written elements ({1}) exceeds the size ({0}) given by the dim")]
    TooManyElements(usize, usize),

    #[error("Number of written elements ({1}) is less than the size ({0}) given by the dim")]
    TooFewElements(usize, usize),
}

/// An error writing a `.npy` file.
#[derive(Error, Debug)]
pub enum WriteNpyError {
    /// An error caused by I/O.
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// An error formatting the header.
    #[error("cannot format header")]
    FormatHeader(#[from] FormatHeaderError),

    #[error(transparent)]
    WriteHeader(#[from] WriteHeaderError),

    #[error(transparent)]
    WriteData(#[from] WriteDataError),
}

/// An error reading array data.
#[derive(Error, Debug)]
pub enum ReadDataError {
    /// An error caused by I/O.
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),

    /// The file does not contain all the data described in the header.
    #[error("reached EOF before reading all data")]
    MissingData,

    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),

    #[error("cannot parse value {0:#04x} as a bool")]
    ParseBoolError(u8),
}

/// An error reading a `.npy` file.
#[derive(Error, Debug)]
pub enum ReadNpyError {
    /// An error caused by I/O.
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// An error parsing the file header.
    #[error("cannot parse header")]
    ParseHeader(#[from] ParseHeaderError),

    #[error("cannot read header")]
    ReadHeader(#[from] ReadHeaderError),

    #[error("cannot read data")]
    ReadData(#[from] ReadDataError),

    /// Overflow while computing the length of the array from the shape
    /// described in the file header.
    #[error("overflow computing length from shape")]
    LengthOverflow,

    /// An error caused by incorrect `Dimension` type.
    #[error("ndim {1} of array did not match Dimension type with NDIM = {0:?}")]
    WrongNdim(Option<usize>, usize),

    /// The type descriptor does not match the element type.
    #[error("incorrect descriptor ({0}) for this type")]
    WrongDescriptor(PyValue),

    /// The file does not contain all the data described in the header.
    #[error("reached EOF before reading all data")]
    MissingData,

    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    #[error("file had {0} extra bytes before EOF")]
    ExtraBytes(usize),
}
