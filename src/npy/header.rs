use super::error::*;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use num_traits::ToPrimitive;
use py_literal::Value as PyValue;
use std::convert::TryFrom;
use std::fmt;
use std::io;

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

/// The total header length (including magic string, version number, header
/// length value, array format description, padding, and final newline) must be
/// evenly divisible by this value.
const HEADER_DIVISOR: usize = 64;

struct HeaderLengthInfo {
    /// Total header length (including magic string, version number, header
    /// length value, array format description, padding, and final newline).
    total_len: usize,
    /// Formatted `HEADER_LEN` value. (This is the number of bytes in the array
    /// format description, padding, and final newline.)
    formatted_header_len: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub type_descriptor: PyValue,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
enum Version {
    V1_0,
    V2_0,
    V3_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    fn from_bytes(bytes: &[u8]) -> Result<Self, ParseHeaderError> {
        debug_assert_eq!(bytes.len(), Self::VERSION_NUM_BYTES);
        match (bytes[0], bytes[1]) {
            (0x01, 0x00) => Ok(Version::V1_0),
            (0x02, 0x00) => Ok(Version::V2_0),
            (0x03, 0x00) => Ok(Version::V3_0),
            (major, minor) => Err(ParseHeaderError::Version { major, minor }),
        }
    }

    /// Major version number.
    fn major_version(self) -> u8 {
        match self {
            Version::V1_0 => 1,
            Version::V2_0 => 2,
            Version::V3_0 => 3,
        }
    }

    /// Major version number.
    fn minor_version(self) -> u8 {
        match self {
            Version::V1_0 => 0,
            Version::V2_0 => 0,
            Version::V3_0 => 0,
        }
    }

    /// Number of bytes in representation of header length.
    fn header_len_num_bytes(self) -> usize {
        match self {
            Version::V1_0 => 2,
            Version::V2_0 | Version::V3_0 => 4,
        }
    }

    /// Read header length.
    fn read_header_len<R: io::Read>(self, mut reader: R) -> Result<usize, ReadHeaderError> {
        match self {
            Version::V1_0 => Ok(usize::from(reader.read_u16::<LittleEndian>()?)),
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = reader.read_u32::<LittleEndian>()?;
                Ok(usize::try_from(header_len)
                    .map_err(|_| ParseHeaderError::HeaderLengthOverflow(header_len))?)
            }
        }
    }

    /// Format header length as bytes for writing to file.
    ///
    /// Returns `None` if the value of `header_len` is too large for this .npy version.
    fn format_header_len(self, header_len: usize) -> Option<Vec<u8>> {
        match self {
            Version::V1_0 => {
                let header_len: u16 = u16::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u16(&mut out, header_len);
                Some(out)
            }
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = u32::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u32(&mut out, header_len);
                Some(out)
            }
        }
    }

    /// Computes the total header length, formatted `HEADER_LEN` value, and
    /// padding length for this .npy version.
    ///
    /// `unpadded_arr_format` is the Python literal describing the array
    /// format, formatted as an ASCII string without any padding.
    ///
    /// Returns `None` if the total header length overflows `usize` or if the
    /// value of `HEADER_LEN` is too large for this .npy version.
    fn compute_lengths(self, unpadded_arr_format: &[u8]) -> Option<HeaderLengthInfo> {
        /// Length of a '\n' char in bytes.
        const NEWLINE_LEN: usize = 1;

        let prefix_len: usize =
            MAGIC_STRING.len() + Version::VERSION_NUM_BYTES + self.header_len_num_bytes();
        let unpadded_total_len: usize = prefix_len
            .checked_add(unpadded_arr_format.len())?
            .checked_add(NEWLINE_LEN)?;
        let pad_res = unpadded_total_len % HEADER_DIVISOR;
        let padding_len: usize = match pad_res {
            0 => 0,
            _ => HEADER_DIVISOR - pad_res,
        };
        let total_len: usize = unpadded_total_len.checked_add(padding_len)?;
        let formatted_header_len = self.format_header_len(total_len - prefix_len)?;
        Some(HeaderLengthInfo {
            total_len,
            formatted_header_len,
        })
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_py_value())
    }
}

impl Header {
    fn from_py_value(value: PyValue) -> Result<Self, ParseHeaderError> {
        if let PyValue::Dict(dict) = value {
            let mut type_descriptor: Option<PyValue> = None;
            let mut fortran_order: Option<bool> = None;
            let mut shape: Option<Vec<usize>> = None;
            for (key, value) in dict {
                match key {
                    PyValue::String(ref k) if k == "descr" => {
                        type_descriptor = Some(value);
                    }
                    PyValue::String(ref k) if k == "fortran_order" => {
                        if let PyValue::Boolean(b) = value {
                            fortran_order = Some(b);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "fortran_order".to_owned(),
                                value,
                            });
                        }
                    }
                    PyValue::String(ref k) if k == "shape" => {
                        fn parse_shape(value: &PyValue) -> Option<Vec<usize>> {
                            value
                                .as_tuple()?
                                .iter()
                                .map(|elem| elem.as_integer()?.to_usize())
                                .collect()
                        }
                        if let Some(s) = parse_shape(&value) {
                            shape = Some(s);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "shape".to_owned(),
                                value,
                            });
                        }
                    }
                    k => return Err(ParseHeaderError::UnknownKey(k)),
                }
            }
            match (type_descriptor, fortran_order, shape) {
                (Some(type_descriptor), Some(fortran_order), Some(shape)) => Ok(Header {
                    type_descriptor,
                    fortran_order,
                    shape,
                }),
                (None, _, _) => Err(ParseHeaderError::MissingKey("descr".to_owned())),
                (_, None, _) => Err(ParseHeaderError::MissingKey("fortran_order".to_owned())),
                (_, _, None) => Err(ParseHeaderError::MissingKey("shaper".to_owned())),
            }
        } else {
            Err(ParseHeaderError::MetaNotDict(value))
        }
    }

    pub fn from_reader<R: io::Read>(mut reader: R) -> Result<Self, ReadHeaderError> {
        // Check for magic string.
        let mut buf = vec![0; MAGIC_STRING.len()];
        reader.read_exact(&mut buf)?;
        if buf != MAGIC_STRING {
            return Err(ParseHeaderError::MagicString.into());
        }

        // Get version number.
        let mut buf = [0; Version::VERSION_NUM_BYTES];
        reader.read_exact(&mut buf)?;
        let version = Version::from_bytes(&buf)?;

        // Get `HEADER_LEN`.
        let header_len = version.read_header_len(&mut reader)?;

        // Parse the dictionary describing the array's format.
        let mut buf = vec![0; header_len];
        reader.read_exact(&mut buf)?;
        let without_newline = match buf.split_last() {
            Some((&b'\n', rest)) => rest,
            Some(_) | None => return Err(ParseHeaderError::MissingNewline.into()),
        };
        let header_str = match version {
            Version::V1_0 | Version::V2_0 => {
                if without_newline.is_ascii() {
                    // ASCII strings are always valid UTF-8.
                    unsafe { std::str::from_utf8_unchecked(without_newline) }
                } else {
                    return Err(ParseHeaderError::NonAscii.into());
                }
            }
            Version::V3_0 => {
                std::str::from_utf8(without_newline).map_err(ParseHeaderError::from)?
            }
        };
        let arr_format: PyValue = header_str.parse().map_err(ParseHeaderError::from)?;
        Ok(Header::from_py_value(arr_format)?)
    }

    fn to_py_value(&self) -> PyValue {
        PyValue::Dict(vec![
            (
                PyValue::String("descr".into()),
                self.type_descriptor.clone(),
            ),
            (
                PyValue::String("fortran_order".into()),
                PyValue::Boolean(self.fortran_order),
            ),
            (
                PyValue::String("shape".into()),
                PyValue::Tuple(
                    self.shape
                        .iter()
                        .map(|&elem| PyValue::Integer(elem.into()))
                        .collect(),
                ),
            ),
        ])
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatHeaderError> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = Vec::new();
        self.to_py_value().write_ascii(&mut arr_format)?;

        // Determine appropriate version based on header length, and compute
        // length information.
        let (version, length_info) = [Version::V1_0, Version::V2_0]
            .iter()
            .find_map(|&version| Some((version, version.compute_lengths(&arr_format)?)))
            .ok_or(FormatHeaderError::HeaderTooLong)?;

        // Write the header.
        let mut out = Vec::with_capacity(length_info.total_len);
        out.extend_from_slice(MAGIC_STRING);
        out.push(version.major_version());
        out.push(version.minor_version());
        out.extend_from_slice(&length_info.formatted_header_len);
        out.extend_from_slice(&arr_format);
        out.resize(length_info.total_len - 1, b' ');
        out.push(b'\n');

        // Verify the length of the header.
        debug_assert_eq!(out.len(), length_info.total_len);
        debug_assert_eq!(out.len() % HEADER_DIVISOR, 0);

        Ok(out)
    }

    pub fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteHeaderError> {
        let bytes = self.to_bytes()?;
        writer.write_all(&bytes)?;
        Ok(())
    }
}
