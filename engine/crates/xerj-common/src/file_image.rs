//! Immutable file images. TRUEOS owns the bytes because its runtime does not
//! provide file mappings; other platforms retain their memory-mapped readers.

use std::io;
use std::path::Path;

#[cfg(not(target_os = "trueos"))]
pub type FileImage = memmap2::Mmap;
#[cfg(target_os = "trueos")]
pub type FileImage = Box<[u8]>;

/// Open an immutable file for shared, random-access reads.
///
/// # Safety
/// On platforms using memory mapping, callers must ensure the file is not
/// modified or truncated while the returned image remains alive.
pub unsafe fn open(path: impl AsRef<Path>) -> io::Result<FileImage> {
    #[cfg(target_os = "trueos")]
    {
        read_owned(path.as_ref())
    }
    #[cfg(not(target_os = "trueos"))]
    {
        let file = std::fs::File::open(path)?;
        unsafe { memmap2::Mmap::map(&file) }
    }
}

#[cfg(any(target_os = "trueos", test))]
fn read_owned(path: &Path) -> io::Result<Box<[u8]>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty file"));
    }
    Ok(bytes.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_image_survives_file_removal_and_shares_without_copying() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("segment");
        std::fs::write(&path, b"immutable segment").unwrap();
        let image = std::sync::Arc::new(read_owned(&path).unwrap());
        std::fs::remove_file(&path).unwrap();
        let cached = image.clone();
        assert_eq!(&cached[..], b"immutable segment");
        assert_eq!(image.as_ptr(), cached.as_ptr());
    }

    #[test]
    fn owned_image_rejects_empty_and_missing_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty");
        assert_eq!(
            read_owned(&path).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        std::fs::write(&path, b"").unwrap();
        assert_eq!(
            read_owned(&path).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
