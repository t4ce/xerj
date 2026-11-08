//! Credentials inherit the kernel-provisioned private workspace's access key.
//! The host publishes its root marker only after validating persistent ownership.
use std::{
    io,
    path::{Component, Path},
};

const ROOT: &str = "/apps/xerj";
const REFERENCE: &str = ".trueos-private-root";
const STAMP: &[u8] = b"trueos-private-workspace-v1\n";
const MARKER: &str = "TRUEOS_PRIVATE_APP_ROOT";

fn denied(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, message)
}

fn check_path(path: &Path) -> io::Result<()> {
    let relative = path
        .strip_prefix(ROOT)
        .map_err(|_| denied("native secrets require the provisioned /apps/xerj workspace"))?;
    if !path.is_absolute()
        || relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || relative == Path::new(REFERENCE)
    {
        return Err(denied("native secret path is not a workspace file"));
    }
    Ok(())
}

#[cfg(target_os = "trueos")]
mod native {
    use super::*;
    use std::{
        fs::{self, File},
        io::{Read, Write},
        path::PathBuf,
    };

    fn absolute_path(path: &Path) -> io::Result<PathBuf> {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        check_path(&path)?;
        Ok(path)
    }

    fn reference() -> io::Result<File> {
        if std::env::var_os(MARKER).as_deref() != Some(std::ffi::OsStr::new(ROOT)) {
            return Err(denied(
                "native private workspace was not provisioned by the host",
            ));
        }
        if fs::canonicalize(ROOT)? != Path::new(ROOT) {
            return Err(denied("native private workspace changed its location"));
        }
        let mut reference = File::open(Path::new(ROOT).join(REFERENCE))?;
        let metadata = reference.metadata()?;
        if !metadata.is_file() || metadata.len() != STAMP.len() as u64 {
            return Err(denied(
                "native private workspace has an invalid ownership stamp",
            ));
        }
        let mut stamp = [0; STAMP.len()];
        reference.read_exact(&mut stamp)?;
        if stamp.as_slice() != STAMP {
            return Err(denied(
                "native private workspace has an invalid ownership stamp",
            ));
        }
        let root = File::open(ROOT)?;
        if !root.metadata()?.is_dir()
            || root.metadata()?.root_id() != reference.metadata()?.root_id()
        {
            return Err(denied("native workspace ownership reference changed"));
        }
        Ok(reference)
    }

    fn check_directory(path: &Path, reference: &File) -> io::Result<()> {
        if !fs::canonicalize(path)?.starts_with(ROOT) {
            return Err(denied("native secret parent escapes its private workspace"));
        }
        let directory = File::open(path)?;
        if !directory.metadata()?.is_dir()
            || directory.metadata()?.root_id() != reference.metadata()?.root_id()
        {
            return Err(denied("native secret parent changed its filesystem"));
        }
        Ok(())
    }

    fn check_inherited_key(path: &Path, reference: &File) -> io::Result<()> {
        // The native equivalence API accepts files, not directories. An
        // empty probe proves the key a new child inherits before any secret
        // is written; its owner removes it on both success and failure.
        let probe = tempfile::Builder::new()
            .prefix(".xerj-secret-parent-probe-")
            .tempfile_in(path)?;
        check_file(probe.as_file(), reference)
    }

    fn prepare(path: &Path, reference: &File) -> io::Result<PathBuf> {
        check_path(path)?;
        let parent = path
            .parent()
            .ok_or_else(|| denied("native secret has no parent"))?;
        let mut ancestor = parent;
        loop {
            match fs::metadata(ancestor) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    ancestor = ancestor
                        .parent()
                        .ok_or_else(|| denied("missing native workspace"))?;
                }
                Err(error) => return Err(error),
            }
        }
        check_directory(ancestor, reference)?;
        check_inherited_key(ancestor, reference)?;
        fs::create_dir_all(parent)?;
        check_directory(parent, reference)?;
        check_inherited_key(parent, reference)?;
        Ok(parent.to_path_buf())
    }

    fn check_file(file: &File, reference: &File) -> io::Result<()> {
        if !file.metadata()?.is_file() || !file.has_same_access_key(reference)? {
            return Err(denied("native secret has another access key"));
        }
        Ok(())
    }

    /// Validate an existing credential without changing its content or grants.
    pub fn validate(path: &Path) -> io::Result<()> {
        let path = absolute_path(path)?;
        let reference = reference()?;
        check_directory(path.parent().unwrap(), &reference)?;
        check_file(&File::open(&path)?, &reference)
    }

    pub fn read_to_string(path: &Path) -> io::Result<String> {
        let path = absolute_path(path)?;
        let reference = reference()?;
        check_directory(path.parent().unwrap(), &reference)?;
        let file = File::open(&path)?;
        check_file(&file, &reference)?;
        if file.metadata()?.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native credential is oversized",
            ));
        }
        let mut contents = String::new();
        file.take(64 * 1024 + 1).read_to_string(&mut contents)?;
        if contents.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native credential is oversized",
            ));
        }
        Ok(contents)
    }

    /// Check inherited native grants before writing any secret bytes.
    pub fn write(path: &Path, contents: &[u8]) -> io::Result<()> {
        let path = absolute_path(path)?;
        let reference = reference()?;
        prepare(&path, &reference)?;
        // A pre-existing foreign-key file must never be truncated first.
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        check_file(&file, &reference)?;
        file.set_len(0)?;
        file.write_all(contents)?;
        file.sync_all()?;
        let named = File::open(&path)?;
        check_file(&named, &reference)?;
        let (ours, named) = (file.metadata()?, named.metadata()?);
        if ours.root_id() != named.root_id()
            || !ours.file_id().is_some_and(|id| Some(id) == named.file_id())
        {
            return Err(denied("native credential path changed during its write"));
        }
        Ok(())
    }
}

#[cfg(target_os = "trueos")]
pub use native::{read_to_string, validate, write};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_paths_reject_escape_and_the_ownership_reference() {
        for path in [
            "/apps/xerj-other/admin.key",
            "/apps/xerj/../admin.key",
            "/tmp/admin.key",
            "apps/xerj/admin.key",
            ROOT,
            "/apps/xerj/.trueos-private-root",
        ] {
            assert_eq!(
                check_path(Path::new(path)).unwrap_err().kind(),
                io::ErrorKind::PermissionDenied,
                "{path}"
            );
        }
    }

    #[test]
    fn credential_paths_accept_nested_workspace_files() {
        for path in [
            "/apps/xerj/data/admin.key",
            "/apps/xerj/data/.xerj_master_key",
            "/apps/xerj/smoke/fixture/admin.key",
        ] {
            check_path(Path::new(path)).unwrap();
        }
    }
}
