use std::fs;
use std::path::Path;

/// Extensions to exclude from file discovery
const EXCLUDED_EXTENSIONS: &[&str] = &["md", "jpg", "pdf", "png", "js", "json", "html"];

/// Recursively finds all files in a directory, excluding specified extensions and hidden files.
///
/// # Arguments
/// * `dir` - The root directory to search
///
/// # Returns
/// A vector of file paths as strings, or an I/O error
///
/// # Behavior
/// - Recursively traverses all subdirectories
/// - Excludes hidden files (starting with `.`)
/// - Excludes symbolic links
/// - Excludes files with these extensions: md, jpg, pdf, png, js, json, html
/// - Returns sorted file paths for deterministic results
///
/// # Example
/// ```ignore
/// let files = find_text_files("corpus")?;
/// // Returns: ["corpus/file1.txt", "corpus/subdir/file2.txt", ...]
/// ```
pub fn find_text_files<P: AsRef<Path>>(dir: P) -> std::io::Result<Vec<String>> {
    let mut files = Vec::new();
    visit_dir(dir.as_ref(), &mut files)?;
    files.sort(); // For deterministic results across platforms
    Ok(files)
}

/// Recursively visits directories and collects file paths.
fn visit_dir(dir: &Path, files: &mut Vec<String>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Get file name for hidden file check
        if let Some(file_name) = path.file_name() {
            if let Some(name_str) = file_name.to_str() {
                // Skip hidden files (starting with .)
                if name_str.starts_with('.') {
                    continue;
                }
            }
        }

        // Check if it's a symlink and skip if so
        let metadata = entry.metadata()?;
        if metadata.is_symlink() {
            continue;
        }

        if metadata.is_dir() {
            // Recursively visit subdirectories
            visit_dir(&path, files)?;
        } else if metadata.is_file() {
            // Check if file extension is in the excluded list
            let should_include = if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    !EXCLUDED_EXTENSIONS.contains(&ext_str)
                } else {
                    true // Include files with non-UTF8 extensions
                }
            } else {
                true // Include files without extension
            };

            if should_include {
                if let Some(path_str) = path.to_str() {
                    files.push(path_str.to_string());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use tempfile::TempDir;

    #[test]
    fn test_find_text_files_basic() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Create some test files
        File::create(temp_path.join("file1.txt")).unwrap();
        File::create(temp_path.join("file2.csv")).unwrap();
        File::create(temp_path.join("image.jpg")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should include .txt and .csv, but not .jpg
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.ends_with("file1.txt")));
        assert!(files.iter().any(|f| f.ends_with("file2.csv")));
        assert!(!files.iter().any(|f| f.ends_with("image.jpg")));
    }

    #[test]
    fn test_find_text_files_excludes_json() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("data.json")).unwrap();
        File::create(temp_path.join("config.html")).unwrap();
        File::create(temp_path.join("script.js")).unwrap();
        File::create(temp_path.join("readme.md")).unwrap();
        File::create(temp_path.join("content.txt")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should only include content.txt
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("content.txt"));
    }

    #[test]
    fn test_find_text_files_recursive() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("root.txt")).unwrap();

        fs::create_dir(temp_path.join("subdir")).unwrap();
        File::create(temp_path.join("subdir/file.txt")).unwrap();

        fs::create_dir(temp_path.join("subdir/nested")).unwrap();
        File::create(temp_path.join("subdir/nested/deep.txt")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should find all 3 text files
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_find_text_files_excludes_hidden_files() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("visible.txt")).unwrap();
        File::create(temp_path.join(".hidden.txt")).unwrap();
        File::create(temp_path.join(".DS_Store")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should only find visible.txt
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("visible.txt"));
        assert!(!files.iter().any(|f| f.contains(".hidden")));
    }

    #[test]
    fn test_find_text_files_handles_special_chars_in_filename() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("file with spaces.txt")).unwrap();
        File::create(temp_path.join("file,with,commas.txt")).unwrap();
        File::create(temp_path.join("file&special$chars.txt")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // All files with special characters should be included
        assert_eq!(files.len(), 3);
        assert!(files.iter().any(|f| f.contains("file with spaces")));
        assert!(files.iter().any(|f| f.contains("file,with,commas")));
        assert!(files.iter().any(|f| f.contains("file&special")));
    }

    #[test]
    fn test_find_text_files_returns_sorted_paths() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Create files in non-alphabetical order
        File::create(temp_path.join("z_file.txt")).unwrap();
        File::create(temp_path.join("a_file.txt")).unwrap();
        File::create(temp_path.join("m_file.txt")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should be sorted
        assert_eq!(files.len(), 3);
        assert!(files[0].ends_with("a_file.txt"));
        assert!(files[1].ends_with("m_file.txt"));
        assert!(files[2].ends_with("z_file.txt"));
    }

    #[test]
    fn test_find_text_files_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let files = find_text_files(temp_dir.path()).unwrap();

        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_find_text_files_only_excluded_files() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("image.jpg")).unwrap();
        File::create(temp_path.join("config.json")).unwrap();
        File::create(temp_path.join("script.js")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_find_text_files_no_extension() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("LICENSE")).unwrap();
        File::create(temp_path.join("README")).unwrap();
        File::create(temp_path.join("data.json")).unwrap();

        let files = find_text_files(temp_path).unwrap();

        // Should include files with no extension
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.ends_with("LICENSE")));
        assert!(files.iter().any(|f| f.ends_with("README")));
    }
}
