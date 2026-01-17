use rkyv::{Archive, Deserialize, Serialize};

/// A fast direct-indexed map from `u32` keys to `u32` values.
/// Keys are used as indices into an internal vector.
/// Unused entries are set to `u32::MAX` as a sentinel value.
#[derive(Archive, Deserialize, Serialize)]
pub struct DirectMap {
    mapping: Vec<u32>,
}

impl DirectMap {
    /// Creates a new `DirectMap` with capacity for keys up to `max_size`.
    /// All entries are initialized to `u32::MAX` (not set).
    ///
    /// # Arguments
    ///
    /// * `max_size` - The maximum key value to support initially.
    pub fn new(max_size: usize) -> Self {
        // Initialize with u32::MAX as a sentinel for "Not Found"
        Self {
            mapping: vec![u32::MAX; max_size + 1],
        }
    }

    /// Inserts or updates the value for the given key.
    /// Resizes the internal vector if the key exceeds current capacity.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert.
    /// * `value` - The value to associate with the key.
    pub fn insert(&mut self, key: u32, value: u32) {
        if (key as usize) >= self.mapping.len() {
            // Resize if we see an unexpectedly large ID
            self.mapping.resize(key as usize + 1, u32::MAX);
        }
        self.mapping[key as usize] = value;
    }

    /// Retrieves the value associated with the given key, if present.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up.
    ///
    /// # Returns
    ///
    /// * `Some(value)` if the key is set, or `None` if not found.
    #[inline(always)]
    pub fn get(&self, key: u32) -> Option<u32> {
        match self.mapping.get(key as usize) {
            Some(&val) if val != u32::MAX => Some(val),
            _ => None,
        }
    }
    /// Returns a vector of all keys that have an associated value.
    ///
    /// # Returns
    ///
    /// * `Vec<u32>` containing all keys with set values.
    pub fn keys(&self) -> Vec<u32> {
        self.mapping
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| {
                if value != u32::MAX {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl FromIterator<(u32, u32)> for DirectMap {
    /// Constructs a `DirectMap` from an iterator of `(key, value)` pairs.
    /// The map is sized to fit the largest key.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator of `(u32, u32)` pairs.
    ///
    /// # Returns
    ///
    /// * A new `DirectMap` containing the provided pairs.
    fn from_iter<I: IntoIterator<Item = (u32, u32)>>(iter: I) -> Self {
        let pairs: Vec<(u32, u32)> = iter.into_iter().collect();

        // Find the maximum key to determine initial size
        let max_key = pairs.iter().map(|(k, _)| *k).max().unwrap_or(0);

        let mut map = DirectMap::new(max_key as usize);

        for (key, value) in pairs {
            map.insert(key, value);
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_empty_map() {
        let map = DirectMap::new(10);
        assert_eq!(map.get(0), None);
        assert_eq!(map.get(5), None);
        assert_eq!(map.get(10), None);
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = DirectMap::new(10);
        map.insert(5, 100);
        assert_eq!(map.get(5), Some(100));
        assert_eq!(map.get(4), None);
        assert_eq!(map.get(6), None);
    }

    #[test]
    fn test_insert_overwrites_existing_value() {
        let mut map = DirectMap::new(10);
        map.insert(3, 50);
        map.insert(3, 75);
        assert_eq!(map.get(3), Some(75));
    }

    #[test]
    fn test_resize_on_large_key() {
        let mut map = DirectMap::new(5);
        map.insert(10, 200);
        assert_eq!(map.get(10), Some(200));
        assert_eq!(map.get(5), None);
    }

    #[test]
    fn test_multiple_insertions() {
        let mut map = DirectMap::new(20);
        map.insert(0, 10);
        map.insert(15, 150);
        map.insert(7, 70);

        assert_eq!(map.get(0), Some(10));
        assert_eq!(map.get(15), Some(150));
        assert_eq!(map.get(7), Some(70));
        assert_eq!(map.get(1), None);
        assert_eq!(map.get(8), None);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let map = DirectMap::new(5);
        assert_eq!(map.get(100), None);
    }

    #[test]
    fn test_keys_method() {
        let mut map = DirectMap::new(10);
        map.insert(2, 20);
        map.insert(5, 50);
        map.insert(8, 80);

        let keys = map.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&2));
        assert!(keys.contains(&5));
        assert!(keys.contains(&8));
    }

    #[test]
    fn test_keys_empty_map() {
        let map = DirectMap::new(5);
        assert_eq!(map.keys(), Vec::<u32>::new());
    }
}
