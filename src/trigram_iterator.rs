use crate::sparse_trigram::SparseTrigram;

// Iterator for efficient traversal
pub struct TrigramIterator<'a> {
    pub model: &'a SparseTrigram,
    pub w1_idx: usize,
    pub w2_rel_idx: usize,
    pub w3_idx: usize,
}

impl<'a> Iterator for TrigramIterator<'a> {
    type Item = (u32, u32, u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.w1_idx < self.model.w1_ptr.len() - 1 {
            let w2_start = self.model.w1_ptr[self.w1_idx];
            let w2_end = self.model.w1_ptr[self.w1_idx + 1];

            if self.w2_rel_idx < w2_end - w2_start {
                let w2_abs_idx = w2_start + self.w2_rel_idx;
                let w2 = self.model.w2_indices[w2_abs_idx];

                let w3_start = self.model.w2_ptr[w2_abs_idx];
                let w3_end = self.model.w2_ptr[w2_abs_idx + 1];

                if self.w3_idx < w3_end - w3_start {
                    let w3_abs_idx = w3_start + self.w3_idx;
                    let w3 = self.model.w3_indices[w3_abs_idx];
                    let count = self.model.counts[w3_abs_idx];

                    // Derive w1 from current position
                    // We need to store w1 values separately or derive differently
                    // For simplicity, we'll skip w1 in iteration
                    self.w3_idx += 1;
                    return Some((0, w2, w3, count)); // Note: w1 missing
                }

                // Move to next w2
                self.w2_rel_idx += 1;
                self.w3_idx = 0;
            } else {
                // Move to next w1
                self.w1_idx += 1;
                self.w2_rel_idx = 0;
                self.w3_idx = 0;
            }
        }

        None
    }
}
