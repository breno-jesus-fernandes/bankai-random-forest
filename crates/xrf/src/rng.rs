/// Built-in pseudo-random generator
///
/// It is a 64 bit PCG generator.
/// Note that it support 32 and 64 bit systems only.
pub struct RfRng {
    state: u64,
    hop: u64,
}

impl RfRng {
    /// Get a random u32
    pub fn get_u32(&mut self) -> u32 {
        assert!(self.hop > 0);
        let prv = self.state;
        //Push generator
        self.state = (prv.wrapping_mul(6_364_136_223_846_793_005)).wrapping_add(self.hop);
        let xorshifted = (((prv >> 18) ^ prv) >> 27) as u32;
        let rot = (prv >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    /// Get a random u64
    pub fn get_u64(&mut self) -> u64 {
        ((self.get_u32() as u64) << 32) + (self.get_u32() as u64)
    }
    /// Get an integer value from [0;up_to)
    pub fn up_to(&mut self, up_to: usize) -> usize {
        //This assumes usize is u64 or less
        assert!(std::mem::size_of::<usize>() <= 64);
        assert!(up_to > 0);
        if up_to == 1 {
            0
        } else {
            let thresh = up_to.wrapping_neg() % up_to;
            loop {
                let mut p = self.get_u32() as usize;
                use std::mem::size_of;
                if size_of::<usize>() == 64 && up_to > u32::MAX as usize {
                    p = p << 32;
                    p += self.get_u32() as usize;
                }
                //rejection method:
                if p >= thresh {
                    break p % up_to;
                }
            }
        }
    }
    /// Create the generator object from seed and hop
    ///
    /// Hop is used as a stream number, to implement independent rng streams useful for reproducible parallel execution; it has to be different from 0.
    /// Seed is a typical PRNG seed.
    pub fn from_seed(seed: u64, hop: u64) -> Self {
        assert!(hop > 0);
        let mut ans = Self { state: seed, hop };
        ans.get_u32();
        ans.get_u32();
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic() {
        let mut rng = RfRng::from_seed(17, 1);
        let zero = rng.up_to(1);
        assert_eq!(zero, 0);
        let mut contents = [0, 0, 0];
        for _ in 0..300 {
            contents[rng.up_to(3)] += 1;
        }
        assert_eq!(contents[1], 93);
    }
    #[test]
    fn reproducible() {
        let mut rng = RfRng::from_seed(17, 1);
        let v64 = rng.get_u64();
        let v32 = rng.get_u32();
        assert_eq!(v64, 13030427649947795236);
        assert_eq!(v32, 1708324272);
    }
}
