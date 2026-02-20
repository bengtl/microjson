/// Binary fragment format for disk-backed tile assembly.
///
/// During streaming ingestion, each feature is clipped to each tile at each
/// zoom level, producing "fragments". These are written to an append-only
/// temp file. During tile encoding, fragments for each tile are read back
/// and merged.
///
/// Fragment binary format (little-endian):
///   [4 bytes] magic: b"MJF1"
///   [4 bytes] feature_id: u32
///   [4 bytes] tile_z: u32
///   [4 bytes] tile_x: u32
///   [4 bytes] tile_y: u32
///   [4 bytes] tile_d: u32
///   [1 byte]  geom_type: u8
///   [4 bytes] n_vertices: u32
///   [n_vertices * 8 bytes] xy: pairs of f64 (flat x,y,x,y,...)
///   [n_vertices * 8 bytes] z: f64 values
///   [4 bytes] n_ring_lengths: u32
///   [n_ring_lengths * 4 bytes] ring_lengths: u32
///   [4 bytes] payload_end_magic: b"FEND"

use std::io::{self, Read, Write, BufWriter, BufReader, Seek, SeekFrom};
use std::fs::File;
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

const MAGIC: &[u8; 4] = b"MJF1";
const END_MAGIC: &[u8; 4] = b"FEND";

/// A single tile fragment — the clipped portion of one feature in one tile.
#[derive(Debug, Clone)]
pub struct Fragment {
    pub feature_id: u32,
    pub tile_z: u32,
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_d: u32,
    pub geom_type: u8,
    pub xy: Vec<f64>,
    pub z: Vec<f64>,
    pub ring_lengths: Vec<u32>,
}

/// Append-only fragment writer.
pub struct FragmentWriter {
    writer: BufWriter<File>,
    count: u64,
}

impl FragmentWriter {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, file), // 1 MB buffer
            count: 0,
        })
    }

    pub fn write(&mut self, frag: &Fragment) -> io::Result<()> {
        let n_verts = frag.z.len() as u32;

        self.writer.write_all(MAGIC)?;
        self.writer.write_u32::<LittleEndian>(frag.feature_id)?;
        self.writer.write_u32::<LittleEndian>(frag.tile_z)?;
        self.writer.write_u32::<LittleEndian>(frag.tile_x)?;
        self.writer.write_u32::<LittleEndian>(frag.tile_y)?;
        self.writer.write_u32::<LittleEndian>(frag.tile_d)?;
        self.writer.write_u8(frag.geom_type)?;
        self.writer.write_u32::<LittleEndian>(n_verts)?;

        // XY: n_verts * 2 doubles
        for &v in &frag.xy {
            self.writer.write_f64::<LittleEndian>(v)?;
        }

        // Z: n_verts doubles
        for &v in &frag.z {
            self.writer.write_f64::<LittleEndian>(v)?;
        }

        // Ring lengths
        self.writer.write_u32::<LittleEndian>(frag.ring_lengths.len() as u32)?;
        for &rl in &frag.ring_lengths {
            self.writer.write_u32::<LittleEndian>(rl)?;
        }

        self.writer.write_all(END_MAGIC)?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

/// Fragment reader — reads all fragments from a file.
pub struct FragmentReader {
    reader: BufReader<File>,
}

impl FragmentReader {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, file),
        })
    }

    pub fn read_next(&mut self) -> io::Result<Option<Fragment>> {
        // Read magic
        let mut magic = [0u8; 4];
        match self.reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment magic"));
        }

        let feature_id = self.reader.read_u32::<LittleEndian>()?;
        let tile_z = self.reader.read_u32::<LittleEndian>()?;
        let tile_x = self.reader.read_u32::<LittleEndian>()?;
        let tile_y = self.reader.read_u32::<LittleEndian>()?;
        let tile_d = self.reader.read_u32::<LittleEndian>()?;
        let geom_type = self.reader.read_u8()?;
        let n_verts = self.reader.read_u32::<LittleEndian>()? as usize;

        // XY
        let mut xy = Vec::with_capacity(n_verts * 2);
        for _ in 0..n_verts * 2 {
            xy.push(self.reader.read_f64::<LittleEndian>()?);
        }

        // Z
        let mut z = Vec::with_capacity(n_verts);
        for _ in 0..n_verts {
            z.push(self.reader.read_f64::<LittleEndian>()?);
        }

        // Ring lengths
        let n_rl = self.reader.read_u32::<LittleEndian>()? as usize;
        let mut ring_lengths = Vec::with_capacity(n_rl);
        for _ in 0..n_rl {
            ring_lengths.push(self.reader.read_u32::<LittleEndian>()?);
        }

        // End magic
        let mut end = [0u8; 4];
        self.reader.read_exact(&mut end)?;
        if &end != END_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment end magic"));
        }

        Ok(Some(Fragment {
            feature_id,
            tile_z,
            tile_x,
            tile_y,
            tile_d,
            geom_type,
            xy,
            z,
            ring_lengths,
        }))
    }

    /// Read all fragments, grouped by tile key.
    pub fn read_all_grouped(&mut self) -> io::Result<ahash::AHashMap<(u32, u32, u32, u32), Vec<Fragment>>> {
        use ahash::AHashMap;
        let mut groups: AHashMap<(u32, u32, u32, u32), Vec<Fragment>> = AHashMap::new();
        while let Some(frag) = self.read_next()? {
            let key = (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d);
            groups.entry(key).or_default().push(frag);
        }
        Ok(groups)
    }

    /// Read all fragments, grouped by feature_id.
    ///
    /// Used by `generate_neuroglancer()` which needs segment-centric grouping
    /// (one mesh per feature) rather than tile-centric grouping.
    pub fn read_all_grouped_by_feature(&mut self) -> io::Result<ahash::AHashMap<u32, Vec<Fragment>>> {
        use ahash::AHashMap;
        self.reset()?;
        let mut groups: AHashMap<u32, Vec<Fragment>> = AHashMap::new();
        while let Some(frag) = self.read_next()? {
            groups.entry(frag.feature_id).or_default().push(frag);
        }
        Ok(groups)
    }

    /// Reset reader to beginning of file.
    pub fn reset(&mut self) -> io::Result<()> {
        self.reader.seek(SeekFrom::Start(0))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_fragments.mjf");

        let frag = Fragment {
            feature_id: 42,
            tile_z: 2,
            tile_x: 1,
            tile_y: 3,
            tile_d: 0,
            geom_type: 5,
            xy: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            z: vec![0.7, 0.8, 0.9],
            ring_lengths: vec![3],
        };

        {
            let mut writer = FragmentWriter::new(&path).unwrap();
            writer.write(&frag).unwrap();
            writer.write(&frag).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.count(), 2);
        }

        {
            let mut reader = FragmentReader::new(&path).unwrap();
            let f1 = reader.read_next().unwrap().unwrap();
            assert_eq!(f1.feature_id, 42);
            assert_eq!(f1.tile_z, 2);
            assert_eq!(f1.xy.len(), 6);
            assert_eq!(f1.z.len(), 3);
            let f2 = reader.read_next().unwrap().unwrap();
            assert_eq!(f2.feature_id, 42);
            let f3 = reader.read_next().unwrap();
            assert!(f3.is_none());
        }

        std::fs::remove_file(&path).ok();
    }
}
