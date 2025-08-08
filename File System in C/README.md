# FSO File System Implementation

A custom Unix-like file system implementation in C, developed as part of the Operating Systems (FSO) course at FCT-UNL. This project demonstrates fundamental file system concepts including block-based storage, bitmap allocation, and directory management.

## 👥 Authors
- **José Morgado** (59457)
- **João Esteves** (47994)

## 🎯 Project Overview

This project implements a simplified file system with the following core features:
- **Block-based storage** with 2KB blocks
- **Bitmap-based free space management**
- **Flat directory structure** with direct block allocation
- **File operations**: create, read, write, copy in/out
- **Interactive shell interface** for file system operations
- **Maximum file size**: 15 blocks × 2048 bytes = 30,720 bytes

## 🏗️ System Architecture

### File System Layout
```
Block 0: Superblock (metadata)
Block 1: Free space bitmap
Block 2+: Data blocks (files and directory entries)
```

### Core Components

#### 1. **Disk Simulation Layer** (`disk.c/h`)
- Simulates a block device using a regular file
- Provides block-level read/write operations with error checking
- Manages disk geometry and tracks I/O statistics
- Fixed block size: 2048 bytes

#### 2. **Bitmap Management** (`bitmap.c/h`)
- Efficient bit manipulation for free space tracking
- Operations: set, clear, and read individual bits
- Memory allocation/deallocation for bitmaps
- Debug printing functionality for visualization

#### 3. **File System Core** (`fs.c/h`)
- **Superblock management** with file system metadata
- **Directory entry (inode) management** with direct block pointers
- **File operations**: creation, reading, writing
- **Mount/unmount functionality** with validation

#### 4. **Interactive Shell** (`fso-sh.c`)
- Command-line interface for all file system operations
- File import/export between host and file system
- Debug and inspection tools

### Key Data Structures

```c
struct fs_superblock {
    uint32_t magic;              // File system magic number (0xfafafefe)
    uint32_t nblocks;            // Total number of blocks
    uint32_t bmapsize_bytes;     // Bitmap size in bytes
    uint32_t bmapsize_blocks;    // Bitmap size in blocks (usually 1)
    uint32_t first_datablock;    // First data block number (usually 2)
    uint32_t data_size;          // Number of data blocks
    uint16_t blocksize;          // Block size (2048 bytes)
    struct fs_dirent rootdir;    // Root directory entry
};

struct fs_dirent {
    uint8_t isvalid;                    // Entry validity flag
    char name[63];                      // File name (null-terminated)
    uint32_t size;                      // File size in bytes
    uint32_t blk[15];                   // Direct block pointers (max 15 blocks)
};
```

## 🚀 Implementation Details

### What We Implemented

The following functions were implemented from scratch after the TODO markers:

#### **1. File System Formatting (`fs_format`)**
- Initializes superblock with proper magic number and metadata
- Sets up bitmap with system blocks marked as used
- Creates root directory entry
- Calculates and stores file system geometry

#### **2. Directory Listing (`fs_lsdir`)**
- Traverses all directory blocks in the root directory
- Displays file entries with size, name, and block allocation
- Handles multiple directory blocks efficiently

#### **3. File Name Resolution (`name2number`)**
- Searches through directory blocks to find files by name
- Returns directory entry number for existing files
- Implements efficient block-by-block search algorithm

#### **4. File Reading (`fs_read`)**
- Reads data from files at specified offset and length
- Handles cross-block reads seamlessly
- Implements proper boundary checking and error handling
- Supports partial reads when requested length exceeds file size

#### **5. File Writing (`fs_write`)**
- Writes data to files with automatic block allocation
- Handles both overwriting existing blocks and allocating new ones
- Implements complex logic for:
  - Cross-block writes
  - Partial block updates
  - Dynamic block allocation
  - File size updates
- Updates directory entries and superblock automatically

### Technical Highlights

- **Block Allocation**: First-fit algorithm using bitmap for efficient space management
- **Cross-block I/O**: Seamless reading/writing across multiple blocks
- **Dynamic Growth**: Files can grow dynamically up to 15 blocks
- **Data Integrity**: Proper synchronization between memory and disk structures
- **Error Handling**: Comprehensive bounds checking and error reporting

## 💻 Building and Usage

### Prerequisites
- GCC compiler
- Make utility
- Unix-like operating system (Linux/macOS)

### Compilation
```bash
make all        # Build the file system
make clean      # Clean build files
```

### Starting the File System
```bash
# Create a new disk with 1000 blocks
./fso-sh mydisk.img 1000

# Use an existing disk
./fso-sh mydisk.img
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `format` | Initialize/format the disk | `format` |
| `mount` | Mount the file system | `mount` |
| `umount` | Unmount the file system | `umount` |
| `ls` | List files in directory | `ls` |
| `create <name>` | Create a new empty file | `create myfile.txt` |
| `copyin <src> <dest>` | Copy file from host to FS | `copyin test.txt myfile` |
| `copyout <src> <dest>` | Copy file from FS to host | `copyout myfile test.txt` |
| `cat <name>` | Display file contents | `cat myfile` |
| `debug` | Show file system internals | `debug` |
| `help` or `?` | Show available commands | `help` |
| `quit` or `exit` | Exit the shell | `quit` |

### Example Session
```bash
$ ./fso-sh mydisk.img 100
opened emulated disk image mydisk.img with 100 blocks

fso-sh> format
superblock:
    magic = fafafefe
    volume size = 100
    total blocks = 100
    bmapsize bytes = 13
    bmapsize blocks = 1
    first data block = 2
    data size = 98
    block size = 2048
disk formatted.

fso-sh> mount
disk mounted.

fso-sh> create hello.txt
created inode 0

fso-sh> copyin test.txt hello.txt
1234 bytes copied

fso-sh> ls
entry   size    name    blocks
0       1234    hello.txt       2 0 0 0 0 0 0 0 0 0 0 0 0 0 0

fso-sh> cat hello.txt
[file contents displayed]

fso-sh> copyout hello.txt output.txt
1234 bytes copied
copied fs hello.txt to file output.txt

fso-sh> quit
```

## 🔧 Technical Specifications

### File System Limits
- **Maximum file size**: 15 blocks × 2048 bytes = 30,720 bytes
- **Maximum files**: Limited by available blocks and directory space
- **File name length**: Up to 63 characters
- **Directory structure**: Flat (no subdirectories)
- **Block size**: 2048 bytes (fixed)

### Performance Characteristics
- **Block allocation**: O(n) first-fit algorithm
- **File lookup**: O(n) linear search through directory
- **Read/Write**: O(blocks) proportional to data size

### Current Limitations
- No subdirectory support
- No file deletion functionality
- No indirect block pointers (limits maximum file size)
- Single-user, single-threaded access
- No file permissions or extended metadata
- No journaling or crash recovery

## 🧪 Testing

The project includes test files and disk images:
- `image20` - Small test disk (20 blocks)
- `image1000` - Larger test disk (1000 blocks)
- `test.txt`, `a.txt`, `b.txt` - Sample files for testing

### Test Scenarios
```bash
# Test basic operations
./fso-sh test_disk 50
fso-sh> format
fso-sh> mount
fso-sh> create test1
fso-sh> copyin a.txt test1
fso-sh> ls
fso-sh> cat test1
fso-sh> copyout test1 output.txt

# Test large files
fso-sh> copyin /etc/passwd large_file
fso-sh> debug  # Check bitmap and block allocation
```

## 📚 Learning Outcomes

This implementation demonstrates mastery of:
- **Low-level file system design** and implementation
- **Block-based storage management** with bitmap allocation
- **C programming** with complex data structures and memory management
- **System programming** concepts including I/O and error handling
- **Data structure design** for systems programming
- **Debugging** and testing of system-level code

## 🔍 Code Quality Features

- **Modular design** with clear separation of concerns
- **Comprehensive error checking** and boundary validation
- **Memory safety** with proper buffer management
- **Consistent coding style** and documentation
- **Efficient algorithms** for core operations

## 📄 Academic Context

This project was developed for the Operating Systems (FSO) course at FCT-UNL, demonstrating practical understanding of file system internals and low-level systems programming concepts.

---

*This implementation serves as an educational tool to understand the fundamental concepts behind modern file systems like ext2/3/4, FAT, and other block-based file systems.*