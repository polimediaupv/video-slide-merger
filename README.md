# Video Slide Merger

This utility replaces the frames of a Power Point presentation video with an alternative version, specified by a sequence of images. It also combine the pointer cursor with the new images.

## Usage

```
merge-pointer original_video.mp4 destination.mp4 sequence/of/images/path
```

In the image path folder you must place:

- A sequence of images, extracted from the original video, named as `frame_0.jpg`, `frame_1.jpg`... `frame_n.jpg`
- A sequence with the alternative images, named as `frame_alt_0.jpg`, `frame_alt_1.jpg`... `frame_alt_n.jpg`

## Dependencies

Video Slide Merger depends on OpenCV. You also need CMake to generate the project files.

## Build

### Linux

Install OpenCV dependencies from sources or using a package manager (line apt-get). Build using CMake and make:

```
cd video-slide-merger
mkdir build
cd build
cmake ..
make
```

### macOS

Install OpenCV dependencies from sources, binaries or using homebrew.

You can build the project using make. You can also use Xcode to build and debug the project:

```
cd video-slide-merger
mkdir build
cd build
cmake .. -G Xcode
open merge-pointer.xcodeproj
```

### Windows

Get the OpenCV binaries for your Visual Studio version and use CMake GUI to generate the Visual Studio project.

