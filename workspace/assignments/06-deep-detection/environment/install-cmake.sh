#!/bin/bash

# configurations:
version=3.13
build=2
nproc=8
# download:
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
# make and install
./bootstrap
make -j$(nproc)
sudo make install
rm -rf cmake-$version.$build/
