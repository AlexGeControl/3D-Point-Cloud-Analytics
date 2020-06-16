FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# ------ PART 0: set environment variables ------

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root SHELL=/bin/bash

USER root

# ------ PART 1: set up sources & downloader ------

# remove default NVIDIA sources 
# otherwise CUDA iwould be upgraded, which breaks the dependency of DL libs like Tensorflow and PyTorch:
RUN rm -f /etc/apt/sources.list.d/*
# use Ubuntu CN sources:
COPY ${PWD}/image/etc/apt/sources.list /etc/apt/sources.list
# use Python CN sources: 
COPY ${PWD}/image/etc/pip.conf /root/.pip/pip.conf

# install apt-fast:
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated dirmngr gnupg2 software-properties-common axel aria2 && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-keys 1EE2FF37CA8DA16B && \
    add-apt-repository ppa:apt-fast/stable && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated apt-fast && \
    rm -rf /var/lib/apt/lists/*

# ------ PART 2: install Ubuntu packages ------

# use apt-fast CN sources:
COPY ${PWD}/image/etc/apt-fast.conf /etc/apt-fast.conf

# install packages:
RUN apt-fast update --fix-missing && \
    apt-fast install -y --no-install-recommends --allow-downgrades --allow-change-held-packages --allow-unauthenticated \
        curl grep sed dpkg wget bzip2 ca-certificates \
        git mercurial subversion \
        supervisor \
        openssh-server pwgen sudo vim-tiny \
        net-tools \
        lxde x11vnc xvfb \
        xorg-dev \
        mesa-utils libgl1-mesa-dri libglu1-mesa-dev \
        gtk2-engines-murrine ttf-ubuntu-font-family \
        firefox \
        nginx \
        python3-pip python3-dev \
        cmake build-essential libboost-all-dev \
        gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        dbus-x11 x11-utils \
        terminator \
        # KITTI evaluation toolkit:
        gnuplot ghostscript \
        # latex:
        texlive-extra-utils texlive-latex-extra \
        cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libdw-dev \
        libpcl-dev && \
    apt-fast autoclean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# ------ PART 3: offline installs ------

COPY ${PWD}/installers /tmp/installers
WORKDIR /tmp/installers

# install tini:
RUN dpkg -i tini.deb && \
    apt-get clean

# install anaconda:
RUN /bin/bash anaconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN rm -rf /tmp/installers

# ------ PART 4: set up VNC servers ------

# config desktop & VNC servers:
COPY image /

EXPOSE 80 5900 9001

# ------ PART 5: set up conda environments ------

WORKDIR /workspace

# keep conda updated to the latest version:
RUN conda update conda

# create environments for assignments:
COPY ${PWD}/environment environment

# the common package will be shared. no duplicated installation at all.
RUN conda env create -f environment/01-introduction.yaml && \
    conda env create -f environment/02-nearest-neighbor.yaml && \ 
    conda env create -f environment/03-clustering.yaml && \ 
    conda env create -f environment/07-feature-detection.yaml && \
    conda env create -f environment/08-feature-description.yaml && \ 
    conda env create -f environment/09-point-cloud-registration.yaml

ENTRYPOINT ["/startup.sh"]
