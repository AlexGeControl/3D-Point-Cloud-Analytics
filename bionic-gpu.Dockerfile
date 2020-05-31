FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# ------ PART 0: set environment variables ------

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root SHELL=/bin/bash

USER root

# ------ PART 1: set CN package sources ------

# for Ubuntu:
COPY ${PWD}/image/etc/apt/cn-bionic-sources.list /etc/apt/sources.list
# COPY ${PWD}/image/etc/apt/sources.list.d/cn-bionic-cuda.list /etc/apt/sources.list.d/cuda.list
# COPY ${PWD}/image/etc/apt/sources.list.d/cn-bionic-nvidia-ml.list /etc/apt/sources.list.d/nvidia-ml.list
RUN rm -f /etc/apt/sources.list.d/*
# for Python: 
COPY ${PWD}/image/etc/pip.conf /root/.pip/pip.conf

# ------ PART 2: install Ubuntu packages ------

# install apt-fast:
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated software-properties-common axel aria2 && \
    add-apt-repository ppa:apt-fast/stable && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated apt-fast

# install packages:
ADD ${PWD}/image/etc/cn-bionic-apt-fast.conf /etc/apt-fast.conf
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
        gnuplot ghostscript \
        texlive-extra-utils texlive-latex-extra \
        cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libdw-dev \
        libpcl-dev && \
    apt-fast autoclean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# ------ PART 3: offline installs ------

ADD ${PWD}/installers /tmp/installers
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
ADD image /

# ------ PART 5: set up conda environments ------

WORKDIR /workspace

# keep conda updated to the latest version:
RUN conda update conda

# point cloud analytics:
# ADD /home/yaoge/Workspace/3d-point-cloud-analytics/workspace/assignments/project-01-kitti-detection-pipeline/environment/kitti-detection-pipeline.yaml /workspace/
#　RUN conda env create -f /workspace/kitti-detection-pipeline.yaml

EXPOSE 80 5900 9001

ENTRYPOINT ["/startup.sh"]
