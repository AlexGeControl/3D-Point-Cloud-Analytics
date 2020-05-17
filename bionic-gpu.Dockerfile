FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root SHELL=/bin/bash

# set CN package sources:
COPY ${PWD}/image/etc/apt/cn-bionic-sources.list /etc/apt/sources.list
COPY ${PWD}/image/etc/apt/sources.list.d/cn-bionic-cuda.list /etc/apt/sources.list.d/cuda.list
COPY ${PWD}/image/etc/apt/sources.list.d/cn-bionic-nvidia-ml.list /etc/apt/sources.list.d/nvidia-ml.list

# install apt-fast:
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated software-properties-common axel aria2 && \
    add-apt-repository ppa:apt-fast/stable && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated apt-fast

# install packages:
ADD ${PWD}/image/etc/apt-fast.conf /etc/apt-fast.conf
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
        cmake build-essential \
        libcupti-dev cuda-10-1 libcudnn7=7.6.4.38-1+cuda10.1 libcudnn7-dev=7.6.4.38-1+cuda10.1 \
        gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        dbus-x11 x11-utils \
        terminator \
        texlive-latex-extra \
        cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libdw-dev \
        libpcl-dev && \
    apt-fast autoclean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# config desktop & VNC servers:
ADD image /

USER root
RUN cp /usr/share/applications/terminator.desktop /root/Desktop

# set CN package sources: 
COPY ${PWD}/image/etc/pip.conf /root/.pip/pip.conf
# TODO: the downgrading of pip is caused by pyangbind-brcd==0.6.14
RUN pip install --upgrade pip && \
    pip install --upgrade pip-tools setuptools wheel && \
    pip install -r /usr/lib/dev/requirements.txt
    # TODO: migrate to Python3
    # pip install -r /usr/lib/web/requirements.txt --force

# install tini:
RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

# install anaconda:
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# create conda environment for point cloud analysis:
WORKDIR /workspace
RUN conda update -n base -c defaults conda && \
    conda env create -f gpu.yaml

EXPOSE 80 5900 9001

ENTRYPOINT ["/startup.sh"]
