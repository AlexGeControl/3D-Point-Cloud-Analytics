FROM ubuntu:18.04

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root SHELL=/bin/bash

# install apt-fast:
ADD ${PWD}/image/etc/apt/sources.list.d/aliyun.list /etc/apt/sources.list.d/
RUN apt-get update --fix-missing && \
    apt-get -y install software-properties-common axel aria2 && \
    add-apt-repository ppa:apt-fast/stable && \
    apt-get update --fix-missing && \
    apt-get -y install apt-fast

# install packages:
ADD ${PWD}/image/etc/apt-fast.conf /etc/apt-fast.conf
RUN apt-fast update --fix-missing && \
    apt-fast install -y --no-install-recommends --allow-unauthenticated \
        curl grep sed dpkg wget bzip2 ca-certificates \
        git mercurial subversion \
        supervisor \
        openssh-server pwgen sudo vim-tiny \
        net-tools \
        lxde x11vnc xvfb \
        gtk2-engines-murrine ttf-ubuntu-font-family \
        firefox \
        nginx \
        python3-pip python3-dev build-essential \
        mesa-utils libgl1-mesa-dri \
        gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta arc-theme \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        dbus-x11 x11-utils \
        terminator && \
    apt-fast autoclean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# install anaconda:
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# install tini:
RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

# config desktop & VNC servers:
ADD image /

RUN cp /usr/share/applications/terminator.desktop /root/Desktop

# TODO: the downgrading of pip is caused by pyangbind-brcd==0.6.14
RUN pip install --upgrade pip pip-tools && \
    pip install setuptools wheel && \
    pip install -r /usr/lib/dev/requirements.txt
    # TODO: migrate to Python3
    # pip install -r /usr/lib/web/requirements.txt --force
    

# create conda environment for point cloud analysis:
WORKDIR /workspace
RUN conda env create -f environment.yml

EXPOSE 80 5900 9001

ENTRYPOINT ["/startup.sh"]
