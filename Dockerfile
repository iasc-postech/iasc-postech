# docker run -it --gpus all --shm-size 64g --name joong -v /home/joonghyuk/Code:/root/code -v /data0/joonghyuk:/root/data --workdir /root/code alex4727/experiment:cuda12.1.1 /bin/zsh

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Metadata indicating an image maintainer
LABEL maintainer="Joonghyuk Shin <joonghyuk4727@gmail.com>"

# Set non-interactive frontend (avoid apt-get prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Timezone setting
ENV TZ=Asia/Seoul

# PATH environment variable to include conda and misc scripts
ENV PATH=/opt/conda/bin:/root/miscs:$PATH

# Prepare the system and install dependencies
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y \
    curl ca-certificates git zsh wget tmux zip htop vim \
    tzdata build-essential p7zip-full freeglut3-dev libglu1-mesa-dev mesa-common-dev \
    libglib2.0-0 bzip2 libx11-6 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/* && \
    chsh -s $(which zsh) 

# Set the default shell to zsh for subsequent RUN commands
SHELL ["/bin/zsh", "-c"]

# Miscellaneous setup
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" && \
    curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/miscs/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ~/miscs/zsh-autosuggestions && \
    git clone https://github.com/tmux-plugins/tpm ~/miscs/.tmux/plugins/tpm

# ZSH syntax highlighting and autosuggestions setup
RUN echo "\n# ZSH syntax highlighting and autosuggestions" >> ~/.zshrc && \
    echo "ZSH_HIGHLIGHT_HIGHLIGHTERS=(main brackets pattern cursor)" >> ~/.zshrc && \
    echo "typeset -A ZSH_HIGHLIGHT_STYLES" >> ~/.zshrc && \
    echo "ZSH_HIGHLIGHT_STYLES[path]='fg=cyan'" >> ~/.zshrc && \
    echo "ZSH_HIGHLIGHT_STYLES[unknown-token]='fg=red'" >> ~/.zshrc && \
    echo "ZSH_HIGHLIGHT_STYLES[command]='fg=yellow,bold'" >> ~/.zshrc && \
    echo "\n# Terminal settings" >> ~/.zshrc && \
    echo "TERM=xterm-256color" >> ~/.zshrc && \
    echo "\n# Networking configuration" >> ~/.zshrc && \
    echo "MASTER_ADDR=localhost" >> ~/.zshrc && \
    echo "MASTER_PORT=31415" >> ~/.zshrc && \
    echo "\n# Source the syntax highlighting and autosuggestions" >> ~/.zshrc && \
    echo "source ~/miscs/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc && \
    echo "source ~/miscs/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc

# Copy configuration files from host to container
COPY configs/.tmux.conf /root/.tmux.conf
COPY configs/.vimrc /root/.vimrc
COPY configs/.p10k.zsh /root/.p10k.zsh

# Run tmux and vim installations in non-interactive mode
RUN tmux start-server && tmux new-session -d && tmux kill-session && \
    vim +'PlugInstall --sync' +qa && \
    echo "colorscheme gruvbox" >> ~/.vimrc && \
    echo "\n# Powerlevel10k theme" >> ~/.zshrc && \
    echo "[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh" >> ~/.zshrc

# Install Anaconda
ARG ANACONDA_VERSION=2023.09-0
RUN wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && \
    ./anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh

# Initialize conda in .bashrc and .zshrc
RUN echo "\n# Initialize conda" >> ~/.zshrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.zshrc && \
    echo "conda activate base" >> ~/.zshrc

# Define default command to run when starting the container
CMD ["zsh"]