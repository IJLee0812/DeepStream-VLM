FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Install Python GStreamer bindings
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-gi \
      python3-gst-1.0 \
      gstreamer1.0-python3-plugin-loader && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Override vLLM to 0.14.0 for NGC FP8 checkpoint compatibility
RUN pip install vllm==0.14.0 kafka-python PyYAML Pillow --ignore-installed

# Install NGC CLI
RUN cd /tmp && \
    wget -q --content-disposition \
      'https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.63.0/files/ngccli_linux.zip' \
      -O ngc_cli.zip && \
    unzip -q ngc_cli.zip -d /usr/local/ && \
    chmod +x /usr/local/ngc-cli/ngc && \
    ln -sf /usr/local/ngc-cli/ngc /usr/local/bin/ngc && \
    rm ngc_cli.zip

WORKDIR /workspace
