FROM rayproject/ray:2.34.0-py311-cu123

WORKDIR /code

# Project initialization:
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000 8265
# Clean up unnecessary files
RUN sudo apt-get purge -y --auto-remove git && \
    sudo rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# Copy Python code to the Docker image
COPY . /code
RUN sudo chown -R ray /code
RUN sudo chmod -R 700 /code