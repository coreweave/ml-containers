ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN mkdir -p /work
COPY ./cuda124.patch /work/cuda124.patch
COPY ./requirements.txt /work/requirements.txt

RUN pip install -r /work/requirements.txt
RUN rm /work/requirements.txt

ARG COMMIT
RUN cd /work && git clone --recursive https://github.com/mega-sam/mega-sam && \
    cd mega-sam && \
    git checkout ${COMMIT}
RUN cd /work/mega-sam && ls -la
RUN cd /work/mega-sam && patch -p2 < /work/cuda124.patch



ENTRYPOINT /work
CMD echo "Hello! You should only need to run python setup.py install from the /work/mega-sam/base directory."


