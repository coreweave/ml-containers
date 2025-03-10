ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY ./cuda124.patch /tmp/cuda124.patch
COPY ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

ARG COMMIT
RUN cd /tmp && git clone --recursive https://github.com/mega-sam/mega-sam && \
    cd mega-sam && \
    git checkout ${COMMIT}
RUN cd /tmp/mega-sam && ls -la
RUN cd /tmp/mega-sam && patch -p2 < /tmp/cuda124.patch



ENTRYPOINT /tmp/mega-sam/base
CMD echo "Hello! You should only need to run python setup.py install from the /tmp/mega-sam/base directory."


