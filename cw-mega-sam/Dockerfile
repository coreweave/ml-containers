ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY ./cuda124.patch /tmp/cuda124.patch
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG COMMIT
RUN git clone https://github.com/mega-sam/mega-sam && \
    cd mega-sam && \
    git checkout ${COMMIT} && \
    rm -rf .git

RUN patch -p1 /tmp/mega-sam < /tmp/cuda124.patch



ENTRYPOINT /tmp
CMD echo "Hello! You should only need to run python /tmp/mega-sam/base/setup.py install"


