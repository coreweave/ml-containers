ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY ./cuda124.patch /tmp/cuda124.patch
COPY ./requirements.txt /tmp/requirements.txt

RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
RUN rm /tmp/requirements.txt

ARG COMMIT
RUN git clone https://github.com/mega-sam/mega-sam && \
    cd mega-sam && \
    git checkout ${COMMIT} && \
    rm -rf .git

RUN patch -p2 < /tmp/cuda124.patch



ENTRYPOINT /tmp/mega-sam/base
CMD echo "Hello! You should only need to run python setup.py install from the /tmp/mega-sam/base directory."


