ARG tag='latest'
FROM docker.greenkeytech.com/base:$tag

COPY . /fastText

WORKDIR /fastText
RUN make -j `cat /proc/cpuinfo | grep -c processor`

