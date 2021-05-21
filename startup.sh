FROM tiangolo/uwsgi-nginx:python3.8-alpine-2020-12-19

RUN apk update
RUN apk add lapack-dev

RUN apk add gfortran musl-dev g++ freetype-dev
RUN apk add gcc clang cmake python-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN apk add make automake g++ subversion python3-dev

LABEL Name=ydemo Version=0.0.1
EXPOSE 8000

ENV LISTEN_PORT=8000

ENV UWSGI_INI uwsgi.ini

WORKDIR /app

ADD . /app

RUN chmod g+w /app
RUN chmod g+w /app/db.sqlite3
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN apk add --no-cache jpeg-dev zlib-dev
RUN apk add --no-cache postgresql-dev
RUN apk add --no-cache libmemcached-dev zlib-dev 
RUN apk add --no-cache --virtual .build-deps build-base linux-headers \
    && pip3 install pip --upgrade 
RUN python3 -m pip install Pillow
RUN python3 -m pip install setuptools wheel
RUN pip3 install --upgrade pip setuptools wheel
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install cmake
RUN python3 -m pip install dlib
RUN python3 -m pip install numpy

RUN python3 -m pip install opencv-python
COPY . /app
