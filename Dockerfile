FROM jor115/neurodocker

COPY . /app

WORKDIR /tmp
ENTRYPOINT ["/bin/python3", "/app/t1_preproc.py"]
CMD [""]