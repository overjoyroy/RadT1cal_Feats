FROM jor115/neurodocker

COPY . /app

# Set FSL environment variables
ENV FSLDIR=/usr/local/fsl
ENV PATH=${FSLDIR}/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Source the FSL configuration script
RUN echo "source ${FSLDIR}/etc/fslconf/fsl.sh" >> /etc/profile


WORKDIR /tmp
ENTRYPOINT ["python3", "/app/t1_preproc.py"]
CMD [""]